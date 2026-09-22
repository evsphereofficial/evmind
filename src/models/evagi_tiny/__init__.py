"""EvAGI Tiny Conversational LLM — from scratch with separated experts.

Architecture: GPT-Neo style with pre-allocated expert FFN neurons.
Dataset: HuggingFaceTB/everyday-conversations-llama3.1-2k
Stack: WeightRegister, HRM Router, Per-Expert Governors, ExpertMaskContext.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class EvagiTinyConfig:
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 192
    n_ctx: int = 256
    # Expert allocation per layer
    expert_fracs: dict = None  # expert_id -> fraction of FFN neurons
    dropout: float = 0.0

    def __post_init__(self):
        if self.expert_fracs is None:
            # Default: 4 experts per layer
            # Expert 0: general (40%), Expert 1: facts (20%), Expert 2: reasoning (20%), Expert 3: reserved (20%)
            self.expert_fracs = {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2}
        self.n_embd = (self.n_embd // self.n_head) * self.n_head  # ensure divisible
        self.inter_size = self.n_embd * 4  # FFN intermediate size


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.n_ctx, config.n_ctx))
                                     .view(1, 1, config.n_ctx, config.n_ctx))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class EvagiExpertFFN(nn.Module):
    """FFN with pre-allocated expert neurons.
    Each expert owns specific neuron indices in the shared FFN weight matrix.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_embd = config.n_embd
        self.inter_size = config.inter_size
        self.n_experts = len(config.expert_fracs)

        # Shared weight matrix — experts own different ROWS
        self.c_fc = nn.Linear(config.n_embd, config.inter_size)
        self.c_proj = nn.Linear(config.inter_size, config.n_embd)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.dropout)

        # Compute expert neuron assignments
        self.expert_masks = {}
        self.expert_neuron_counts = {}
        assigned = 0
        for eid, frac in config.expert_fracs.items():
            count = int(config.inter_size * frac)
            mask = torch.zeros(config.inter_size, dtype=torch.bool)
            mask[assigned:assigned + count] = True
            self.expert_masks[eid] = mask
            self.expert_neuron_counts[eid] = count
            assigned += count
        # Remaining neurons go to expert 0
        if assigned < config.inter_size:
            remaining = config.inter_size - assigned
            self.expert_masks[0][assigned:] = True
            self.expert_neuron_counts[0] += remaining

    def forward(self, x, expert_mask=None):
        h = self.c_fc(x)
        if expert_mask is not None:
            h = h * expert_mask.float()
        h = self.act(h)
        h = self.drop(h)
        h = self.c_proj(h)
        return h


class EvagiTransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = EvagiExpertFFN(config, layer_idx)
        self.layer_idx = layer_idx

    def forward(self, x, expert_mask=None):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x), expert_mask=expert_mask)
        return x


class EvagiTinyLM(nn.Module):
    """EvAGI Tiny Conversational LLM with pre-allocated expert neurons."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            EvagiTransformerBlock(config, i) for i in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Print expert allocation summary
        total = 0
        for eid, frac in config.expert_fracs.items():
            count = int(config.inter_size * frac)
            total += count * config.n_layer
            print(f"  Expert {eid}: {count} neurons/layer, {count * config.n_layer} total, "
                  f"{count * 257 * config.n_layer:,} weights")
        print(f"  Total FFN neurons: {config.inter_size * config.n_layer}")
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {n_params:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def get_expert_masks(self, device=None):
        """Get per-layer, per-expert boolean masks."""
        if device is None:
            device = next(self.parameters()).device
        masks = {}
        for eid in self.config.expert_fracs:
            layer_masks = []
            for block in self.blocks:
                mask = block.ffn.expert_masks[eid].clone().to(device)
                layer_masks.append(mask)
            masks[eid] = layer_masks
        return masks

    def forward(self, idx, targets=None, expert_mask=None):
        B, T = idx.size()
        assert T <= self.config.n_ctx
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, expert_mask=expert_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=0.8, top_p=0.9,
                 expert_mask=None, pad_token_id=None):
        """Autoregressive generation."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context size
            idx_cond = idx if idx.size(1) <= self.config.n_ctx else idx[:, -self.config.n_ctx:]
            logits, _ = self(idx_cond, expert_mask=expert_mask)
            logits = logits[:, -1, :] / temperature

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if pad_token_id is not None and idx_next.item() == pad_token_id:
                break
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
