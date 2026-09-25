"""Instant learning — detached adapter experts.

Path (roadmap #3): base forward (no_grad) -> cached hidden h_L ->
train ONLY a small adapter module on h_L -> backward never touches the
1.17B base. Sub-second installs, zero forgetting by construction
(each expert is its own module; base weights never change).

Placement: adapter is a residual bottleneck on the FINAL hidden state
(right before lm_head), so a forward_pre_hook on lm_head applies it to
every generate() step without touching FFN weights or masks.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# rank grid for adapter bottleneck (doubled on probe failure)
RANK_GRID = (16, 32, 64, 128, 256, 512, 1024, 2048)
# V4-equivalent neuron accounting: FFN neuron = 3*hidden weights,
# adapter = 2*hidden*rank weights -> equiv = 2*rank/3
def equiv_neurons(hidden: int, rank: int) -> int:
    return max(1, (2 * hidden * rank) // (3 * hidden))


def next_rank(r: int) -> int:
    for g in RANK_GRID:
        if g > r:
            return g
    return RANK_GRID[-1]


class AdapterExpert(nn.Module):
    """Residual bottleneck on hidden state. fp32 params, bf16 I/O."""

    def __init__(self, hidden: int, rank: int):
        super().__init__()
        self.hidden = hidden
        self.rank = rank
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)  # identity at init

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h32 = h.float()
        out = h32 + self.up(F.silu(self.down(h32)))
        return out.to(h.dtype)


class AdapterCtx:
    """Apply one adapter as lm_head pre-hook for the duration of a block."""

    def __init__(self, model, adapter: AdapterExpert):
        self.model = model
        self.adapter = adapter
        self.handle = None

    def __enter__(self):
        ad = self.adapter

        def hook(mod, args):
            if not args:
                return args
            return (ad(args[0]),)

        self.handle = self.model.lm_head.register_forward_pre_hook(hook)
        return self

    def __exit__(self, *a):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


@torch.no_grad()
def cache_hidden(model, data: dict) -> torch.Tensor:
    """One base forward for the whole QA set; hiddens reused for all epochs."""
    dev = next(model.parameters()).device
    out = model.model(
        input_ids=data["input_ids"].to(dev),
        attention_mask=data["attention_mask"].to(dev),
    )
    return out.last_hidden_state.detach()


def train_instant(
    model,
    adapter: AdapterExpert,
    data: dict,
    epochs: int = 40,
    lr: float = 5e-3,
    batch: int = 4,
    device: str = "cuda",
) -> float:
    """Train ONLY the adapter on cached base hiddens. Base graph never built.

    Supervised-position gather: lm_head is per-position, so computing logits
    only where labels != -100 is exactly equivalent to full-sequence CE
    (ignore_index elsewhere) but skips ~99% of the vocab matmul.
    """
    h = cache_hidden(model, data)
    labels = data["labels"].to(h.device)
    n = h.size(0)
    opt = torch.optim.SGD(adapter.parameters(), lr=lr, momentum=0.9)
    adapter.train()
    final = 0.0
    # predict token t+1 from hidden at t
    sup = labels[:, 1:] != -100  # [N, T-1]
    h_src = h[:, :-1]            # [N, T-1, H]
    y_all = labels[:, 1:]
    for _ in range(epochs):
        perm = torch.randperm(n)
        el = []
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            m = sup[idx]                              # [B, T-1]
            hs = h_src[idx][m]                        # [Nsup, H]
            ys = y_all[idx][m]                        # [Nsup]
            if hs.numel() == 0:
                continue
            had = adapter(hs)                         # fp32 internal, bf16 out
            logits = model.lm_head(had)               # frozen weights, grads flow to adapter
            loss = F.cross_entropy(logits.float(), ys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            el.append(float(loss.item()))
        if not el:
            break
        final = sum(el) / len(el)
    adapter.eval()
    return final


def train_instant_timed(*args, **kwargs) -> tuple[float, float]:
    t0 = time.time()
    loss = train_instant(*args, **kwargs)
    return loss, time.time() - t0
