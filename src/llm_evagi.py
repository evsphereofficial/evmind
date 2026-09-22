"""EvAGI LLM components: neuron register, experts, governors, support router.

Maps the v2 capacity law onto GPT-Neo FFN neurons (c_fc rows / c_proj cols):
  Register  — per-layer occupancy of the 12288-neuron FFN pool
  Expert    — disjoint neuron set sized by predict_required_weights()
  Governor  — tiny per-expert plasticity gates (HRM-style) over owned neurons
  Router    — live support-set g(x): pick expert by min loss on 32 shots
              (no task_id at test)

Hard isolation: during task t forward, only expert t's neurons are active
(non-owned FFN channels zeroed via hooks); only those receive gradients.
Everything else (attention, embeddings, LN, non-owned FFN) stays frozen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .registry import predict_required_weights


# ---------------------------------------------------------------------------
# Neuron <-> law conversion (fix 1)
# ---------------------------------------------------------------------------
# Law k is calibrated in scalar-weight units on the 17K 2D model.
# One GPT-Neo FFN neuron ≈ (hidden + 1 + hidden) = 2*hidden+1 scalar params.
# Using the raw law (k/1536) yields <1 neuron/task — unusable.
# neuron_div maps law-k -> neurons. Default 8 => ~700 neurons total (5.7% of
# 12288), enough for yes/no skills while staying a small expert budget.
DEFAULT_NEURON_DIV = 8


def k_to_neuron_counts(
    k_alloc: int,
    n_layers: int,
    neuron_div: int = DEFAULT_NEURON_DIV,
    min_total: int = 8,
) -> list[int]:
    """Distribute law-k allocation across layers as neuron counts."""
    n_total = max(min_total, int(math.ceil(k_alloc / max(1, neuron_div))))
    base = n_total // n_layers
    rem = n_total % n_layers
    return [base + (1 if i < rem else 0) for i in range(n_layers)]


def predict_neurons(task_name: str, n_layers: int, neuron_div: int = DEFAULT_NEURON_DIV,
                    target_acc: float = 98.0) -> tuple[int, list[int]]:
    """v2 law -> (k_alloc, per-layer neuron counts)."""
    k = predict_required_weights(task_name, target_acc=target_acc)
    counts = k_to_neuron_counts(k, n_layers, neuron_div=neuron_div)
    return k, counts


# ---------------------------------------------------------------------------
# Register: per-layer occupancy over the FFN neuron pool
# ---------------------------------------------------------------------------
class NeuronRegister:
    """Tracks which FFN neurons are owned by which expert/task."""

    def __init__(self, inters: list[int], device: torch.device):
        self.inters = list(inters)
        self.n_layers = len(inters)
        self.device = device
        self.occupied = [
            torch.zeros(inter, dtype=torch.bool, device=device) for inter in inters
        ]
        self.ownership = [
            torch.full((inter,), -1, dtype=torch.long, device=device)
            for inter in inters
        ]
        self.allocations: list[dict] = []

    def allocate(self, task_name: str, counts: list[int], expert_id: int) -> list[torch.Tensor]:
        """Carve out free neurons per layer; returns per-layer Bool masks."""
        masks: list[torch.Tensor] = []
        for li, take in enumerate(counts):
            occ = self.occupied[li]
            free = torch.where(~occ)[0]
            t = min(int(take), free.numel())
            chosen = free[:t]
            mask = torch.zeros(self.inters[li], dtype=torch.bool, device=self.device)
            mask[chosen] = True
            occ[chosen] = True
            self.ownership[li][chosen] = expert_id
            masks.append(mask)
        n_owned = int(sum(m.sum().item() for m in masks))
        self.allocations.append(
            {"task": task_name, "expert_id": expert_id, "counts": [int(c) for c in counts],
             "owned": n_owned}
        )
        return masks

    def total_occupied(self) -> int:
        return int(sum(o.sum().item() for o in self.occupied))

    def total_pool(self) -> int:
        return int(sum(self.inters))

    def summary(self) -> dict:
        return {
            "pool": self.total_pool(),
            "occupied": self.total_occupied(),
            "allocations": self.allocations,
        }


# ---------------------------------------------------------------------------
# Hard-isolation forward hooks: zero non-owned FFN channels
# ---------------------------------------------------------------------------
class ExpertMaskContext:
    """Context manager zeroing non-active FFN channels (binary) and/or
    scaling active channels by differentiable governor gates.

    masks_per_expert[eid][li] -> Bool (inter,)
    gates: optional per-layer (inter,) float gates for the active expert only
           (non-owned entries must already be 0). When set, gates replace the
     binary mask so task loss can train TinyPerExpertGovernor.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        masks_per_expert: list[list[torch.Tensor]],
        expert_ids: list[int],
        gates: list[torch.Tensor] | None = None,
    ):
        self.layers = layers
        self.masks_per_expert = masks_per_expert
        self.expert_ids = expert_ids
        self.gates = gates
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _combined(self, layer_idx: int) -> torch.Tensor:
        inter = self.masks_per_expert[0][layer_idx].numel()
        acc = torch.zeros(
            inter,
            dtype=torch.bool,
            device=self.masks_per_expert[0][layer_idx].device,
        )
        for eid in self.expert_ids:
            acc |= self.masks_per_expert[eid][layer_idx]
        return acc

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            binary = self._combined(li)
            gate = self.gates[li] if self.gates is not None else None

            def make_fc_hook(m_mask, g):
                def hook(module, inputs, output):
                    out = output * 1.0  # keep graph if g requires grad
                    if g is not None:
                        out = out * g.to(out.dtype)
                    out = out.masked_fill(~m_mask, 0)
                    return out
                return hook

            self._handles.append(
                layer.mlp.c_fc.register_forward_hook(make_fc_hook(binary, gate))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def hard_mask_grads(layers: nn.ModuleList, masks: list[torch.Tensor]) -> None:
    """Zero gradients outside the active expert's neurons (fix 3)."""
    for li, layer in enumerate(layers):
        m = masks[li]
        fc_w = layer.mlp.c_fc.weight
        if fc_w.grad is not None:
            # rows = neurons
            grad = fc_w.grad
            grad[~m, :] = 0
        if layer.mlp.c_fc.bias is not None and layer.mlp.c_fc.bias.grad is not None:
            layer.mlp.c_fc.bias.grad[~m] = 0
        cp_w = layer.mlp.c_proj.weight
        if cp_w.grad is not None:
            # cols = neurons
            cp_w.grad[:, ~m] = 0


def set_ffn_trainable(layers: nn.ModuleList, masks: list[torch.Tensor] | None) -> None:
    """Freeze entire model except FFN; if masks given, only owned rows trainable
    via requires_grad on full tensors + hard grad mask after backward."""
    for layer in layers:
        layer.mlp.c_fc.weight.requires_grad_(True)
        if layer.mlp.c_fc.bias is not None:
            layer.mlp.c_fc.bias.requires_grad_(True)
        layer.mlp.c_proj.weight.requires_grad_(True)


def freeze_all_but_ffn(model: nn.Module, layers: nn.ModuleList) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in layers:
        layer.mlp.c_fc.weight.requires_grad_(True)
        if layer.mlp.c_fc.bias is not None:
            layer.mlp.c_fc.bias.requires_grad_(True)
        layer.mlp.c_proj.weight.requires_grad_(True)


# ---------------------------------------------------------------------------
# Tiny per-expert HRM governor (plasticity gates over owned neurons)
# ---------------------------------------------------------------------------
class TinyPerExpertGovernor(nn.Module):
    """Tiny gate net: per-neuron features -> sigmoid plasticity in (0,1].

    Amortized like HRMIntentGovernor: shared 8->hidden->1 MLP, so the governor
    stays ~100-300 params while controlling all of its expert's neurons.
    Init bias high so gates start ~open (1.0); light L1 encourages selectivity.
    """

    def __init__(self, hidden: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # start near open
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 4.0)  # sigmoid(4)≈0.982

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (N, 8) -> gates (N,) in (0,1)."""
        return torch.sigmoid(self.net(feats).squeeze(-1))


def neuron_features(
    weights: torch.Tensor,
    grads: torch.Tensor | None,
    mask: torch.Tensor,
    layer_idx: int,
    n_layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (N, 8) features for governor over owned neurons.

    weights/grads: c_fc.weight (inter, hidden) — row i is neuron i.
    Returns (feats, idx) where idx are owned neuron indices.
    """
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return torch.zeros(0, 8, device=mask.device), idx
    w_rows = weights[idx]  # (N, hidden)
    if grads is not None:
        g_rows = grads[idx]
    else:
        g_rows = torch.zeros_like(w_rows)
    n = mask.numel()
    pos = idx.float() / max(n - 1, 1)
    layer_f = torch.full_like(pos, layer_idx / max(n_layers - 1, 1))
    owned = torch.ones_like(pos)
    feats = torch.stack(
        [
            torch.log1p(w_rows.abs().mean(dim=-1)),
            torch.log1p(g_rows.abs().mean(dim=-1)),
            pos,
            layer_f,
            owned,
            torch.tanh(w_rows.mean(dim=-1)),
            torch.tanh(g_rows.mean(dim=-1)),
            w_rows.std(dim=-1),
        ],
        dim=-1,
    )
    return feats, idx


# ---------------------------------------------------------------------------
# Live support-set router (fix 2): no task_id
# ---------------------------------------------------------------------------
@torch.no_grad()
def support_pick_expert(
    model: nn.Module,
    layers: nn.ModuleList,
    masks_per_expert: list[list[torch.Tensor]],
    tokenizer,
    pairs: list[tuple[str, int]],
    device,
    max_length: int,
) -> tuple[int, list[float]]:
    """Pick expert with min yes/no NLL on the support set under each mask."""
    from .llm_tasks import tokenize_pairs, yes_no_token_ids

    yes_id, no_id = yes_no_token_ids(tokenizer)
    enc = tokenize_pairs(tokenizer, pairs, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention = enc["attention_mask"].to(device)
    labels = enc["labels"]
    first_lab = (labels != -100).float().argmax(dim=1).to(device)
    b_idx = torch.arange(input_ids.size(0), device=device)
    pred_pos = (first_lab - 1).clamp(min=0)
    true = torch.tensor([lab for _, lab in pairs], device=device)

    losses: list[float] = []
    model.eval()
    for eid in range(len(masks_per_expert)):
        with ExpertMaskContext(layers, masks_per_expert, [eid]):
            out = model(input_ids=input_ids, attention_mask=attention)
        logits = out.logits[b_idx, pred_pos, :]  # (B, V)
        # NLL of the true yes/no token
        target = torch.where(true.bool(), yes_id, no_id)
        nll = torch.nn.functional.cross_entropy(logits, target)
        losses.append(float(nll.item()))
    best = int(min(range(len(losses)), key=lambda i: losses[i]))
    return best, losses


class HRMRouter(nn.Module):
    """Main HRM router: global batch features -> expert logits (9 -> h -> E)."""

    def __init__(self, num_experts: int, hidden: int = 16, feat_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_experts),
        )
        self.num_experts = num_experts

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return self.net(feats)


def batch_global_feats(x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """9 global features for the main router (input-conditional, no task_id)."""
    with torch.no_grad():
        p = torch.sigmoid(logits.view(-1))
        loss_ = torch.nn.functional.binary_cross_entropy(
            p.clamp(1e-6, 1 - 1e-6), y.view(-1)
        )
        feats = torch.tensor(
            [
                x.mean().item(),
                x.std().item(),
                x.abs().mean().item(),
                y.float().mean().item(),
                loss_.item(),
                p.mean().item(),
                p.std().item(),
                (p > 0.5).float().mean().item(),
                float(x.shape[0]),
            ],
            device=x.device,
        )
    return feats
