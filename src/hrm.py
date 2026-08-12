"""HRM-inspired learning governor (Phase 2) — "intent" network.

Architecture reference: architecture_v2.md §132 — HRM-in-the-node learning
governor. A small recursive intent network is trained FIRST (see
meta_pretrain.py) to decide how the host model's parameters should change:
M = f_theta(x, s),  W' = W - lr * M * grad(W).

The intent network outputs one update gate per parameter MODULE (module-level
granularity, §132.2 starting point). It is frozen during the measured
continual stream (§132.3: frozen governance core + mutable knowledge).

Hooks mirror the Phase-2 integration interface (§17): the stream calls
controller.control_update(...) between loss.backward() and optimizer.step().
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ParamGroup:
    """One controllable parameter module (e.g. a Linear's weight tensor)."""
    name: str
    param: nn.Parameter
    indices: slice  # reserved for future per-weight granularity

    @property
    def size(self) -> int:
        return self.param.numel()


def build_module_groups(model: nn.Module) -> list[ParamGroup]:
    """One control group per leaf parameter tensor (module-level granularity)."""
    groups = []
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups.append(ParamGroup(name=name, param=p, indices=slice(0, p.numel())))
            total += p.numel()
    return groups


def compute_governor_features(
    x: torch.Tensor,
    y: torch.Tensor,
    loss: torch.Tensor,
    model: nn.Module,
    groups: list[ParamGroup],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """State features s for the governor.

    Returns:
        group_feats: (num_groups, 5) per-module statistics.
        global_feats: (9,) task/batch-level statistics.
    """
    eps = 1e-8
    grads = [g.param.grad.detach().flatten() for g in groups
             if g.param.grad is not None]
    params = [g.param.detach().flatten() for g in groups]

    g_all = torch.cat(grads)
    p_all = torch.cat(params)
    g_mean = g_all.abs().mean()
    p_mean = p_all.abs().mean()
    gp_all = torch.cat([gi * pi for gi, pi in zip(grads, params)])
    gp_mean = gp_all.abs().mean()

    group_feats = []
    for g, p in zip(grads, params):
        f0 = g.abs().mean() / (g_mean + eps)            # relative grad magnitude
        f1 = p.abs().mean() / (p_mean + eps)            # relative param scale
        f2 = (g * p).abs().mean() / (gp_mean + eps)     # grad/param alignment
        f3 = torch.log1p(g.abs().mean())                # absolute grad level
        f4 = torch.log1p((g ** 2).mean())               # gradient sharpness
        group_feats.append(torch.stack([f0, f1, f2, f3, f4]))
    group_feats = torch.stack(group_feats).to(device)   # (G, 5)

    # Batch geometry hints (task-agnostic: no task-ID ever reaches the model).
    xb = x.detach()
    global_feats = torch.tensor([
        torch.log1p(loss.detach()),
        torch.log1p(g_mean),
        torch.log1p(p_mean),
        xb[:, 0].mean(),
        xb[:, 1].mean(),
        (xb[:, 0] ** 2 + xb[:, 1] ** 2).mean(),
        xb[:, 0].std(),
        xb[:, 1].std(),
        y.detach().mean(),
    ], device=device)

    return group_feats, global_feats


class HRMIntentGovernor(nn.Module):
    """Tiny recursive intent network: batch features -> per-module update gates.

    Architecture (HRM-inspired, "recursive refinement of a decision state"):
        per-group embeddings -> GRU refinement passes over the group sequence
        -> final per-group gate logits -> sigmoid masks M in (0, 1).
    """

    def __init__(
        self,
        num_groups: int,
        group_feat_dim: int = 5,
        global_feat_dim: int = 9,
        hidden_dim: int = 20,
        refine_steps: int = 2,
        init_mask: float = 0.9,
    ) -> None:
        super().__init__()
        self.num_groups = num_groups

        self.mlp_g = nn.Sequential(
            nn.Linear(group_feat_dim, hidden_dim), nn.ReLU())
        self.mlp_c = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.mlp_m = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

        # Start near-identity gates (m ~ init_mask) so meta-training begins
        # from baseline-like behavior, then specializes.
        if init_mask is not None:
            b = math.log(init_mask / (1.0 - init_mask))
            with torch.no_grad():
                self.mlp_m[-1].bias.fill_(b)

    def forward(
        self,
        group_feats: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Returns masks of shape (num_groups,) in (0, 1)."""
        g = self.mlp_g(group_feats)          # (G, h)
        c = self.mlp_c(global_feats.unsqueeze(0)).squeeze(0)  # (h,)

        h = c.repeat(self.num_groups, 1)
        for _ in range(self.refine_steps):
            h = self.gru(g, h)               # refine decision state recursively
        gates = self.mlp_m(torch.cat([h, c.repeat(self.num_groups, 1)], dim=-1))
        return torch.sigmoid(gates.squeeze(-1))

    @staticmethod
    def mask_entropy(m: torch.Tensor) -> torch.Tensor:
        """Entropy of gates (regularizer encouraging decisive 0/1 masks)."""
        m = m.clamp(1e-6, 1 - 1e-6)
        return -(m * torch.log(m) + (1 - m) * torch.log(1 - m)).mean()

    def governor_params(self) -> int:
        """Number of parameters of the intent network itself."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HRMController:
    """Phase-2 controller wrapping the frozen intent governor (§17 hooks).

    The baseline stream calls control_update() after loss.backward() and
    before optimizer.step(): gates are multiplied into the raw gradients.
    """

    def __init__(self, governor: HRMIntentGovernor, groups: list[ParamGroup],
                 device: torch.device) -> None:
        self.governor = governor
        self.groups = groups
        self.device = device

    @torch.no_grad()
    def control_update(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        """Gate the accumulated gradients in place; returns the masks."""
        group_feats, global_feats = compute_governor_features(
            x, y, loss, model, self.groups, self.device)
        masks = self.governor(group_feats, global_feats)

        for group, m in zip(self.groups, masks):
            if group.param.grad is not None:
                group.param.grad.mul_(float(m))
        return masks

    def mask_stats(self) -> dict[str, float]:
        """Summary of current gate statistics (diagnostics)."""
        total = self.governor.num_groups
        return {"num_groups": total}


def measure_update_fraction(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
    threshold: float = 1e-3,
) -> float:
    """§132.10 UpdateFraction: fraction of params substantially changed vs
    a phase-start snapshot (relative change |dp|/(|p|+eps) > threshold)."""
    eps = 1e-8
    changed = 0
    total = 0
    for name, p in model.named_parameters():
        if name not in snapshot:
            continue
        rel = (p.detach() - snapshot[name]).abs() / (snapshot[name].abs() + eps)
        changed += (rel > threshold).sum().item()
        total += p.numel()
    return 100.0 * changed / total if total else 0.0