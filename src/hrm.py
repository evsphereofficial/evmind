"""HRM-inspired learning governor (Phase 2) — "intent" network.

Architecture reference: architecture_v2.md §132 — HRM-in-the-node learning
governor. A small intent network is trained FIRST (see meta_pretrain.py) to
decide how the host model's parameters should change:

    M_i = gate for weight i,   W' = W - lr * M * grad(W)

Granularity:
- "weight": one gate per PARAMETER (17,249 gates for the 17K base model).
  Gates come from a tiny SHARED gate network over per-weight features
  [log1p(|grad|), log1p(|weight|), position-in-module, module id] + global
  task/batch context. The gates are amortized, so the intent network stays
  tiny (~2-4K params) while controlling every single weight.
- "module": one gate per parameter tensor (coarser reference point, §132.2).

The intent network is FROZEN during the measured continual stream (frozen
governance core, §132.3). Hooks mirror the Phase-2 interface (§17): the
stream calls controller.control_update(...) between loss.backward() and
optimizer.step().
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
    indices: slice  # reserved

    @property
    def size(self) -> int:
        return self.param.numel()


def build_module_groups(model: nn.Module) -> list[ParamGroup]:
    """One control group per leaf parameter tensor."""
    groups = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups.append(ParamGroup(name=name, param=p, indices=slice(0, p.numel())))
    return groups


def compute_global_features(
    x: torch.Tensor,
    y: torch.Tensor,
    loss: torch.Tensor,
    g_all_flat: torch.Tensor,
    p_all_flat: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Task/batch-level context vector for the governor (9 features)."""
    eps = 1e-8
    xb = x.detach()
    return torch.tensor([
        torch.log1p(loss.detach()),
        torch.log1p(g_all_flat.abs().mean().detach() + eps),
        torch.log1p(p_all_flat.abs().mean().detach() + eps),
        xb[:, 0].mean(),
        xb[:, 1].mean(),
        (xb[:, 0] ** 2 + xb[:, 1] ** 2).mean(),
        xb[:, 0].std(),
        xb[:, 1].std(),
        y.detach().mean(),
    ], device=device)


class HRMIntentGovernor(nn.Module):
    """Tiny intent network: parameter/gradient state -> per-weight update gates.

    Design:
        * per-weight features: [log1p(|g|), log1p(|p|), position_in_module,
          module_id(one-hot)]  (module_id + position = structural identity,
          NOT task identity -- no task-ID ever reaches it)
        * + global context (9) broadcast to every weight
        * shared MLP -> gate logits -> sigmoid masks M in (0, 1)
        * recursive refinement: refine_steps passes, each re-feeding the mean
          gate of the previous pass (HRM-style iterative decision refinement)

    Total masks = total parameter count (e.g. 17,249 for the base model).
    """

    def __init__(
        self,
        num_groups: int,
        granularity: str = "weight",
        hidden_dim: int = 24,
        refine_steps: int = 2,
        init_mask: float = 0.5,
        global_feat_dim: int = 9,
        per_weight_feat_dim: int = 7,
        # log1p|gB|, log1p|W|, log1p|gA|, cos(gA,gB), sign(gA.gB)log1p|gA.gB|,
        # rel-change history, position (onehot+ctx appended)
    ) -> None:
        super().__init__()
        assert granularity in ("weight", "module")
        self.granularity = granularity
        self.num_groups = num_groups
        self.refine_steps = refine_steps

        # input width: plus 1 for the recursive refinement channel (prev mean gate)
        w_in = per_weight_feat_dim + num_groups + global_feat_dim + 1
        # module: group stats + context
        m_in = 5 + global_feat_dim + 1
        in_dim = w_in if granularity == "weight" else m_in

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.per_weight_feat_dim = per_weight_feat_dim
        self.global_feat_dim = global_feat_dim

        if init_mask is not None:
            b = math.log(init_mask / (1.0 - init_mask))
            with torch.no_grad():
                self.mlp[-1].bias.fill_(b)

    # -- main API -------------------------------------------------------------
    def gate(
        self,
        p_list: list[torch.Tensor],
        g_list: list[torch.Tensor],
        global_feats: torch.Tensor,
        g_old_list: list[torch.Tensor] | None = None,
        hist_list: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Return one gate tensor per group, shapes matching p_list.

        Per-weight gradient RELATIONSHIP features (the heart of selective
        plasticity):
            cos(gA, gB): aligned (+1) -> updating helps both -> open;
                         conflicting (-1) -> updating harms old -> close
            gA, gB magnitudes: gB strong + gA weak = free capacity -> open
            sign(gA.gB)*log1p|gA.gB|: raw conflict magnitude
            rel-change history: |p - p0|/|p0| so far (already-touched weights)
        g_old_list: per-weight gradient of OLD-task loss w.r.t. params before
        the burst (task-A sensitivity / EWC diagonal).

        Fully vectorized over all controlled weights (single MLP pass).
        """
        sizes = [p.numel() for p in p_list]
        G = len(sizes)
        N = sum(sizes)
        p_all = torch.cat([p.flatten() for p in p_list])
        g_all = torch.cat([g.flatten() for g in g_list])
        eps = 1e-12
        if g_old_list is None:
            g_old_all = torch.zeros_like(g_all)
        else:
            g_old_all = torch.cat([g.detach().flatten() for g in g_old_list])
        if hist_list is None:
            hist_all = torch.zeros_like(g_all)
        else:
            hist_all = torch.cat([h.detach().flatten() for h in hist_list])

        gids = torch.repeat_interleave(torch.arange(G, device=p_all.device),
                                       torch.tensor(sizes, device=p_all.device))
        local_idx = torch.arange(N, device=p_all.device) - torch.cumsum(
            torch.tensor([0] + sizes, device=p_all.device)[:-1], 0)[gids]
        idx_frac = local_idx / torch.tensor(sizes, device=p_all.device)[gids].float()

        if self.granularity == "weight":
            onehot = torch.nn.functional.one_hot(gids, num_classes=G).float()
            ctx = global_feats.unsqueeze(0).expand(N, -1)
            denom = g_all.abs() * g_old_all.abs() + eps
            cos_align = (g_all * g_old_all) / denom        # [-1, 1]
            raw_dot = g_all * g_old_all
            feats = torch.cat([
                torch.log1p(g_all.abs() + eps).unsqueeze(-1),
                torch.log1p(p_all.abs() + eps).unsqueeze(-1),
                torch.log1p(g_old_all.abs() + eps).unsqueeze(-1),
                cos_align.unsqueeze(-1),
                (raw_dot.sign() * torch.log1p(raw_dot.abs() + eps)).unsqueeze(-1),
                torch.log1p(hist_all + eps).unsqueeze(-1),
                idx_frac.unsqueeze(-1),
                onehot,
                ctx,
            ], dim=-1)
        else:
            # module-level: segment-reduced stats per group (vectorized)
            counts = torch.zeros(G, device=p_all.device)
            counts.scatter_add_(0, gids, torch.ones_like(p_all))
            absg = torch.zeros(G, device=p_all.device)
            absg.scatter_add_(0, gids, g_all.abs())
            absp = torch.zeros(G, device=p_all.device)
            absp.scatter_add_(0, gids, p_all.abs())
            abgp = torch.zeros(G, device=p_all.device)
            abgp.scatter_add_(0, gids, (g_all * p_all).abs())
            gsq = torch.zeros(G, device=p_all.device)
            gsq.scatter_add_(0, gids, g_all ** 2)
            feats = torch.cat([
                (absg / counts).unsqueeze(-1),
                (absp / counts).unsqueeze(-1),
                (abgp / counts).unsqueeze(-1),
                torch.log1p(absg / counts).unsqueeze(-1),
                torch.log1p(gsq / counts).unsqueeze(-1),
                global_feats.unsqueeze(0).expand(G, -1),
            ], dim=-1)

        init_bias = self.mlp[-1].bias.detach().sigmoid() \
            if hasattr(self.mlp[-1], "bias") else 0.5
        prev_mean = torch.full((1,), float(init_bias), device=p_all.device)
        for _ in range(self.refine_steps):
            gate_logit = self.mlp(torch.cat([
                feats, prev_mean.expand(feats.shape[0], -1)], dim=-1))
            m = torch.sigmoid(gate_logit).squeeze(-1)
            prev_mean = m.mean().reshape(1)

        if self.granularity == "weight":
            return torch.split(m, list(sizes))
        return torch.split(m.unsqueeze(-1), [1] * G)

    def gate_from_model(
        self,
        model: nn.Module,
        groups: list[ParamGroup],
        x: torch.Tensor,
        y: torch.Tensor,
        loss: torch.Tensor,
        device: torch.device,
        g_old_list: list[torch.Tensor] | None = None,
        hist_list: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Gate tensors computed from a live model's params + .grad."""
        p_list = [g.param.detach().flatten() for g in groups]
        g_list = [g.param.grad.detach().flatten() for g in groups]
        global_feats = compute_global_features(
            x, y, loss,
            torch.cat(g_list), torch.cat(p_list), device)
        return self.gate(p_list, g_list, global_feats, g_old_list, hist_list)

    def gate_from_state(
        self,
        p_cur: dict[str, torch.Tensor],
        g_list: list[torch.Tensor],
        groups: list[ParamGroup],
        x: torch.Tensor,
        y: torch.Tensor,
        loss: torch.Tensor,
        device: torch.device,
        differentiable: bool = True,
        g_old_list: list[torch.Tensor] | None = None,
        hist_list: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Gate tensors from a functional parameter dict (meta-training path)."""
        p_list = [p_cur[g.name].detach().flatten() for g in groups]
        g_flat = [g.detach().flatten() for g in g_list]
        global_feats = compute_global_features(
            x, y, loss, torch.cat(g_flat), torch.cat(p_list), device)
        if differentiable:
            return self.gate(p_list, g_flat, global_feats, g_old_list, hist_list)
        with torch.no_grad():
            return self.gate(p_list, g_flat, global_feats, g_old_list, hist_list)

    def total_masks(self, groups: list[ParamGroup]) -> int:
        return sum(g.size for g in groups)

    def governor_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def mask_entropy(m: torch.Tensor) -> torch.Tensor:
        m = m.clamp(1e-6, 1 - 1e-6)
        return -(m * torch.log(m) + (1 - m) * torch.log(1 - m)).mean()


class DirectGateGovernor(nn.Module):
    """Phase 2b: ONE trainable gate per weight, no shared network.

    The decisive 'WHERE' test: 17,249 free logits, each learning its own
    gate against the same 3-step smooth meta-objective. Each gate integrates
    its own per-weight harm/benefit signal over the meta trajectories --
    the shared-MLP bottleneck (features -> one mapping for ALL weights) is
    removed. If a selective solution exists, these gates can find it.

    At stream time the gates are FROZEN static masks per weight (the
    learned soft-mask regime; importance is LEARNED here, not FIM-computed).
    """

    def __init__(self, groups: list[ParamGroup], init_mask: float = 0.3) -> None:
        super().__init__()
        self.sizes = [g.size for g in groups]
        b = math.log(init_mask / (1.0 - init_mask))
        self.logits = nn.Parameter(torch.full((sum(self.sizes),), b))

    def _masks(self, differentiable: bool) -> list[torch.Tensor]:
        m = torch.sigmoid(self.logits)
        if not differentiable:
            m = m.detach()
        return list(m.split(self.sizes))

    def gate_from_state(
        self,
        p_cur=None,
        g_list=None,
        groups=None,
        x=None,
        y=None,
        loss=None,
        device=None,
        differentiable: bool = True,
        g_old_list=None,
        hist_list=None,
    ) -> list[torch.Tensor]:
        """Same interface as HRMIntentGovernor; state inputs are ignored."""
        return self._masks(differentiable)

    def gate_from_model(
        self,
        model=None,
        groups=None,
        x=None,
        y=None,
        loss=None,
        device=None,
        g_old_list=None,
        hist_list=None,
    ) -> list[torch.Tensor]:
        return self._masks(differentiable=False)

    def total_masks(self, groups: list[ParamGroup]) -> int:
        return sum(self.sizes)

    def governor_params(self) -> int:
        return self.logits.numel()


def mask_stats(masks: list[torch.Tensor]) -> dict[str, float]:
    m = torch.cat([mi.detach().flatten() for mi in masks])
    return {
        "mask_mean": float(m.mean()),
        "mask_std": float(m.std()),
        "mask_min": float(m.min()),
        "mask_max": float(m.max()),
        "frac_lt_0.1": float((m < 0.1).float().mean()),
        "frac_gt_0.9": float((m > 0.9).float().mean()),
    }


class HRMController:
    """Phase-2 controller wrapping the frozen intent governor (§17 hooks)."""

    def __init__(self, governor: HRMIntentGovernor, groups: list[ParamGroup],
                 device: torch.device) -> None:
        self.governor = governor
        self.groups = groups
        self.device = device

    @torch.no_grad()
    def compute_masks(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss: torch.Tensor,
        g_old_list: list[torch.Tensor] | None = None,
        snapshot: dict[str, torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Read-only gate computation (does NOT touch .grad).

        g_old_list: gradients of ALL previous tasks' losses w.r.t. current
        params -> the "impact of changing this weight on old knowledge"
        signal, same semantics as meta-training. snapshot: phase-start
        params -> rel-change history.
        """
        hist_list = None
        if snapshot is not None:
            eps = 1e-8
            hist_list = [
                (g.param.detach() - snapshot[g.name]).abs()
                / (snapshot[g.name].abs() + eps)
                for g in self.groups if g.name in snapshot
            ]
        return self.governor.gate_from_model(
            model, self.groups, x, y, loss, self.device,
            g_old_list, hist_list)

    @torch.no_grad()
    def scale_update(
        self,
        model: nn.Module,
        masks: list[torch.Tensor],
        pre: dict[str, torch.Tensor],
    ) -> None:
        """Apply the gates to the ACTUAL optimizer update:
        W = pre + M o (W_adam - pre). AdamW's per-weight normalization
        would otherwise destroy the mask (it acts on raw grads only)."""
        for group, m in zip(self.groups, masks):
            if group.name not in pre:
                continue
            p = group.param.data
            p.add_((m.reshape(p.shape) - 1.0) * (p - pre[group.name]))


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


def measure_rel_change(
    model: nn.Module, snapshot: dict[str, torch.Tensor]
) -> float:
    """Mean relative |dW|/|W| over all parameters (change cost statistic)."""
    eps = 1e-8
    total = 0.0
    n = 0
    for name, p in model.named_parameters():
        if name not in snapshot:
            continue
        rel = (p.detach() - snapshot[name]).abs() / (snapshot[name].abs() + eps)
        total += float(rel.mean())
        n += 1
    return total / n if n else 0.0