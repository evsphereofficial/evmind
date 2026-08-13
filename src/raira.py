"""RAIRAW-V1 core — Retention-Attention Intent Recursive Adapted Weights.

Experiment Series 2 (RAIRAW_V1_Architecture_Series_2.md). The adaptive unit
is a small RECURSIVE model (the RAIRAW) holding local authority over a
bounded region of main-model weights, instead of an undifferentiated scalar
weight or a single global per-weight gate.

Hierarchy (see RAIRAW_V1/README.md):
  HRM (frozen governor)  ->  WHERE: region importance + active count
  RAIRAW pool            ->  HOW: per-weight gates from a recursive
                             R / A / I / Controller cell
  H_MEM                  ->  observed influence memory from RAIRAW reports,
                             fed back into future HRM allocation

Design of the recursive cell (budget < ~1K params, Experiment 2.1):
  state+ctx -> h_t = tanh(W_h [h_{t-1}; ctx_t])
  heads: R (retention), A (attention) per region; I (intent) 4-dim
  controller: gate_i = sigmoid(w_c [f_i; R; A; I; h]) per weight

Region pool: all RAIRAW instances SHARE the cell parameters (weight-tied);
each active region carries its own recurrent state h, so differentiation
emerges from region features + observed influence (H_MEM), not from
per-instance parameters. The number of states = MAX_RAIRAW.

The update rule is identical to the HRM system's enforced semantics:
  W = pre + M o (W_adam - pre), gates < close_threshold are closed nodes
  (reuse HRMController.scale_update / zero_closed_moments in the stream).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .hrm import compute_global_features


@dataclass(frozen=True)
class Region:
    """A contiguous chunk of one module's flattened weight tensor.

    region_size <= ~1,000 weights by default (Experiment 2.2).
    """
    region_id: int
    group_name: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def weight_slice(self) -> slice:
        return slice(self.start, self.stop)


def make_regions(groups, region_size: int = 1000) -> list[Region]:
    """Partition every module's flattened weights into contiguous regions.

    Regions never span modules; a 17,249-weight model with region_size=1000
    yields 18 regions (17 x 1000 + 249)."""
    regions: list[Region] = []
    rid = 0
    for g in groups:
        n = g.size
        for start in range(0, n, region_size):
            stop = min(start + region_size, n)
            regions.append(Region(region_id=rid, group_name=g.name,
                                  start=start, stop=stop))
            rid += 1
    return regions


class RairawCell(nn.Module):
    """Tiny recursive controller: R / A / I / Controller.

    Budget (h_dim=16, intent_dim=4, global 9, region ctx 7):
        W_h: 16 x 32 + 16      = 528  (recurrence)
        R  : 17,  A: 17          =  34  (retention / attention heads)
        I  : 4 x 32 + 4         = 132  (intent head)
        C  : 1 x (5+16+1+1+4)+1  =  29  (per-weight controller head)
        total                   = 723  parameters  (< 1,000, doc section 3)

    ctx_t per region+step (16 dims):
        region aggregates     7: mean|g|, mean|p|, mean|gp|, log1p(mean g^2),
                                mean HRM mask (governor signal), mean I_mem
                                importance, H_MEM influence feedback
        global context        9: compute_global_features (batch/task state)
    per-weight features f_i (5 dims):
        log1p|g_i|, log1p|p_i|, position-in-region, log1p(I_mem_i),
        log1p(ret_i)   <-- old-task sensitivity field (gA_hat, normalized):
        the exact quantity the meta objective judges; gives the Retention
        node explicit knowledge of "how dangerous is this weight"

    The controller head maps [f_i; R; A; I; h] -> gate in (0, 1).
    """

    def __init__(self, h_dim: int = 16, intent_dim: int = 4,
                 ctx_dim: int = 16, fw_dim: int = 4) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.intent_dim = intent_dim
        self.ctx_dim = ctx_dim
        self.fw_dim = fw_dim

        self.rec = nn.Linear(h_dim + ctx_dim, h_dim)       # recurrence
        self.ret_head = nn.Linear(h_dim, 1)                # R
        self.att_head = nn.Linear(h_dim, 1)                # A
        self.int_head = nn.Linear(h_dim + ctx_dim, intent_dim)  # I
        self.ctrl = nn.Linear(fw_dim + h_dim + 2 + intent_dim, 1)  # C

        # init: neutral retention/attention, slight open bias
        with torch.no_grad():
            self.ret_head.bias.fill_(0.0)
            self.att_head.bias.fill_(0.0)
            b = math.log(0.3 / 0.7)
            self.ctrl.bias.fill_(b)
            self.rec.bias.fill_(0.0)

    @property
    def cell_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _heads(self, h: torch.Tensor, ctx: torch.Tensor):
        r = self.ret_head(h).sigmoid()                     # (1,)
        a = self.att_head(h).sigmoid()                     # (1,)
        i = self.int_head(torch.cat([h, ctx], dim=-1)).softmax(-1)  # (I,)
        return r, a, i

    def step(self, h: torch.Tensor, ctx: torch.Tensor,
             fw: torch.Tensor, need_info: bool = False
             ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """One recursive step.

        Args:
            h: (h_dim,) recurrent state (region's RAIRAW state).
            ctx: (ctx_dim,) region+global context (no per-weight info).
            fw: (n, fw_dim) per-weight features of the region.
            need_info: compute R/A/I scalar logging (GPU->CPU syncs; only
              the live logger needs it — the meta burst passes False).
        Returns:
            (h_new, gates, info) where gates: (n,), info: {R, A, I, adapter_need}.
        """
        h_in = torch.cat([h, ctx], dim=-1)
        h_new = torch.tanh(self.rec(h_in))
        r, a, i = self._heads(h_new, ctx)
        g = self.ctrl(torch.cat([
            fw,
            r.expand(fw.shape[0], 1),
            a.expand(fw.shape[0], 1),
            i.unsqueeze(0).expand(fw.shape[0], -1),
            h_new.unsqueeze(0).expand(fw.shape[0], -1),
        ], dim=-1)).sigmoid().squeeze(-1)
        if not need_info:
            return h_new, g, {}
        # adapter need: high attention on a highly-retained region
        adapter_need = float((a * r).clamp(0, 1).item())
        return h_new, g, {"R": float(r.item()), "A": float(a.item()),
                          "I": i.detach().cpu().tolist(),
                          "adapter_need": adapter_need}


def region_context(
    g_flat: torch.Tensor,
    p_flat: torch.Tensor,
    hrm_mask: torch.Tensor,
    mem_imp: torch.Tensor | None,
    hmem_influence: float,
    global_feats: torch.Tensor,
    region: Region,
    alloc_frac: float,
    region_frac: float,
) -> torch.Tensor:
    """7 region aggregates + alloc context + 9 global features -> (18,) ctx.

    alloc_frac: active RAIRAW count / region count (how much capacity the
      HRM gave the system this phase — lets the cell rescale its gates).
    region_frac: region weight share of the main pool (normalized size)."""
    w = region.weight_slice
    g_r, p_r = g_flat[w], p_flat[w]
    eps = 1e-12
    feats = [
        g_r.abs().mean(),
        p_r.abs().mean(),
        (g_r * p_r).abs().mean(),
        torch.log1p((g_r ** 2).mean() + eps),
        hrm_mask[w].mean(),                       # HRM protective signal
        mem_imp[w].mean() if mem_imp is not None
        else torch.tensor(0.0, device=g_r.device),
        torch.tensor(hmem_influence, device=g_r.device),   # H_MEM feedback
        torch.tensor(alloc_frac, device=g_r.device),
        torch.tensor(max(region_frac, eps), device=g_r.device),
    ]
    return torch.cat([torch.stack(feats), global_feats])


def per_weight_features(
    g_flat: torch.Tensor,
    p_flat: torch.Tensor,
    mem_imp: torch.Tensor | None,
    region: Region,
    ret_flat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-weight features (n, 5): log1p|g|, log1p|p|, pos-frac, log1p I_mem,
    log1p|ret| (normalized old-task sensitivity field gA_hat)."""
    w = region.weight_slice
    g_r, p_r = g_flat[w], p_flat[w]
    eps = 1e-12
    pos = (torch.arange(region.size, device=g_r.device).float()
           / max(1, region.size - 1))
    if mem_imp is None:
        mem_r = torch.zeros_like(g_r)
    else:
        mem_r = mem_imp[w]
    if ret_flat is None:
        ret_r = torch.zeros_like(g_r)
    else:
        ret_r = ret_flat[w]
    return torch.stack([
        torch.log1p(g_r.abs() + eps),
        torch.log1p(p_r.abs() + eps),
        pos,
        torch.log1p(mem_r.abs() + eps),
        torch.log1p(ret_r.abs() + eps),
    ], dim=-1)


class RairawPool(nn.Module):
    """Bounded pool of RAIRAW instances (MAX_RAIRAW), weight-tied cell.

    The cell parameters are SHARED across the pool (§6 pool; the doc's pool
    of available RAIRAW models); each active region instance carries its own
    recurrent state h so per-region behavior differentiates over the stream.
    """

    def __init__(self, cell: RairawCell, max_rairaw: int = 20,
                 total_weights: int = 17249) -> None:
        super().__init__()
        self.cell = cell
        self.max_rairaw = max_rairaw
        self.total_weights = total_weights
        self._states: list[torch.Tensor] = [None] * max_rairaw
        self._slot_map: dict[int, int] = {}   # region_id -> pool slot
        self.active: set[int] = set()

    def reset_states(self, device: torch.device) -> None:
        self._states = [torch.zeros(self.cell.h_dim, device=device)
                        for _ in range(self.max_rairaw)]
        self._slot_map = {}
        self.active = set()

    @torch.no_grad()
    def activate(self, region_ids: list[int]) -> None:
        """Declare which regions are active; assign them pool slots by rank."""
        self.active = set(region_ids)
        self._slot_map = {rid: slot for slot, rid in enumerate(sorted(region_ids))}
        for slot in self._slot_map.values():
            self._states[slot].zero_()

    def gate_region(
        self,
        region: Region,
        g_flat: torch.Tensor,
        p_flat: torch.Tensor,
        hrm_mask_flat: torch.Tensor,
        mem_imp: torch.Tensor | None,
        ret_flat: torch.Tensor | None,
        hmem_influence: float,
        global_feats: torch.Tensor,
        alloc_frac: float,
        need_info: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """One recursive step for one active region; returns (gates, info).

        Differentiable: gradient flows to the shared cell through the gates
        and through the recurrence (h_{t-1} -> h_t). Callers that must not
        propagate (live stream) wrap the whole call in torch.no_grad()."""
        slot = self._slot_map[region.region_id]
        ctx = region_context(g_flat, p_flat, hrm_mask_flat, mem_imp,
                             hmem_influence, global_feats, region,
                             alloc_frac, region.size / self.total_weights)
        fw = per_weight_features(g_flat, p_flat, mem_imp, region, ret_flat)
        h_new, gates, info = self.cell.step(self._states[slot], ctx, fw,
                                            need_info=need_info)
        self._states[slot] = h_new
        return gates, info


class HmemMemory:
    """H_MEM: observed memory of RAIRAW influence (§7-§10).

    Begins empty. Each task, every active RAIRAW reports its region's
    influence (relative gradient magnitude observed during training);
    H_MEM accumulates a per-region EMA of those reports and serves the
    result back as an allocation prior and as region context.
    """

    def __init__(self, num_regions: int, alpha: float = 0.5,
                 device: torch.device | None = None) -> None:
        self.alpha = alpha
        self.influence = torch.zeros(num_regions, device=device)
        self.n = 0

    def is_empty(self) -> bool:
        return self.n == 0

    def update(self, reports: dict[int, float]) -> None:
        """reports: {region_id: influence} from the active RAIRAWs."""
        if not reports:
            return
        flat = torch.stack([
            torch.tensor(v, dtype=torch.float32,
                         device=self.influence.device)
            for v in reports.values()])
        for rid, v in reports.items():
            self.influence[rid] = (self.alpha * float(v)
                                   + (1 - self.alpha) * self.influence[rid])
        self.n += 1

    def prior(self) -> torch.Tensor:
        """Normalized per-region influence in [0, 1] (1 = most influential)."""
        if self.n == 0:
            return torch.zeros_like(self.influence)
        v = self.influence
        return (v - v.min()) / (v.max() - v.min() + 1e-12)

    def value(self, region_id: int) -> float:
        return float(self.influence[region_id])


def region_importance(
    masks: list[torch.Tensor],
    groups,
    regions: list[Region],
    device: torch.device,
) -> torch.Tensor:
    """HRM per-weight masks -> per-region importance (mean mask)."""
    req = {}
    gnames = {g.name: i for i, g in enumerate(groups)}
    for g, m in zip(groups, masks):
        req[g.name] = m.detach().flatten()
    imp = torch.zeros(len(regions), device=device)
    for r in regions:
        imp[r.region_id] = req[r.group_name][r.weight_slice].mean()
    return imp


def allocate_rairaws(
    region_imp: torch.Tensor,
    masks: list[torch.Tensor],
    hmem: HmemMemory,
    n_regions: int,
    max_rairaw: int,
    close_threshold: float = 0.02,
    blend: float = 0.7,
    sparse_target: float = 0.3,
) -> tuple[list[int], torch.Tensor]:
    """HRM WHERE decision: which regions get a RAIRAW this phase.

    Importance = blend * HRM mask importance + (1-blend) * H_MEM prior
    (H_MEM empty -> pure HRM on the first task). Active count K follows the
    HRM's open mask mass:
        k = round(n_regions * frac_open / sparse_target)
    so that only a bounded fraction of the 17K pool is under RAIRAW
    authority; the rest are closed nodes (remain available later).

    Returns (active region ids, blended importance).
    """
    if hmem.is_empty():
        blended = region_imp.clone()
    else:
        blended = (blend * region_imp
                   + (1 - blend) * hmem.prior())
    m_all = torch.cat([m.detach().flatten() for m in masks])
    frac_open = float((m_all > close_threshold).float().mean())
    k = int(round(n_regions * frac_open / max(sparse_target, 1e-3)))
    k = max(1, min(k, max_rairaw, n_regions))
    top = torch.argsort(blended, descending=True)[:k]
    return sorted(top.tolist()), blended


def influence_report(
    g_flat_all: torch.Tensor,
    g_flat: torch.Tensor,
    region: Region,
) -> float:
    """RAIRAW -> HRM feedback (§9): region influence = relative mean |g|.

    How strongly the current task pulls this region, normalized by the
    global mean |g| (observed importance; not policy-dependent)."""
    w = region.weight_slice
    denom = g_flat_all.abs().mean() + 1e-12
    return float(g_flat[w].abs().mean() / denom)