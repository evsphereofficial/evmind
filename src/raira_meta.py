"""RAIRAW-V1 meta-training of the shared recursive controller.

Experiment Series 2 (RAIRAW_V1_Architecture_Series_2.md). The RAIRAW cell is
trained FIRST with the same selective-plasticity objective as the HRM
governor:

    L = L_new(after burst)
      + lambda_old * L_old(after burst)
      + lambda_sparse * (mean_gate - sparse_target)^2
      + lambda_ewc * mean(M * (gA_hat^2 - 1))
      + lambda_delta * mean(|dW|/|W|)  (extrapolated to live-phase length)

Method (mirrors meta_pretrain.py):
- task pair (A, B) from the 5 geometry families with randomized shifts
- warmup on A (old knowledge installed, captured into a SensitivityMemory)
- a BURST of RAIRAW-gated updates on B (functional_call, first-order):
    per step: grads -> frozen HRM governor probe masks (allocation only)
              -> region importance -> top-K regions get a RAIRAW
              -> recursive cell emits per-weight gates for active regions
              -> stateless AdamW update W -= lr * M o m_hat/sqrt(v_hat)
              (exactly the live stream's enforced semantics)
- the HRM governor is FROZEN and only used for the WHERE decision; the
  RAIRAW cell is the only trainable component (its recurrence chains
  through the burst: h_t -> h_{t+1}).

H_MEM begins empty in every meta step (fresh RAIRAW instance states);
the cell receives a RANDOM H_MEM influence value per meta step so the
context channel is meaningful when the live stream accumulates real
influence reports across tasks.

The trained cell is frozen afterwards and evaluated via experiment3.py
(live 5-task stream).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from .config import load_config
from .experiment import set_seed
from .hrm import (
    build_module_groups, compute_global_features, compute_influence,
    HRMIntentGovernor, mask_stats, SensitivityMemory,
)
from .meta_pretrain import (
    make_meta_task, meta_task_labels, reset_model, sample_chunk,
    warmup_batches, _sample_batch, BCE,
)
from .model import TinyNumericTransformer
from .raira import (
    allocate_rairaws, HmemMemory, influence_report, make_regions,
    RairawCell, RairawPool, region_importance,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_GOVERNOR = str(PROJECT_ROOT / "results_hmem" / "magnitude_governor"
                       / "governor_pretrained.pt")


# ---------------------------------------------------------------------------
# Differentiable RAIRAW-gated burst (functional unroll)
# ---------------------------------------------------------------------------

def raira_burst(
    model: nn.Module,
    pool: RairawPool,
    groups,
    regions: list,
    governor_hrm: HRMIntentGovernor,
    p_cur: dict[str, torch.Tensor],
    task,
    lr: float,
    steps: int,
    batch_size: int,
    seed_base: int,
    device: torch.device,
    differentiable: bool,
    mem_imp: list[torch.Tensor] | None,
    mem_dir: list[torch.Tensor] | None,
    offs: dict[str, int],
    close_threshold: float = 0.02,
    hmem_influence: float = 0.0,
    ungated: bool = False,
    meta_mode: bool = False,
    train_governor: bool = False,
    alloc_imp: torch.Tensor | None = None,
    alloc_frac_open: float | None = None,
    alloc_alpha: float = 0.2,
    sparse_target: float = 0.3,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Unroll `steps` RAIRAW-gated updates on `task`.

    Allocation happens ONCE at burst start: the frozen HRM governor's masks
    on a probe batch -> region importance -> top-K regions get a RAIRAW
    (WHERE). The recursive cell then emits per-weight gates for active
    regions every step (HOW); inactive regions are closed nodes (gate 0,
    no Adam moments), mirroring the live stream's enforcement.

    meta_mode: skip the close_threshold zeroing — inside the meta unroll a
      zeroed gate creates a dead zone with NO gradient (the cell can never
      learn to reopen it); the threshold is a live-stream semantic only.
    alloc_imp / alloc_frac_open: externally EMA'd WHERE (stable stage for
      the cell; the raw governor allocation oscillates between meta steps).

    offs: module name -> offset into the global concatenated flat tensor.

    Returns (final params dict, diagnostics dict).
    """
    beta1, beta2, ad_eps = 0.9, 0.999, 1e-8
    sizes = [g.size for g in groups]

    if ungated:
        active = []
        hrm_flat = None
        gfeats_p = None
    else:
        # ---- WHERE: HRM allocation on a probe batch (frozen governor) ----
        xp = _sample_batch(batch_size, seed_offset=seed_base * 3 + 1).to(device)
        yp = meta_task_labels(xp, task).to(device)
        pred = functional_call(model, p_cur, (xp,))
        loss_p = BCE(pred, yp)
        g_probe = torch.autograd.grad(
            loss_p, [p_cur[g.name] for g in groups], create_graph=False)
        gp_all = torch.cat([g.detach().flatten() for g in g_probe])
        pp_all = torch.cat([p_cur[g.name].detach().flatten()
                            for g in groups])
        gfeats_p = compute_global_features(xp, yp, loss_p, gp_all, pp_all,
                                           device)
        hrm_masks = governor_hrm.gate_from_state(
            p_cur, g_probe, groups, xp, yp, loss_p, device,
            differentiable=(meta_mode and train_governor),
            g_old_list=None, mem_imp_list=None,
            mem_dir_list=None,
            hmem_list=compute_influence(g_probe, "magnitude", device=device))
        if meta_mode and train_governor:
            # ---- v5 Phase B: trainable WHERE + LEARNED capacity ----
            # importance differentiates into the governor's masks; the
            # allocation blends EMA (stable stage) with the current masks
            # (gradient path). The capacity demand D = cap_head(context)
            # sets the budget: w sums to D * n_regions, so the composed
            # meta loss trains D directly (learning pressure raises it,
            # retention damage lowers it) — no fixed sparse target.
            imp_raw = region_importance(hrm_masks, groups, regions, device,
                                        detach=False)
            hrm_flat = torch.cat([m.detach().flatten() for m in hrm_masks])
            if alloc_imp is not None:
                imp_use = ((1 - alloc_alpha) * alloc_imp
                           + alloc_alpha * imp_raw)
            else:
                imp_use = imp_raw
            demand = governor_hrm.capacity_demand(
                gfeats_p, torch.zeros(3, device=device))
            k = int(round(float(demand) * len(regions)))
            k = max(1, min(k, pool.max_rairaw, len(regions)))
            r_sizes = torch.tensor(
                [regions[i].size for i in range(len(regions))],
                device=device, dtype=torch.float32)
            sel = imp_use + 0.05 * torch.log1p(r_sizes)
            topk = torch.argsort(sel, descending=True)[:k]
            tau = 0.1
            w_topk = demand * len(regions) * torch.softmax(
                sel[topk] / tau, dim=0)
            active = sorted(topk.tolist())
            pool.activate(active)
            w_map = {int(rid): w_topk[pos]
                     for pos, rid in enumerate(topk.tolist())}
            imp = imp_raw.detach().clone()
            frac_open = float(demand.detach())
        else:
            hrm_flat = torch.cat([m.detach().flatten() for m in hrm_masks])
            imp = region_importance(hrm_masks, groups, regions, device)
            m_all = torch.cat([m.detach().flatten() for m in hrm_masks])
            frac_open = float((m_all > close_threshold).float().mean())
            hmem = HmemMemory(len(regions), alpha=0.0, device=device)
            if alloc_imp is not None:
                imp_use = alloc_imp
                frac_use = alloc_frac_open
            else:
                imp_use = imp
                frac_use = frac_open
            active, blended = allocate_rairaws(
                imp_use, hrm_masks, hmem, len(regions), pool.max_rairaw,
                close_threshold=close_threshold,
                frac_open_override=frac_use)
            pool.activate(active)
            w_map = None
        active_set = set(active)

    m_buf = [torch.zeros_like(p_cur[g.name]) for g in groups]
    v_buf = [torch.zeros_like(p_cur[g.name]) for g in groups]

    mask_log: list[dict] = []
    mean_gates = []        # mean gate over ACTIVE weights (sparse scope)
    ewc_costs = []
    g_old_hat_all = None
    if mem_dir is not None:
        g_old_hat_all = torch.cat([g.detach() / (g.abs().mean() + 1e-12)
                                   for g in mem_dir])

    xb = sample_chunk(steps, batch_size, seed_base, device)
    yb = meta_task_labels(xb, task)
    for s in range(steps):
        x = xb[s * batch_size:(s + 1) * batch_size]
        y = yb[s * batch_size:(s + 1) * batch_size]
        pred = functional_call(model, p_cur, (x,))
        loss = BCE(pred, y)
        grad_list = torch.autograd.grad(
            loss, [p_cur[g.name] for g in groups],
            create_graph=False, allow_unused=False)

        if ungated:
            masks = [torch.ones_like(p_cur[g.name].flatten()) for g in groups]
            mean_gates.append(torch.ones((), device=device))
            act_ms, act_idx = [], []
        else:
            g_flat = torch.cat([g.detach().flatten() for g in grad_list])
            p_flat = torch.cat([p_cur[g.name].detach().flatten()
                                for g in groups])
            mem_imp_flat = (torch.cat([v.detach().flatten() for v in mem_imp])
                            if mem_imp is not None else None)
            gfeats = compute_global_features(x, y, loss, g_flat, p_flat,
                                             device)
            # per-region gates from the recursive cell (differentiable)
            full = torch.zeros_like(g_flat)
            act_ms = []       # gates of ACTIVE regions only (cost scope)
            act_idx = []      # their flat indices (for A-sensitivity)
            for r in regions:
                if r.region_id not in active_set:
                    continue
                gates_r, info_r = pool.gate_region(
                    r, g_flat, p_flat, hrm_flat, mem_imp_flat,
                    g_old_hat_all, hmem_influence, gfeats,
                    alloc_frac=len(active) / len(regions), need_info=False)
                if w_map is not None:
                    gates_r = gates_r * w_map[r.region_id]
                base = offs[r.group_name]
                act_ms.append(gates_r)
                act_idx.append(torch.arange(r.start, r.stop,
                                            device=g_flat.device) + base)
                full[base + r.start: base + r.stop] = gates_r
            masks = list(full.split(sizes))
            # cost scope = ACTIVE REGIONS ONLY (the closed mass is ~95% of
            # the pool; a global target is unsatisfiable). Region-level scope
            # keeps the sparse/ewc gradients undiluted by the closed mass.
            if act_ms:
                act_m = torch.cat(act_ms)
                mean_gates.append(act_m.mean())
            else:
                mean_gates.append(torch.zeros((), device=g_flat.device))

        mask_log.append(mask_stats([m.detach() for m in masks]))
        if g_old_hat_all is not None:
            if act_idx:
                m_act = torch.cat(act_ms)
                gA_act = g_old_hat_all[torch.cat(act_idx)]
                ewc_costs.append((m_act * gA_act ** 2).mean())

        for i, (group, m, g) in enumerate(zip(groups, masks, grad_list)):
            if close_threshold > 0.0 and not meta_mode:
                m = torch.where(m < close_threshold, torch.zeros_like(m), m)
            mf = m.reshape(g.shape)
            closed = mf == 0.0
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * g
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * g * g
            if closed.any():
                m_buf[i] = m_buf[i].masked_fill(closed, 0.0)
                v_buf[i] = v_buf[i].masked_fill(closed, 0.0)
            t = s + 1
            m_hat = m_buf[i] / (1 - beta1 ** t)
            v_hat = v_buf[i] / (1 - beta2 ** t)
            p_cur[group.name] = p_cur[group.name] - lr * mf * (
                m_hat / (v_hat.sqrt() + ad_eps))

    mean_gate = torch.stack(mean_gates).mean()
    mean_ewc = torch.stack(ewc_costs).mean() if ewc_costs else None
    diag = {
        "mean_gate": mean_gate,
        "mean_ewc": mean_ewc,
        "allocated_regions": len(active) if not ungated else 0,
        "allocated_weights": (sum(r.size for r in regions
                                  if r.region_id in active_set)
                              if not ungated else 0),
        "frac_active": (len(active) / len(regions)) if not ungated else 1.0,
        "mask_log": mask_log,
    }
    if not ungated:
        diag["imp"] = imp.detach().clone()
        diag["frac_open"] = frac_open
        diag["demand"] = float(demand.detach()) \
            if (meta_mode and train_governor) else frac_open
    return p_cur, diag


def raira_meta_step(
    model: nn.Module,
    pool: RairawPool,
    governor_hrm: HRMIntentGovernor,
    groups,
    regions: list,
    offs: dict[str, int],
    config,
    device: torch.device,
    seed_off: int,
    rng: torch.Generator,
    alloc_imp: torch.Tensor | None = None,
    alloc_frac_open: float | None = None,
) -> dict:
    """One meta step: task sequence -> memory -> RAIRAW burst on B."""
    m = config.meta
    ra = config.raira
    base_lr = m.lr
    warmup_lr = getattr(m, "warmup_lr", m.lr)
    bsize = m.batch_size
    old_tasks_max = int(getattr(m, "old_tasks_max", 0))

    torch.manual_seed(100_000 + seed_off * 7)
    reset_model(model)
    pool.reset_states(device)
    k_old = int(torch.randint(0, old_tasks_max + 1, (1,), generator=rng)[0])
    old_tasks = [make_meta_task(rng) for _ in range(k_old)]
    fam_b = make_meta_task(rng)

    memory = SensitivityMemory(groups, agg=getattr(m, "memory_agg", "ewc"),
                               device=device)
    g_old_imm = None
    for t_i, fam_a in enumerate(old_tasks):
        warmup_batches(model, fam_a, m.warmup_batches, bsize, warmup_lr,
                       seed_base=seed_off * 31 + t_i * 7, device=device)
        model.zero_grad(set_to_none=True)
        x_cap = _sample_batch(bsize, seed_offset=seed_off * 45 + 13 + t_i * 17).to(device)
        y_cap = meta_task_labels(x_cap, fam_a).to(device)
        g_t = torch.autograd.grad(
            BCE(model(x_cap), y_cap), [g.param for g in groups],
            create_graph=False, allow_unused=False)
        memory.update(g_t)
        g_old_imm = [gg.detach().clone() for gg in g_t]
    model.zero_grad(set_to_none=True)

    p_cur = {g.name: g.param.detach().clone().requires_grad_(True)
             for g in groups}
    p0 = {g.name: t.detach().clone() for g, t in
          zip(groups, [p_cur[g.name] for g in groups])}

    mem_imp = memory.importance() if k_old > 0 else None
    mem_dir = memory.direction() if k_old > 0 else None
    # H_MEM begins empty each meta step; randomize the influence context so
    # the channel is meaningful when the live stream accumulates reports.
    hmem_influence = float(torch.rand(1, generator=rng)[0])

    # PRE-BURST baselines (post-A-warmup state): the objective measures
    # CAUSAL deltas, not absolutes. An unlearned B costs only ~0.3-0.7
    # while a destroyed A costs ~1.5-1.9; absolute losses make "protect
    # A at all costs" dominate and collapse the gates.
    x_nb = _sample_batch(bsize, seed_offset=seed_off * 43 + 11).to(device)
    y_nb = meta_task_labels(x_nb, fam_b).to(device)
    loss_new_before = BCE(functional_call(model, p_cur, (x_nb,)), y_nb)
    old_before = []
    for t_i, fam_a in enumerate(old_tasks):
        x_ab = _sample_batch(bsize,
                             seed_offset=seed_off * 41 + 9 + t_i * 19).to(device)
        y_ab = meta_task_labels(x_ab, fam_a).to(device)
        old_before.append(BCE(functional_call(model, p_cur, (x_ab,)), y_ab))
    loss_old_before = (torch.stack(old_before).mean() if old_before
                       else loss_new_before.detach() * 0.0)

    p_cur, diag = raira_burst(
        model, pool, groups, regions, governor_hrm, p_cur, fam_b,
        lr=base_lr, steps=m.burst_steps, batch_size=bsize,
        seed_base=seed_off * 37 + 5, device=device, differentiable=True,
        mem_imp=mem_imp, mem_dir=mem_dir, offs=offs,
        close_threshold=getattr(m, "close_threshold", 0.02),
        hmem_influence=hmem_influence, meta_mode=True,
        train_governor=getattr(m, "train_governor", False),
        alloc_imp=alloc_imp, alloc_frac_open=alloc_frac_open,
        alloc_alpha=getattr(m, "alloc_alpha", 0.2),
        sparse_target=getattr(m, "sparse_target", 0.3))

    x_bv = _sample_batch(bsize, seed_offset=seed_off * 43 + 11).to(device)
    y_bv = meta_task_labels(x_bv, fam_b).to(device)
    loss_new_after = BCE(functional_call(model, p_cur, (x_bv,)), y_bv)

    old_losses = []
    for t_i, fam_a in enumerate(old_tasks):
        x_av = _sample_batch(bsize, seed_offset=seed_off * 41 + 9 + t_i * 19).to(device)
        y_av = meta_task_labels(x_av, fam_a).to(device)
        old_losses.append(BCE(functional_call(model, p_cur, (x_av,)), y_av))
    loss_old_after = (torch.stack(old_losses).mean() if old_losses else
                      loss_new_after.detach() * 0.0)

    loss_new = loss_new_after - loss_new_before.detach()
    loss_old = (loss_old_after - loss_old_before.detach()).clamp(min=0.0)
    diag.update({
        "loss_new_after": float(loss_new_after.detach()),
        "loss_new_before": float(loss_new_before.detach()),
        "loss_old_after": float(loss_old_after.detach()),
        "loss_old_before": float(loss_old_before.detach()),
    })

    dW = torch.cat([(p_cur[g.name] - p0[g.name]).abs().flatten()
                    for g in groups])
    p0_abs = torch.cat([p0[g.name].abs().flatten() for g in groups])
    mean_rel_change = (dW.mean() + 1e-12) / (p0_abs.mean() + 1e-12)

    train_batches = int(math.ceil(config.train_samples / config.train.batch_size))
    live_steps = train_batches * int(config.train.epochs_per_task)
    phase_scale = live_steps / max(1, int(m.burst_steps))
    # penalty on BURST movement only: the extrapolated live cost is
    # astronomically larger and collapses the gates to ~0 (catatonia);
    # selectivity (ewc) is what protects old tasks in the live phase.
    delta_cost = mean_rel_change

    sparse_cost = (diag["mean_gate"] - getattr(m, "sparse_target", 0.3)) ** 2
    ewc_cost = diag["mean_ewc"] if diag["mean_ewc"] is not None \
        else torch.zeros((), device=device)

    # ---- v3 node-isolated objectives ----
    # Each node's parameters receive gradient ONLY from its own objective
    # (the composition controller mediates; see RairawCell docstring).
    # The absolute after-loss (lambda_after) is what makes "do nothing"
    # unattractive: closed gates leave after ~ before ~ 0.7, which still
    # costs the attention node. Pure deltas have "do nothing" as a
    # zero-gradient fixed point and the cell collapses to catatonia.
    lambda_after = float(getattr(m, "lambda_after", 0.3))
    lambda_old = float(getattr(m, "lambda_old", 1.0))
    lambda_sparse = float(getattr(m, "lambda_sparse", 1.0))
    lambda_ewc = float(getattr(m, "lambda_ewc", 0.05))
    lambda_delta = float(getattr(m, "lambda_delta", 5.0))

    # v5.1: phase-scaled retention. The burst measures damage over
    # burst_steps (20) but the live phase runs ~395 steps — burst-horizon
    # L_old (~0.08) is ~20x smaller than the real live damage. The v1
    # phase-scaling collapsed the SHARED cell (att died with ret); with
    # node isolation the scaled term only reaches ret/ctrl/int/gov —
    # attention keeps its unscaled learning objective and can still open
    # gates (additive composition), so the asymmetry is safe now.
    phase_scale_ret = (phase_scale if getattr(m, "phase_scale_ret", True)
                       else 1.0)

    loss_att = loss_new + lambda_after * loss_new_after
    loss_ret = (lambda_old * phase_scale_ret * loss_old
                + lambda_ewc * ewc_cost)
    loss_int = loss_att + loss_ret + lambda_sparse * sparse_cost
    loss_ctrl = (loss_new
                 + lambda_old * phase_scale_ret * loss_old
                 + lambda_sparse * sparse_cost
                 + lambda_ewc * ewc_cost
                 + lambda_delta * delta_cost)
    # v5: the governor's objective IS the composed loss. Its capacity
    # demand D gets its gradient through the allocation budget w (which
    # scales with D), not through any fixed-target capacity term.
    loss_gov = loss_ctrl
    gov_loss = loss_ctrl
    model.zero_grad(set_to_none=True)
    return {
        "gov_loss": gov_loss,
        "loss_gov": loss_gov,
        "loss_att": loss_att,
        "loss_ret": loss_ret,
        "loss_int": loss_int,
        "loss_ctrl": loss_ctrl,
        "loss_new": loss_new,
        "loss_old": loss_old,
        "sparse_cost": sparse_cost,
        "ewc_cost": ewc_cost,
        "delta_cost": delta_cost,
        "delta_cost_live": mean_rel_change * phase_scale,
        "phase_scale": phase_scale,
        "mask_stats": diag["mask_log"][-1],
        "allocated_weights": diag["allocated_weights"],
        "frac_active": diag["frac_active"],
        "imp": diag.get("imp"),
        "frac_open": diag.get("frac_open"),
        "demand": diag.get("demand"),
        "loss_new_before": diag["loss_new_before"],
        "loss_new_after": diag["loss_new_after"],
        "loss_old_before": diag["loss_old_before"],
        "loss_old_after": diag["loss_old_after"],
    }


def distill_cell(
    model: nn.Module,
    pool: RairawPool,
    governor_hrm: HRMIntentGovernor,
    groups,
    regions: list,
    offs: dict[str, int],
    config,
    device: torch.device,
    steps: int = 120,
    lr: float = 1e-2,
    seed: int = 999,
) -> None:
    """Init the cell at the sparse target: teach it uniform ~0.3 gates.

    The frozen governor's masks collapsed to near zero (mask_mean ~0.02)
    during its own meta-training, so distilling to them initializes the
    cell at catatonia (v2 did exactly that: mask_mean 0.004 at step 1).
    The meta phase is responsible for shaping; distillation only gives a
    neutral, open-at-target start (base=0.3, R=0, A=0)."""
    m = config.meta
    target = float(getattr(m, "sparse_target", 0.3))
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(pool.cell.parameters(), lr=lr)
    bsize = m.batch_size
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        s_off = int(torch.randint(1, 10 ** 9, (1,), generator=rng)[0])
        xp = _sample_batch(bsize, seed_offset=s_off).to(device)
        yp = meta_task_labels(xp, make_meta_task(rng)).to(device)
        p_cur = {g.name: g.param.detach().clone().requires_grad_(True)
                 for g in groups}
        loss_p = BCE(functional_call(model, p_cur, (xp,)), yp)
        g_probe = torch.autograd.grad(
            loss_p, [p_cur[g.name] for g in groups],
            create_graph=False, allow_unused=False)
        hrm_masks = governor_hrm.gate_from_state(
            p_cur, g_probe, groups, xp, yp, loss_p, device,
            differentiable=False, g_old_list=None, mem_imp_list=None,
            mem_dir_list=None,
            hmem_list=compute_influence(g_probe, "magnitude", device=device))
        hrm_flat = torch.cat([mm.detach().flatten() for mm in hrm_masks])
        g_flat = torch.cat([gg.detach().flatten() for gg in g_probe])
        p_flat = torch.cat([p_cur[g.name].detach().flatten() for g in groups])
        gfeats = compute_global_features(xp, yp, loss_p, g_flat, p_flat,
                                         device)
        perm = torch.randperm(len(regions), generator=rng)[:pool.max_rairaw]
        pool.reset_states(device)
        pool.activate([regions[int(i)].region_id for i in perm])
        total = torch.zeros((), device=device)
        for rid in sorted(pool.active):
            r = regions[rid]
            gates_r, _ = pool.gate_region(
                r, g_flat, p_flat, hrm_flat, None, None,
                float(torch.rand(1, generator=rng)[0]), gfeats,
                alloc_frac=len(pool.active) / len(regions), need_info=False)
            total = total + F.mse_loss(
                gates_r, torch.full_like(gates_r, target))
        total = total / len(pool.active)
        total.backward()
        opt.step()
        if it == 0 or (it + 1) % 30 == 0:
            print(f"distill {it + 1:4d}/{steps}  "
                  f"cellMSE={float(total.detach()):.4f}")
    model.zero_grad(set_to_none=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAIRAW-V1 recursive controller meta-training")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline2.yaml"))
    parser.add_argument("--governor", default=DEFAULT_GOVERNOR,
                        help="frozen HRM governor for the WHERE decision")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results_raira"))
    parser.add_argument("--seed", type=int, default=None,
                        help="override the experiment seed")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sparse-target", type=float, default=None)
    parser.add_argument("--lambda-ewc", type=float, default=None)
    parser.add_argument("--old-tasks-max", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.old_tasks_max is not None:
        cfg.meta.old_tasks_max = args.old_tasks_max
    if args.sparse_target is not None:
        cfg.meta.sparse_target = args.sparse_target
    if args.lambda_ewc is not None:
        cfg.meta.lambda_ewc = args.lambda_ewc
    if args.steps is not None:
        cfg.meta.steps = args.steps
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        if f.is_file():
            f.unlink()

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Device: {device}")

    model = TinyNumericTransformer(
        input_dim=cfg.model.input_dim, seq_len=cfg.model.seq_len,
        embedding_dim=cfg.model.embedding_dim, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    groups = build_module_groups(model)
    total_weights = sum(g.size for g in groups)
    regions = make_regions(groups, region_size=cfg.raira.region_size)
    n_regions = len(regions)
    # module name -> offset into the global concatenated flat tensor
    offs = {}
    flat_off = 0
    for g in groups:
        offs[g.name] = flat_off
        flat_off += g.size

    # frozen HRM governor: WHERE authority only (never trained here)
    ckpt = torch.load(args.governor, map_location="cpu")
    gfeat = 9  # compute_global_features dim
    pwd = int(ckpt["mlp.0.weight"].shape[1]) - len(groups) - gfeat - 1
    governor_hrm = HRMIntentGovernor(
        num_groups=len(groups),
        granularity=cfg.governor.granularity,
        hidden_dim=cfg.governor.hidden_dim,
        refine_steps=cfg.governor.refine_steps,
        init_mask=cfg.governor.init_mask,
        per_weight_feat_dim=pwd,
    ).to(device)
    governor_hrm.load_state_dict(ckpt, strict=False)
    governor_hrm.eval()
    for p in governor_hrm.parameters():
        p.requires_grad_(False)

    cell = RairawCell(h_dim=cfg.raira.h_dim, intent_dim=cfg.raira.intent_dim,
                      ctx_dim=7 + 2 + 9, fw_dim=5).to(device)
    pool = RairawPool(cell, max_rairaw=cfg.raira.max_rairaw,
                              total_weights=total_weights).to(device)

    m = cfg.meta
    print("=" * 60)
    print("RAIRAW-V1 META-TRAINING (recursive controller)")
    print("=" * 60)
    print(f"Base model params: {model.count_parameters():,}  "
          f"main weight pool: {total_weights:,}  regions: {n_regions} "
          f"(<= {cfg.raira.region_size} weights each)")
    print(f"RAIRAW cell params: {cell.cell_params:,}  "
          f"(budget < 1000), pool max {cfg.raira.max_rairaw}")
    print(f"HRM governor ({'trainable' if getattr(m, 'train_governor', False) else 'frozen'}, "
          f"WHERE + capacity demand): {governor_hrm.governor_params():,} "
          f"params from {args.governor}")
    print(f"Objective (v3 nodes): att <- L_new+{getattr(m, 'lambda_after', 0.3)}*after; "
          f"ret <- {m.lambda_old}*L_old+{getattr(m, 'lambda_ewc', 0.02)}*EWC; "
          f"int <- composed; ctrl <- composed+{m.lambda_sparse}*sparse"
          f"+{m.lambda_delta}*dW")
    print(f"Meta: {m.meta_batch} parallel steps x {m.steps} updates, "
          f"warmup={m.warmup_batches}, burst={m.burst_steps}x, batch={m.batch_size}")
    print(f"H_MEM: empty per meta step; random influence context "
          f"(alpha={cfg.raira.hmem_alpha} live)")
    reset_model(model)

    rng = torch.Generator().manual_seed(7_777 + cfg.train.seed)
    start = time.perf_counter()

    n_distill = int(getattr(m, "distill_steps", 0))
    if n_distill > 0:
        print(f"Distill: {n_distill} region-gate steps (teacher: frozen HRM)")
        distill_cell(model, pool, governor_hrm, groups, regions, offs, cfg,
                     device, steps=n_distill,
                     lr=float(getattr(m, "distill_lr", 1e-2)),
                     seed=cfg.train.seed)

    log: list[dict] = []
    ema = {"L_new": 0.0, "L_old": 0.0, "sparse": 0.0, "ewc": 0.0, "delta": 0.0}

    # v3: one optimizer per node — each node steps ONLY on its own loss.
    node_opts = {
        "att": torch.optim.AdamW(cell.att_node.parameters(), lr=m.lr),
        "ret": torch.optim.AdamW(cell.ret_node.parameters(), lr=m.lr),
        "int": torch.optim.AdamW(cell.int_node.parameters(), lr=m.lr),
        "ctrl": torch.optim.AdamW(cell.ctrl_node.parameters(), lr=m.lr),
    }
    train_gov = bool(getattr(m, "train_governor", False))
    if train_gov:
        gov_lr = float(getattr(m, "gov_lr", 1e-3))
        gov_opt = torch.optim.AdamW(governor_hrm.parameters(), lr=gov_lr)
        print(f"Governor: TRAINABLE (lr={gov_lr}, cap_lambda="
              f"{getattr(m, 'gov_cap_lambda', 1.0)})")
    else:
        gov_opt = None
        print("Governor: frozen (WHERE fixed)")

    print("Pre-warm kernels (first-batch cudnn/optimizer penalty) ...")
    xw = _sample_batch(m.batch_size, seed_offset=99_999).to(device)
    warmup_opt = torch.optim.AdamW(model.parameters(), lr=m.warmup_lr)
    yw = torch.rand(m.batch_size).to(device)
    warmup_opt.zero_grad(set_to_none=True)
    BCE(model(xw), yw).backward()
    warmup_opt.step()
    reset_model(model)
    torch.cuda.synchronize()

    # WHERE EMA: the raw governor allocation (fresh random model per meta
    # step) oscillates wildly (alloc 2705 -> 33 -> 1 -> 2705 in v2), making
    # the meta objective noise. The cell gets a slowly-drifting stage.
    ema_imp: torch.Tensor | None = None
    ema_frac: float | None = None

    for step in range(1, m.steps + 1):
        outs = []
        for k in range(m.meta_batch):
            seed_off = (step - 1) * m.meta_batch + k
            outs.append(raira_meta_step(model, pool, governor_hrm, groups,
                                        regions, offs, cfg, device, seed_off,
                                        rng, alloc_imp=ema_imp,
                                        alloc_frac_open=ema_frac))

        # ---- node-isolated backward (see RairawCell docstring) ----
        # ALL backwards run BEFORE any optimizer step: stepping mutates the
        # cell/governor params in place, which would invalidate the
        # retained graphs (the governor's path crosses the cell's gates).
        mean_ctrl = sum(o["loss_ctrl"] for o in outs) / len(outs)
        mean_att = sum(o["loss_att"] for o in outs) / len(outs)
        mean_ret = sum(o["loss_ret"] for o in outs) / len(outs)
        mean_int = sum(o["loss_int"] for o in outs) / len(outs)
        if train_gov:
            for p in governor_hrm.parameters():
                p.requires_grad_(False)
        for name, loss in (("ctrl", mean_ctrl), ("int", mean_int),
                           ("att", mean_att), ("ret", mean_ret)):
            if loss.grad_fn is None:
                continue  # e.g. no old tasks: ret has nothing to protect
            cell.freeze_only((name,))
            node_opts[name].zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
        cell.freeze_only(cell.NODE_NAMES)
        if train_gov:
            mean_gov = sum(o["loss_gov"] for o in outs) / len(outs)
            if mean_gov.grad_fn is not None:
                cell.freeze(True)
                for p in governor_hrm.parameters():
                    p.requires_grad_(True)
                gov_opt.zero_grad(set_to_none=True)
                mean_gov.backward()
                cell.freeze(False)
                governor_hrm.eval()
        # ---- steps (all in-place mutations happen here, after backwards) ----
        for opt in node_opts.values():
            opt.step()
        if train_gov:
            for p in governor_hrm.parameters():
                p.requires_grad_(True)
            gov_opt.step()
            for p in governor_hrm.parameters():
                p.requires_grad_(False)

        # WHERE EMA update (from the last sub-step's governor probe)
        o = outs[-1]
        if o["imp"] is not None:
            imp_k = 0.05
            ema_imp = (o["imp"] if ema_imp is None
                       else (1 - imp_k) * ema_imp + imp_k * o["imp"])
            ema_frac = (o["frac_open"] if ema_frac is None
                        else 0.95 * ema_frac + 0.05 * o["frac_open"])

        o = outs[-1]
        vals = (o["loss_new"], o["loss_old"], o["sparse_cost"],
                o["ewc_cost"], o["delta_cost"])
        for key, v in zip(ema, vals):
            ema[key] = 0.98 * ema[key] + 0.02 * float(v.detach())

        if step % 50 == 0 or step == 1:
            stats = o["mask_stats"]
            log.append({
                "step": step,
                "gov_loss": float(mean_ctrl.detach()),
                "loss_att": float(mean_att.detach()),
                "loss_ret": float(mean_ret.detach()),
                "loss_int": float(mean_int.detach()),
                "loss_new": float(o["loss_new"].detach()),
                "loss_new_before": round(o["loss_new_before"], 4),
                "loss_new_after": round(o["loss_new_after"], 4),
                "loss_old": float(o["loss_old"].detach()),
                "loss_old_before": round(o["loss_old_before"], 4),
                "loss_old_after": round(o["loss_old_after"], 4),
                "sparse_cost": float(o["sparse_cost"].detach()),
                "ewc_cost": float(o["ewc_cost"].detach()),
                "delta_cost": float(o["delta_cost"].detach()),
                "delta_cost_live": float(o["delta_cost_live"].detach()),
                "phase_scale": float(o["phase_scale"]),
                "allocated_weights": int(o["allocated_weights"]),
                "frac_active": round(o["frac_active"], 3),
                "demand": round(o.get("demand", 0.0), 3),
                **{k: round(v, 4) for k, v in stats.items()},
            })
            print(f"step {step:5d}/{m.steps}  L_new={ema['L_new']:.4f} "
                  f"(b {o['loss_new_before']:.2f}->a {o['loss_new_after']:.2f}) "
                  f"L_old={ema['L_old']:.4f}  sp={ema['sparse']:.4f} "
                  f"ewc={ema['ewc']:.4f} dW={ema['delta']:.5f}  "
                  f"| D={o.get('demand', 0.0):.2f} alloc {o['allocated_weights']}"
                  f"/{total_weights} w ({100*o['frac_active']:.0f}% regions) "
                  f"| mask mean={stats['mask_mean']:.3f} "
                  f"<0.1:{stats['frac_lt_0.1']:.2f}")

    pd.DataFrame(log).to_csv(outdir / "raira_meta_train_log.csv", index=False)
    torch.save(cell.state_dict(), outdir / "raira_cell_pretrained.pt")
    if train_gov:
        torch.save(governor_hrm.state_dict(),
                   outdir / "raira_governor_pretrained.pt")
        print(f"Saved trained governor to {outdir / 'raira_governor_pretrained.pt'}")

    info = {
        "steps": m.steps,
        "meta_batch": m.meta_batch,
        "cell_params": cell.cell_params,
        "base_params": model.count_parameters(),
        "controlled_weights": total_weights,
        "n_regions": n_regions,
        "region_size": cfg.raira.region_size,
        "max_rairaw": cfg.raira.max_rairaw,
        "h_dim": cfg.raira.h_dim,
        "intent_dim": cfg.raira.intent_dim,
        "hmem_alpha": cfg.raira.hmem_alpha,
        "alloc_blend": cfg.raira.alloc_blend,
        "governor_file": args.governor,
        "objective": {"lambda_old": m.lambda_old,
                      "lambda_sparse": m.lambda_sparse,
                      "sparse_target": m.sparse_target,
                      "lambda_ewc": getattr(m, "lambda_ewc", 0.0),
                      "lambda_delta": m.lambda_delta,
                      "lambda_after": getattr(m, "lambda_after", 0.3)},
        "memory": {"old_tasks_max": int(getattr(m, "old_tasks_max", 0)),
                   "agg": getattr(m, "memory_agg", "ewc")},
        "training_seconds": round(time.perf_counter() - start, 1),
    }
    with open(outdir / "pretrain_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("RAIRAW META-TRAINING COMPLETE")
    print("=" * 60)
    print(f"Cell params: {info['cell_params']:,} controlling "
          f"{total_weights:,} weights via {n_regions} regions "
          f"(pool max {cfg.raira.max_rairaw})")
    print(f"Last allocation: {log[-1]['allocated_weights']}/{total_weights} "
          f"weights active ({100*log[-1]['frac_active']:.0f}% regions)")
    print(f"Training took {info['training_seconds']:.0f}s")
    print(f"Saved cell to {outdir / 'raira_cell_pretrained.pt'}")


if __name__ == "__main__":
    main()