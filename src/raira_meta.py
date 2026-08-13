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
) -> tuple[dict[str, torch.Tensor], dict]:
    """Unroll `steps` RAIRAW-gated updates on `task`.

    Allocation happens ONCE at burst start: the frozen HRM governor's masks
    on a probe batch -> region importance -> top-K regions get a RAIRAW
    (WHERE). The recursive cell then emits per-weight gates for active
    regions every step (HOW); inactive regions are closed nodes (gate 0,
    no Adam moments), mirroring the live stream's enforcement.

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
            differentiable=False, g_old_list=None, mem_imp_list=None,
            mem_dir_list=None,
            hmem_list=compute_influence(g_probe, "magnitude", device=device))
        hrm_flat = torch.cat([m.detach().flatten() for m in hrm_masks])
        imp = region_importance(hrm_masks, groups, regions, device)
        hmem = HmemMemory(len(regions), alpha=0.0, device=device)
        active, blended = allocate_rairaws(
            imp, hrm_masks, hmem, len(regions), pool.max_rairaw,
            close_threshold=close_threshold)
        pool.activate(active)
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
                    hmem_influence, gfeats,
                    alloc_frac=len(active) / len(regions))
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
            if close_threshold > 0.0:
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
    p_cur, diag = raira_burst(
        model, pool, groups, regions, governor_hrm, p_cur, fam_b,
        lr=base_lr, steps=m.burst_steps, batch_size=bsize,
        seed_base=seed_off * 37 + 5, device=device, differentiable=True,
        mem_imp=mem_imp, mem_dir=mem_dir, offs=offs,
        close_threshold=getattr(m, "close_threshold", 0.02),
        hmem_influence=hmem_influence)

    x_bv = _sample_batch(bsize, seed_offset=seed_off * 43 + 11).to(device)
    y_bv = meta_task_labels(x_bv, fam_b).to(device)
    loss_new = BCE(functional_call(model, p_cur, (x_bv,)), y_bv)

    old_losses = []
    for t_i, fam_a in enumerate(old_tasks):
        x_av = _sample_batch(bsize, seed_offset=seed_off * 41 + 9 + t_i * 19).to(device)
        y_av = meta_task_labels(x_av, fam_a).to(device)
        old_losses.append(BCE(functional_call(model, p_cur, (x_av,)), y_av))
    loss_old = torch.stack(old_losses).mean() if old_losses else (
        loss_new.detach() * 0.0)

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

    gov_loss = (
        loss_new
        + m.lambda_old * loss_old
        + m.lambda_sparse * sparse_cost
        + getattr(m, "lambda_ewc", 0.0) * ewc_cost
        + m.lambda_delta * delta_cost
    )
    model.zero_grad(set_to_none=True)
    return {
        "gov_loss": gov_loss,
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
    """Node-level pretraining: teach the shared cell to reproduce the frozen
    governor's per-weight gates region by region.

    The governor is the competent per-weight policy; the cell must learn the
    feature->gate mapping (|g|, |p|, pos, I_mem + region/global context) on
    its own. Distilling each region's behavior gives every node a competent
    initialization; the meta loop then refines the recursion for the live
    horizon."""
    m = config.meta
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
                r, g_flat, p_flat, hrm_flat, None,
                float(torch.rand(1, generator=rng)[0]), gfeats,
                alloc_frac=len(pool.active) / len(regions))
            base = offs[r.group_name]
            target = hrm_flat[base + r.start: base + r.stop]
            total = total + F.mse_loss(gates_r, target)
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
    governor_hrm.load_state_dict(ckpt)
    governor_hrm.eval()
    for p in governor_hrm.parameters():
        p.requires_grad_(False)

    cell = RairawCell(h_dim=cfg.raira.h_dim, intent_dim=cfg.raira.intent_dim,
                      ctx_dim=7 + 2 + 9, fw_dim=4).to(device)
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
    print(f"HRM governor (frozen, allocation only): "
          f"{governor_hrm.governor_params():,} params from {args.governor}")
    print(f"Objective: L_new + {m.lambda_old}*L_old "
          f"+ {m.lambda_sparse}*(mean(M)-{getattr(m, 'sparse_target', 0.3)})^2 "
          f"+ {getattr(m, 'lambda_ewc', 0.0)}*mean(M*(gA_hat^2-1)) "
          f"+ {m.lambda_delta}*mean(|dW|/|W|)")
    print(f"Meta: {m.meta_batch} parallel steps x {m.steps} updates, "
          f"warmup={m.warmup_batches}, burst={m.burst_steps}x, batch={m.batch_size}")
    print(f"H_MEM: empty per meta step; random influence context "
          f"(alpha={cfg.raira.hmem_alpha} live)")
    reset_model(model)

    gov_opt = torch.optim.AdamW(cell.parameters(), lr=m.lr)
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

    for step in range(1, m.steps + 1):
        outs = []
        for k in range(m.meta_batch):
            seed_off = (step - 1) * m.meta_batch + k
            outs.append(raira_meta_step(model, pool, governor_hrm, groups,
                                        regions, offs, cfg, device, seed_off,
                                        rng))

        total_loss = sum(o["gov_loss"] for o in outs) / len(outs)
        gov_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        gov_opt.step()

        o = outs[-1]
        vals = (o["loss_new"], o["loss_old"], o["sparse_cost"],
                o["ewc_cost"], o["delta_cost"])
        for key, v in zip(ema, vals):
            ema[key] = 0.98 * ema[key] + 0.02 * float(v.detach())

        if step % 50 == 0 or step == 1:
            stats = o["mask_stats"]
            log.append({
                "step": step,
                "gov_loss": float(total_loss.detach()),
                "loss_new": float(o["loss_new"].detach()),
                "loss_old": float(o["loss_old"].detach()),
                "sparse_cost": float(o["sparse_cost"].detach()),
                "ewc_cost": float(o["ewc_cost"].detach()),
                "delta_cost": float(o["delta_cost"].detach()),
                "delta_cost_live": float(o["delta_cost_live"].detach()),
                "phase_scale": float(o["phase_scale"]),
                "allocated_weights": int(o["allocated_weights"]),
                "frac_active": round(o["frac_active"], 3),
                **{k: round(v, 4) for k, v in stats.items()},
            })
            print(f"step {step:5d}/{m.steps}  L_new={ema['L_new']:.4f} "
                  f"L_old={ema['L_old']:.4f}  sp={ema['sparse']:.4f} "
                  f"ewc={ema['ewc']:.4f} dW={ema['delta']:.5f}  "
                  f"| alloc {o['allocated_weights']}/{total_weights} "
                  f"w ({100*o['frac_active']:.0f}% regions) "
                  f"| mask mean={stats['mask_mean']:.3f} "
                  f"<0.1:{stats['frac_lt_0.1']:.2f}")

    pd.DataFrame(log).to_csv(outdir / "raira_meta_train_log.csv", index=False)
    torch.save(cell.state_dict(), outdir / "raira_cell_pretrained.pt")

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
                      "lambda_delta": m.lambda_delta},
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