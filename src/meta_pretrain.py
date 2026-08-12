"""Meta-pretraining of the HRM intent network (Phase 2, first stage).

The user requirement: train the tiny HRM intent network FIRST so it can
decide retention.

Objective (making parameter modification itself expensive -- prevents the
"M = 1 everywhere" collapse):

    L = L_new(after burst)
      + lambda_old  * L_old(after burst)       # penalize knowledge damage
      + lambda_sparse * mean(M)                # penalize allowing gates open
      + lambda_delta  * mean(|W_after-W_before| / (|W_before|+eps))

The intent network must therefore solve a real constrained problem:
  which parameters should I allow to change to learn the new task,
  while changing as little of the existing model as possible?

Method:
- task pair (A, B) sampled from the same geometry families as the stream
  with RANDOMIZED boundary shifts (no task-ID reaches base model or governor)
- warm up a fresh base model on A (old knowledge installed)
- a BURST of governor-gated updates on B:  W_{s+1} = W_s - lr * M_s * grad
- gradient path: losses after the burst depend on all gated steps, and the
  gates depend on the governor -> first-order meta-learning unroll
  (FOMAML-style: inner-loop gradients detached; no double backward through
  attention, whose efficient SDPA backend has no second derivative)
- meta-batching: K super-parallel meta-steps per governor optimizer step
  (fresh base models, independent graphs, averaged gradients) to use the GPU
  properly instead of serial tiny kernels.

The governor is FROZEN afterwards (frozen governance core, section 132.3)
and only evaluated through the Phase-2 continual stream (experiment2.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.func import functional_call

from .config import load_config
from .experiment import set_seed
from .hrm import (
    build_module_groups, DirectGateGovernor, HRMIntentGovernor, mask_stats,
)
from .model import TinyNumericTransformer

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

BCE = nn.BCEWithLogitsLoss()


def reset_model(model: nn.Module) -> None:
    """Reinitialize all parameter tensors (fresh base model state)."""

    def _reset(module: nn.Module) -> None:
        fn = getattr(module, "_reset_parameters", None) or getattr(
            module, "reset_parameters", None)
        if fn is not None:
            fn()

    model.apply(_reset)


# ---------------------------------------------------------------------------
# Meta task universe: the 5 geometry families with randomized boundaries.
# ---------------------------------------------------------------------------

def _sample_batch(n: int, seed_offset: int = 0) -> torch.Tensor:
    """Fresh uniform x in [-1,1]^2 from a per-call generator."""
    gen = torch.Generator().manual_seed(seed_offset)
    return torch.rand(n, 2, generator=gen) * 2.0 - 1.0


def sample_chunk(
    n_batches: int, n: int, seed_base: int, device: torch.device
) -> torch.Tensor:
    """All batches of a phase in ONE generator call + ONE transfer (GPU-width).

    Deterministic per (n_batches, n, seed_base); the exact draws differ from
    the previous step-by-step scheme (same seeds, different splitting).
    Returns (n_batches * n, 2) on the target device.
    """
    x = _sample_batch(n_batches * n, seed_offset=seed_base)
    return x.to(device)


def make_meta_task(rng: torch.Generator) -> tuple[str, int, float]:
    """Pick (family_name, kind, boundary_param) for a meta task."""
    families = ["horizontal", "vertical", "circle", "diagonal", "xor"]
    family = families[int(torch.randint(0, len(families), (1,), generator=rng)[0])]
    if family == "circle":
        return family, 2, float(torch.rand(1, generator=rng)[0]) * 0.3 + 0.4
    if family == "xor":
        return family, 4, 0.0
    shift = (float(torch.rand(1, generator=rng)[0]) - 0.5) * 0.6
    kind = {"horizontal": 0, "vertical": 1, "diagonal": 3}[family]
    return family, kind, shift


def meta_task_labels(x: torch.Tensor, task) -> torch.Tensor:
    """Labels for a meta task (same geometry families as the stream)."""
    x1, x2 = x[:, 0], x[:, 1]
    _, kind, param = task
    if kind == 0:    # horizontal: x2 > shift
        return (x2 > param).float()
    if kind == 1:    # vertical: x1 > shift
        return (x1 > param).float()
    if kind == 2:    # circle radius
        return (x1 ** 2 + x2 ** 2 > param ** 2).float()
    if kind == 3:    # diagonal: x1 + x2 > shift
        return (x1 + x2 > param).float()
    return ((x1 > 0) != (x2 > 0)).float()  # xor


# ---------------------------------------------------------------------------
# Gated burst (functional, first-order differentiable unroll)
# ---------------------------------------------------------------------------

def gated_burst(
    model: nn.Module,
    governor: HRMIntentGovernor,
    groups,
    p_cur: dict[str, torch.Tensor],
    task,
    lr: float,
    steps: int,
    batch_size: int,
    seed_base: int,
    device: torch.device,
    differentiable: bool,
    ungated: bool = False,
    g_old: list[torch.Tensor] | None = None,
    p0: dict[str, torch.Tensor] | None = None,
    second_order: bool = False,
) -> tuple[dict[str, torch.Tensor], list[dict], float]:
    """Unroll `steps` governor-gated updates on `task`.

    Returns (final params dict, per-step mask stats, mean gate across steps).
    """
    mask_log: list[dict] = []
    gate_means = []
    ewc_costs = []
    g_old_hat = None
    g_old_hat_all = None
    if g_old is not None:
        g_old_hat = [g.detach() / (g.abs().mean() + 1e-12) for g in g_old]
        g_old_hat_all = torch.cat([h.flatten() for h in g_old_hat])
    xb = sample_chunk(steps, batch_size, seed_base, device)
    yb = meta_task_labels(xb, task)
    for s in range(steps):
        x = xb[s * batch_size:(s + 1) * batch_size]
        y = yb[s * batch_size:(s + 1) * batch_size]
        pred = functional_call(model, p_cur, (x,))
        loss = BCE(pred, y)

        grad_list = torch.autograd.grad(
            loss, [p_cur[g.name] for g in groups],
            create_graph=second_order, allow_unused=False)

        if ungated:
            masks = [torch.ones_like(p_cur[g.name].flatten()) for g in groups]
        else:
            hist = None
            if p0 is not None:
                hist = [(p_cur[g.name] - p0[g.name]).abs()
                        / (p0[g.name].abs() + 1e-8) for g in groups]
            masks = governor.gate_from_state(
                p_cur, grad_list, groups, x, y, loss, device,
                differentiable=differentiable, g_old_list=g_old,
                hist_list=hist)

        mask_log.append(mask_stats(masks))
        gate_means.append(torch.cat([m.flatten() for m in masks]).mean())
        if g_old_hat_all is not None:
            m_all = torch.cat([m.flatten() for m in masks])
            # zero-mean selectivity bias: mean(m*(gA_hat^2 - 1))
            # = 0 for ANY uniform gate level -> shapes the distribution
            # (close A-sensitive / open A-insensitive weights) without
            # fighting the sparse-target level term
            ewc_costs.append((m_all * g_old_hat_all ** 2).mean() - m_all.mean())

        for group, m, g in zip(groups, masks, grad_list):
            p_cur[group.name] = p_cur[group.name] - lr * m.reshape(g.shape) * g

    mean_gate = torch.stack(gate_means).mean()
    mean_ewc = torch.stack(ewc_costs).mean() if ewc_costs else None
    return p_cur, mask_log, mean_gate, mean_ewc


def warmup_batches(
    model: nn.Module,
    task,
    n_batches: int,
    batch_size: int,
    lr: float,
    seed_base: int,
    device: torch.device,
) -> None:
    """Plain (ungated) training on task A until it is well installed."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    xb = sample_chunk(n_batches, batch_size, seed_base, device)
    yb = meta_task_labels(xb, task)
    for k in range(n_batches):
        x = xb[k * batch_size:(k + 1) * batch_size]
        y = yb[k * batch_size:(k + 1) * batch_size]
        loss = BCE(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


# ---------------------------------------------------------------------------
# Meta step
# ---------------------------------------------------------------------------

def meta_step(
    model: nn.Module,
    governor: HRMIntentGovernor,
    groups,
    config,
    device: torch.device,
    seed_off: int,
    rng: torch.Generator,
) -> dict:
    """One meta step (warmup A -> gated burst on B -> governor objective).

    Returns tensors kept attached to the governor graph:
        gov_loss, loss_new, loss_old, sparse_cost, delta_cost
    plus scalar diagnostics.
    """
    m = config.meta
    base_lr = m.burst_lr
    warmup_lr = getattr(m, "warmup_lr", m.lr)
    bsize = m.batch_size

    torch.manual_seed(100_000 + seed_off * 7)
    reset_model(model)
    fam_a = make_meta_task(rng)
    fam_b = make_meta_task(rng)

    warmup_batches(model, fam_a, m.warmup_batches, bsize, warmup_lr,
                   seed_base=seed_off * 31, device=device)
    model.zero_grad(set_to_none=True)

    # current params as functional leaves (no link to live model)
    p_cur = {g.name: g.param.detach().clone().requires_grad_(True)
             for g in groups}
    p0 = {g.name: t.detach().clone() for g, t in zip(groups, [p_cur[g.name]
                                                              for g in groups])}

    # pre-burst OLD-task sensitivity: grad of L_A w.r.t. pre-burst params
    # (the EWC-diagonal importance measure; per-weight feature for the
    # governor so selective gating is directly learnable)
    x_as = _sample_batch(bsize, seed_offset=seed_off * 45 + 13).to(device)
    y_as = meta_task_labels(x_as, fam_a).to(device)
    aloss = BCE(functional_call(model, p_cur, (x_as,)), y_as)
    g_old = torch.autograd.grad(
        aloss, [p_cur[g.name] for g in groups],
        create_graph=False, allow_unused=False)

    p_cur, mask_log, mean_gate, mean_ewc = gated_burst(
        model, governor, groups, p_cur, fam_b,
        lr=base_lr, steps=m.burst_steps, batch_size=bsize,
        seed_base=seed_off * 37 + 5, device=device, differentiable=True,
        g_old=g_old, p0=p0, second_order=m.second_order)

    x_av = _sample_batch(bsize, seed_offset=seed_off * 41 + 9).to(device)
    y_av = meta_task_labels(x_av, fam_a).to(device)
    x_bv = _sample_batch(bsize, seed_offset=seed_off * 43 + 11).to(device)
    y_bv = meta_task_labels(x_bv, fam_b).to(device)

    loss_new = BCE(functional_call(model, p_cur, (x_bv,)), y_bv)
    loss_old = BCE(functional_call(model, p_cur, (x_av,)), y_av)

    # parameter-change cost: mean relative |dW|/|W_before| over all weights
    rel_changes = []
    for g in groups:
        d = (p_cur[g.name] - p0[g.name]).abs()
        rel = d / (p0[g.name].abs() + 1e-8)
        rel_changes.append(rel.mean())
    mean_rel_change = torch.stack(rel_changes).mean()

    sparse_cost = (mean_gate - getattr(m, "sparse_target", 0.3)) ** 2
    ewc_cost = mean_ewc if mean_ewc is not None else torch.zeros((), device=device)

    # objective per user spec: parameter modification itself is expensive
    gov_loss = (
        loss_new
        + m.lambda_old * loss_old
        + m.lambda_sparse * sparse_cost
        + getattr(m, "lambda_ewc", 0.0) * ewc_cost
        + m.lambda_delta * mean_rel_change
    )

    model.zero_grad(set_to_none=True)
    return {
        "gov_loss": gov_loss,
        "loss_new": loss_new,
        "loss_old": loss_old,
        "sparse_cost": sparse_cost,
        "ewc_cost": ewc_cost,
        "delta_cost": mean_rel_change,
        "mask_stats": mask_log[-1],
    }


# ---------------------------------------------------------------------------
# Post-training paired sanity check (gated burst vs ungated burst)
# ---------------------------------------------------------------------------

def meta_evaluate(
    model: nn.Module,
    governor: HRMIntentGovernor,
    groups,
    config,
    device: torch.device,
    rng: torch.Generator,
    num_pairs: int,
) -> tuple[pd.DataFrame, dict]:
    """For each pair (A, B): warm up A, burst on B gated or ungated, measure."""
    m = config.meta
    base_lr = m.burst_lr
    warmup_lr = getattr(m, "warmup_lr", m.lr)
    bsize = m.batch_size

    rows = []
    for p in range(num_pairs):
        torch.manual_seed(500_000 + p * 13)
        reset_model(model)
        fam_a = make_meta_task(rng)
        fam_b = make_meta_task(rng)

        warmup_batches(model, fam_a, m.warmup_batches, bsize, warmup_lr,
                       seed_base=p * 131, device=device)
        model.zero_grad(set_to_none=True)

        xa_te = _sample_batch(512, seed_offset=p * 151).to(device)
        ya_te = meta_task_labels(xa_te, fam_a).to(device)
        with torch.no_grad():
            acc_a_before = (model(xa_te) > 0).float().eq(ya_te).float().mean().item()

        # ---- gated burst (frozen governor) ----
        p_g = {g.name: g.param.detach().clone().requires_grad_(True)
               for g in groups}
        x_as = _sample_batch(bsize, seed_offset=p * 161).to(device)
        y_as = meta_task_labels(x_as, fam_a).to(device)
        aloss = BCE(functional_call(model, p_g, (x_as,)), y_as)
        g_old = torch.autograd.grad(
            aloss, [p_g[g.name] for g in groups],
            create_graph=False, allow_unused=False)
        p0 = {g.name: t.detach().clone() for g, t in
              zip(groups, [p_g[g.name] for g in groups])}
        p_g, _, _, _ = gated_burst(
            model, governor, groups, p_g, fam_b,
            lr=base_lr, steps=m.burst_steps, batch_size=bsize,
            seed_base=p * 157 + 5, device=device, differentiable=False,
            g_old=g_old, p0=p0)
        with torch.no_grad():
            pred_a = functional_call(model, p_g, (xa_te,))
            acc_a_gated = (pred_a > 0).float().eq(ya_te).float().mean().item()
        xb_te = _sample_batch(512, seed_offset=p * 159).to(device)
        yb_te = meta_task_labels(xb_te, fam_b).to(device)
        with torch.no_grad():
            pred_b = functional_call(model, p_g, (xb_te,))
            acc_b_gated = (pred_b > 0).float().eq(yb_te).float().mean().item()

        # ---- ungated burst (ordinary updates = Phase 1 behavior) ----
        p_u = {g.name: g.param.detach().clone().requires_grad_(True)
               for g in groups}
        p_u, _, _, _ = gated_burst(
            model, governor, groups, p_u, fam_b,
            lr=base_lr, steps=m.burst_steps, batch_size=bsize,
            seed_base=p * 157 + 5, device=device, differentiable=False,
            ungated=True)
        with torch.no_grad():
            pred_a = functional_call(model, p_u, (xa_te,))
            acc_a_ungated = (pred_a > 0).float().eq(ya_te).float().mean().item()
            pred_b = functional_call(model, p_u, (xb_te,))
            acc_b_ungated = (pred_b > 0).float().eq(yb_te).float().mean().item()

        rows.append({
            "pair": p,
            "acc_a_before": acc_a_before,
            "acc_a_gated": acc_a_gated,
            "acc_a_ungated": acc_a_ungated,
            "acc_b_gated": acc_b_gated,
            "acc_b_ungated": acc_b_ungated,
        })

    df = pd.DataFrame(rows)
    summary = {
        "pairs": num_pairs,
        "acc_a_before_mean": float(df.acc_a_before.mean()),
        "acc_a_gated_mean": float(df.acc_a_gated.mean()),
        "acc_a_ungated_mean": float(df.acc_a_ungated.mean()),
        "acc_b_gated_mean": float(df.acc_b_gated.mean()),
        "acc_b_ungated_mean": float(df.acc_b_ungated.mean()),
        "old_task_retention_gain": float(df.acc_a_gated.mean() - df.acc_a_ungated.mean()),
        "new_task_cost": float(df.acc_b_gated.mean() - df.acc_b_ungated.mean()),
    }
    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="HRM intent governor meta-pretraining")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline2.yaml"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results_governor"))
    parser.add_argument("--sparse-target", type=float, default=None,
                        help="plasticity budget: desired mean(M) (Pareto sweep)")
    parser.add_argument("--steps", type=int, default=None,
                        help="override governor optimizer update count")
    parser.add_argument("--mode", choices=["mlp", "direct"], default="mlp",
                        help="mlp = shared gate network (phase 2); "
                             "direct = one trainable gate per weight (phase 2b)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        if f.is_file():
            f.unlink()

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(cfg.meta, "second_order", False):
        # true MAML unroll needs the gradient of the inner gradients;
        # efficient SDPA backends have no double backward -> math attention
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cudnn.benchmark = True  # speed over strict determinism
    print(f"Device: {device}")

    # plasticity-budget override: --sparse-target sweeps the objective's
    # desired mean(M), tracing the retention-vs-new-task Pareto frontier
    if args.sparse_target is not None:
        cfg.meta.sparse_target = args.sparse_target
    if args.steps is not None:
        cfg.meta.steps = args.steps

    model = TinyNumericTransformer(
        input_dim=cfg.model.input_dim,
        seq_len=cfg.model.seq_len,
        embedding_dim=cfg.model.embedding_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    groups = build_module_groups(model)
    total_weights = sum(g.size for g in groups)

    governor = HRMIntentGovernor(
        num_groups=len(groups),
        granularity=cfg.governor.granularity,
        hidden_dim=cfg.governor.hidden_dim,
        refine_steps=cfg.governor.refine_steps,
        init_mask=cfg.governor.init_mask,
    ).to(device)
    if args.mode == "direct":
        governor = DirectGateGovernor(
            groups, init_mask=cfg.governor.init_mask).to(device)
    m = cfg.meta

    print("=" * 60)
    print("HRM INTENT GOVERNOR META-PRETRAINING")
    print("=" * 60)
    print(f"Base model params: {model.count_parameters():,}  "
          f"totally controlled weights: {total_weights:,}")
    print(f"Mode: {args.mode}  "
          f"governor params: {governor.governor_params():,}")
    print(f"Objective: L_new + {m.lambda_old}*L_old "
          f"+ {m.lambda_sparse}*(mean(M)-{getattr(m, 'sparse_target', 0.3)})^2 "
          f"+ {getattr(m, 'lambda_ewc', 0.0)}*mean(M*(gA_hat^2-1)) "
          f"+ {m.lambda_delta}*mean(|dW|/|W|)")
    print(f"Meta-batching: {m.meta_batch} parallel steps per governor update, "
          f"{m.steps} updates, warmup={m.warmup_batches} burst={m.burst_steps} "
          f"batch={m.batch_size}\n")
    reset_model(model)

    gov_opt = torch.optim.AdamW(governor.parameters(), lr=m.lr)
    rng = torch.Generator().manual_seed(7_777 + cfg.train.seed)
    start = time.perf_counter()

    log: list[dict] = []
    ema = {"L_new": 0.0, "L_old": 0.0, "sparse": 0.0, "ewc": 0.0, "delta": 0.0}

    for step in range(1, m.steps + 1):
        outs = []
        for k in range(m.meta_batch):
            seed_off = (step - 1) * m.meta_batch + k
            outs.append(meta_step(model, governor, groups, cfg, device, seed_off, rng))

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
                **{k: round(v, 4) for k, v in stats.items()},
            })
            print(f"step {step:5d}/{m.steps}  L_new={ema['L_new']:.4f} "
                  f"L_old={ema['L_old']:.4f}  sp={ema['sparse']:.4f} "
                  f"ewc={ema['ewc']:.4f} dW={ema['delta']:.5f}  "
                  f"| mask mean={stats['mask_mean']:.3f} "
                  f"min={stats['mask_min']:.3f} max={stats['mask_max']:.3f} "
                  f"<0.1:{stats['frac_lt_0.1']:.2f} >0.9:{stats['frac_gt_0.9']:.2f}")

    pd.DataFrame(log).to_csv(outdir / "meta_train_log.csv", index=False)

    # --- post-training sanity check (paired) -------------------------------
    torch.manual_seed(99_999)
    df_ev, summary = meta_evaluate(model, governor, groups, cfg, device, rng,
                                   m.eval_pairs)
    df_ev.to_csv(outdir / "meta_eval.csv", index=False)

    torch.save(governor.state_dict(), outdir / "governor_pretrained.pt")

    info = {
        "steps": m.steps,
        "meta_batch": m.meta_batch,
        "governor_params": governor.governor_params(),
        "base_params": model.count_parameters(),
        "controlled_weights": total_weights,
        "granularity": governor.granularity if hasattr(governor, "granularity") else "weight",
        "mode": args.mode,
        "group_names": [g.name for g in groups],
        "objective": {"lambda_old": m.lambda_old, "lambda_sparse": m.lambda_sparse,
                      "sparse_target": m.sparse_target,
                      "lambda_ewc": getattr(m, "lambda_ewc", 0.0),
                      "lambda_delta": m.lambda_delta},
        "training_seconds": round(time.perf_counter() - start, 1),
        "mask_stats_last": {k: round(v, 4) for k, v in log[-1].items()
                            if k.startswith(("mask", "frac"))}
                        if log else {},
        "meta_eval": {k: round(float(v), 4) for k, v in summary.items()},
    }
    with open(outdir / "pretrain_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("GOVERNOR META-PRETRAINING COMPLETE")
    print("=" * 60)
    print(f"Governor params: {info['governor_params']:,} controlling "
          f"{total_weights:,} weights "
          f"({getattr(governor, 'granularity', args.mode)}-level)")
    print(f"Pairs evaluated: {summary['pairs']}")
    print(f"  old-task acc BEFORE B burst : {summary['acc_a_before_mean']:.2%}")
    print(f"  old-task acc AFTER  gated   : {summary['acc_a_gated_mean']:.2%}")
    print(f"  old-task acc AFTER  ungated : {summary['acc_a_ungated_mean']:.2%}")
    print(f"  retention gain (mask vs plain): {summary['old_task_retention_gain']:+.2%}")
    print(f"  new-task acc gated   : {summary['acc_b_gated_mean']:.2%}")
    print(f"  new-task acc ungated : {summary['acc_b_ungated_mean']:.2%}")
    print(f"  new-task cost of protecting: {summary['new_task_cost']:+.2%}")
    print(f"Training took {info['training_seconds']:.0f}s")
    print(f"Saved governor to {outdir / 'governor_pretrained.pt'}")


if __name__ == "__main__":
    main()