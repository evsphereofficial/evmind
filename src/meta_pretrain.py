"""Meta-pretraining of the HRM intent network (Phase 2, first stage).

The user requirement: train the tiny HRM intent network FIRST so it can
decide retention.

Method (honest, disclosed in MASTER.md):
- The intent governor is trained with a masked-update meta-learning objective:
  * sample task pair (A, B) from the same geometric boundary families that
    the measured stream uses, with RANDOMIZED boundary shifts (no task-ID is
    ever given to base model or governor; the governor must infer retention
    intent from input geometry + parameter/gradient state).
  * warm up a fresh base model on A (mimics "old knowledge installed"),
  * perform ONE governor-gated update on B:  W' = W - lr * M * grad,
  * governor loss = new-task loss(W') + beta * old-task loss(W')
    -> the intent network learns to gate updates so old knowledge survives
       while new knowledge is still acquired.
- The governor is FROZEN afterwards (frozen governance core, §132.3) and is
  only evaluated through the Phase-2 continual stream (experiment2.py).
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
from .hrm import build_module_groups, compute_governor_features, HRMIntentGovernor
from .model import TinyNumericTransformer

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

BCE = nn.BCEWithLogitsLoss()


# ---------------------------------------------------------------------------
# Meta task universe: the 5 geometry families with randomized boundaries.
# ---------------------------------------------------------------------------

def _sample_batch(n: int, seed_offset: int = 0) -> torch.Tensor:
    """Fresh uniform x in [-1,1]^2 from a per-call generator."""
    gen = torch.Generator().manual_seed(seed_offset)
    return torch.rand(n, 2, generator=gen) * 2.0 - 1.0


def make_meta_task(rng: torch.Generator) -> tuple[str, int, float]:
    """Pick (family, boundary-variant) for a meta task.

    Returns (family_name, kind, param) where kind selects the label fn.
    """
    families = ["horizontal", "vertical", "circle", "diagonal", "xor"]
    family = families[int(torch.randint(0, len(families), (1,), generator=rng)[0])]
    if family == "circle":
        return family, 2, float(torch.rand(1, generator=rng)[0]) * 0.3 + 0.4
    if family == "xor":
        return family, 4, 0.0
    # linear boundaries get a random shift in [-0.3, 0.3]
    shift = (float(torch.rand(1, generator=rng)[0]) - 0.5) * 0.6
    kind = {"horizontal": 0, "vertical": 1, "diagonal": 3}[family]
    return family, kind, shift


def meta_task_labels(
    x: torch.Tensor, family: str, kind: int, param: float
) -> torch.Tensor:
    """Labels for a meta task (same geometry families as the stream)."""
    x1, x2 = x[:, 0], x[:, 1]
    if kind == 0:   # horizontal: x2 > shift
        return (x2 > param).float()
    if kind == 1:   # vertical: x1 > shift
        return (x1 > param).float()
    if kind == 2:   # circle radius
        return (x1 ** 2 + x2 ** 2 > param ** 2).float()
    if kind == 3:   # diagonal: x1 + x2 > shift
        return (x1 + x2 > param).float()
    return ((x1 > 0) != (x2 > 0)).float()  # xor


# ---------------------------------------------------------------------------
# Meta-training step
# ---------------------------------------------------------------------------

def masked_step_forward(
    model: nn.Module,
    governor: HRMIntentGovernor,
    groups,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    base_lr: float,
    device: torch.device,
    differentiable: bool,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """One governor-gated update; returns new_params dict + feature tensors.

    differentiable=True keeps the gate graph for meta-gradients (pretraining);
    differentiable=False returns detached masks (stream/eval usage not needed
    here -- kept for symmetry with experiment2 flow if reused).
    """
    loss_b = BCE(model(x_b), y_b)
    loss_b.backward()

    gfeats, cfeats = compute_governor_features(
        x_b, y_b, loss_b, model, groups, device)

    with torch.set_grad_enabled(differentiable):
        masks = governor(gfeats, cfeats)

    new_params = {}
    for group, m in zip(groups, masks):
        gd = group.param.grad.detach()
        new_params[group.name] = group.param.detach() - base_lr * m * gd
    return new_params, masks, loss_b


def meta_step(
    model: nn.Module,
    governor: HRMIntentGovernor,
    groups,
    config,
    device: torch.device,
    step_counter: int,
    rng: torch.Generator,
) -> dict[str, float]:
    """One full meta-training step (warmup A -> gated update B -> gov loss)."""
    base_lr = config.meta.lr
    bsize = config.meta.batch_size
    val_size = config.meta.batch_size

    # fresh base model state
    torch.manual_seed(100_000 + step_counter * 7)
    model.apply(lambda m: m.reset_parameters())
    base_opt = torch.optim.AdamW(model.parameters(), lr=base_lr)

    # --- warmup on task A --------------------------------------------------
    fam_a = make_meta_task(rng)
    for k in range(config.meta.warmup_batches):
        x = _sample_batch(bsize, seed_offset=step_counter * 31 + k).to(device)
        y = meta_task_labels(x, *fam_a).to(device)
        loss = BCE(model(x), y)
        base_opt.zero_grad(set_to_none=True)
        loss.backward()
        base_opt.step()
    model.zero_grad(set_to_none=True)

    # --- one gated update on task B ----------------------------------------
    fam_b = make_meta_task(rng)
    x_b = _sample_batch(bsize, seed_offset=step_counter * 37 + 5).to(device)
    y_b = meta_task_labels(x_b, *fam_b).to(device)
    new_params, masks, _ = masked_step_forward(
        model, governor, groups, x_b, y_b, base_lr, device, differentiable=True)

    x_av = _sample_batch(val_size, seed_offset=step_counter * 41 + 9).to(device)
    y_av = meta_task_labels(x_av, *fam_a).to(device)
    x_bv = _sample_batch(val_size, seed_offset=step_counter * 43 + 11).to(device)
    y_bv = meta_task_labels(x_bv, *fam_b).to(device)

    loss_b_val = BCE(functional_call(model, new_params, (x_bv,)), y_bv)
    loss_a_val = BCE(functional_call(model, new_params, (x_av,)), y_av)

    entropy = HRMIntentGovernor.mask_entropy(masks)
    gov_loss = loss_b_val + config.meta.beta_old * loss_a_val \
        + config.meta.entropy_weight * entropy

    model.zero_grad(set_to_none=True)
    base_opt.zero_grad(set_to_none=True)
    return gov_loss, loss_b_val, loss_a_val, entropy, masks.detach()


def meta_evaluate(
    model: nn.Module,
    governor: HRMIntentGovernor,
    groups,
    config,
    device: torch.device,
    rng: torch.Generator,
    num_pairs: int,
) -> dict:
    """Paired post-training sanity check: gated update vs ungated update.

    For each pair (A, B): warm up A, apply ONE update on B either gated
    (governor) or ungated (baseline-like), then measure task A accuracy
    before and after, and task B accuracy after.
    """
    bsize = config.meta.batch_size
    base_lr = config.meta.lr

    rows = []
    for p in range(num_pairs):
        torch.manual_seed(500_000 + p * 13)
        model.apply(lambda m: m.reset_parameters())
        fam_a = make_meta_task(rng)
        fam_b = make_meta_task(rng)
        base_opt = torch.optim.AdamW(model.parameters(), lr=base_lr)

        for k in range(config.meta.warmup_batches):
            x = _sample_batch(bsize, seed_offset=p * 131 + k).to(device)
            y = meta_task_labels(x, *fam_a).to(device)
            loss = BCE(model(x), y)
            base_opt.zero_grad(set_to_none=True)
            loss.backward()
            base_opt.step()
        model.zero_grad(set_to_none=True)

        xa_te = _sample_batch(512, seed_offset=p * 151).to(device)
        ya_te = meta_task_labels(xa_te, *fam_a).to(device)
        with torch.no_grad():
            acc_a_before = (model(xa_te) > 0).float().eq(ya_te).float().mean().item()

        x_b = _sample_batch(bsize, seed_offset=p * 157 + 5).to(device)
        y_b = meta_task_labels(x_b, *fam_b).to(device)

        # gated (governor, frozen)
        with torch.no_grad():
            loss_b = BCE(model(x_b), y_b)
            loss_b.backward()
            gfeats, cfeats = compute_governor_features(x_b, y_b, loss_b, model, groups, device)
            masks = governor(gfeats, cfeats)
            new_params = {}
            for group, m in zip(groups, masks):
                gd = group.param.grad.detach()
                new_params[group.name] = group.param.detach() - base_lr * m * gd
            pred_a = functional_call(model, new_params, (xa_te,))
            pred_b = functional_call(model, new_params, (x_b,))
            acc_a_gated = (pred_a > 0).float().eq(ya_te).float().mean().item()
            acc_b_gated = (pred_b > 0).float().eq(y_b).float().mean().item()
            model.zero_grad(set_to_none=True)

        # ungated (ordinary update = Phase-1 behavior)
        loss_b = BCE(model(x_b), y_b)
        loss_b.backward()
        with torch.no_grad():
            new_params = {}
            for group in groups:
                gd = group.param.grad.detach()
                new_params[group.name] = group.param.detach() - base_lr * gd
            pred_a = functional_call(model, new_params, (xa_te,))
            acc_a_ungated = (pred_a > 0).float().eq(ya_te).float().mean().item()
            acc_b_ungated = (pred_b > 0).float().eq(y_b).float().mean().item()
        model.zero_grad(set_to_none=True)

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
        "acc_a_before_mean": df.acc_a_before.mean(),
        "acc_a_gated_mean": df.acc_a_gated.mean(),
        "acc_a_ungated_mean": df.acc_a_ungated.mean(),
        "acc_b_gated_mean": df.acc_b_gated.mean(),
        "acc_b_ungated_mean": df.acc_b_ungated.mean(),
        "old_task_retention_gain": df.acc_a_gated.mean() - df.acc_a_ungated.mean(),
    }
    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="HRM intent governor meta-pretraining")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline2.yaml"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results_governor"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        if f.is_file():
            f.unlink()

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- base model (host node) + governor (intent network) -----------------
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

    governor = HRMIntentGovernor(
        num_groups=len(groups),
        hidden_dim=cfg.governor.hidden_dim,
        refine_steps=cfg.governor.refine_steps,
        init_mask=cfg.governor.init_mask,
    ).to(device)
    print(f"Base model params: {model.count_parameters():,}  "
          f"Governor (intent net) params: {governor.governor_params():,}")
    model.apply(lambda m: m.reset_parameters())

    gov_opt = torch.optim.AdamW(governor.parameters(), lr=cfg.meta.lr)
    rng = torch.Generator().manual_seed(7_777 + cfg.train.seed)
    start = time.perf_counter()

    log: list[dict] = []
    ema = {"L_new": 0.0, "L_old": 0.0, "H": 0.0}
    for step in range(1, cfg.meta.steps + 1):
        gov_loss, lnew, lold, ent, masks = meta_step(
            model, governor, groups, cfg, device, step, rng)
        gov_opt.zero_grad(set_to_none=True)
        gov_loss.backward()
        gov_opt.step()

        for k, v in zip(ema, (lnew, lold, ent)):
            ema[k] = 0.98 * ema[k] + 0.02 * float(v)

        if step % 100 == 0 or step == 1:
            log.append({
                "step": step,
                "gov_loss": float(gov_loss),
                "loss_new": float(lnew),
                "loss_old": float(lold),
                "entropy": float(ent),
                "mask_mean": float(masks.mean()),
            })
            print(f"step {step:6d}/{cfg.meta.steps}  L_new={ema['L_new']:.4f} "
                  f"L_old={ema['L_old']:.4f}  H={ema['H']:.4f} "
                  f"mask_mean={float(masks.mean()):.3f}")

    pd.DataFrame(log).to_csv(outdir / "meta_train_log.csv", index=False)

    # --- post-training sanity check (paired) -------------------------------
    torch.manual_seed(99_999)
    df_ev, summary = meta_evaluate(model, governor, groups, cfg, device, rng,
                                   cfg.meta.eval_pairs)
    df_ev.to_csv(outdir / "meta_eval.csv", index=False)

    torch.save(governor.state_dict(), outdir / "governor_pretrained.pt")

    info = {
        "steps": cfg.meta.steps,
        "governor_params": governor.governor_params(),
        "base_params": model.count_parameters(),
        "num_groups": len(groups),
        "group_names": [g.name for g in groups],
        "training_seconds": round(time.perf_counter() - start, 1),
        "meta_eval": {k: round(float(v), 4) for k, v in summary.items()},
    }
    with open(outdir / "pretrain_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("GOVERNOR META-PRETRAINING COMPLETE")
    print("=" * 60)
    print(f"Governor params: {info['governor_params']:,}  groups: {len(groups)}")
    print(f"Pairs evaluated: {summary['pairs']}")
    print(f"  old-task acc BEFORE B update : {summary['acc_a_before_mean']:.2%}")
    print(f"  old-task acc AFTER  gated    : {summary['acc_a_gated_mean']:.2%}")
    print(f"  old-task acc AFTER  ungated  : {summary['acc_a_ungated_mean']:.2%}")
    print(f"  retention gain (mask vs plain): {summary['old_task_retention_gain']:+.2%}")
    print(f"  new-task acc gated   : {summary['acc_b_gated_mean']:.2%}")
    print(f"  new-task acc ungated : {summary['acc_b_ungated_mean']:.2%}")
    print(f"Saved governor to {outdir / 'governor_pretrained.pt'}")


if __name__ == "__main__":
    main()