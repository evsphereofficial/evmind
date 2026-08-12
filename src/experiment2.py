"""EvMind Phase 2 — HRM-governed continual learning stream.

Identical protocol to Phase 1 (same model arch, seeds, tasks, epochs,
evaluation sets), the ONLY difference: after loss.backward() the FROZEN HRM
intent governor gates the gradients (W' = W - lr * M * grad) before the
optimizer step.

Compare outputs here (results_phase2/) against Phase 1 (results/) to measure
whether the learned intent network improves retention.

Run:
    .venv/bin/python -m src.meta_pretrain --config configs/baseline2.yaml
    .venv/bin/python -m src.experiment2 --config configs/baseline2.yaml
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .dataset import generate_dataset
from .evaluate import evaluate
from .experiment import make_optimizer, plot_accuracy_matrix, set_seed
from .hrm import (
    build_module_groups, DirectGateGovernor, HRMController,
    HRMIntentGovernor, mask_stats,
    measure_rel_change, measure_update_fraction,
)
from .metrics import compute_forgetting
from .model import TinyNumericTransformer
from .tasks import build_tasks

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_GOVERNOR = str(PROJECT_ROOT / "results_governor" / "governor_pretrained.pt")
DEFAULT_BASELINE_CSV = str(PROJECT_ROOT / "results" / "task_accuracies.csv")


def old_task_grads(
    model: torch.nn.Module,
    groups,
    datasets: dict,
    loss_fn: torch.nn.Module,
    device: torch.device,
    phase: int,
) -> list[torch.Tensor] | None:
    """Accumulated gradients of ALL previous tasks' losses on the current
    params = per-weight impact signal sent to the frozen governor (gA in
    meta-training terms). One fixed 512-sample batch per prior task."""
    if phase == 0:
        return None
    model.zero_grad(set_to_none=True)
    for i in range(phase):
        cached = datasets.get((i, "test_grad"))
        if cached is None:
            task_ds = datasets[(i, "test")]
            gen = torch.Generator().manual_seed(10_000 + i)
            idx = torch.randperm(len(task_ds), generator=gen)[:512]
            xs = torch.stack([task_ds[j][0] for j in idx]).to(device)
            ys = torch.stack([task_ds[j][1] for j in idx]).to(device)
            datasets[(i, "test_grad")] = (xs, ys)
        else:
            xs, ys = cached
        loss = loss_fn(model(xs), ys)
        loss.backward()  # accumulate into .grad
    g_old = [g.param.grad.detach().clone() for g in groups]
    model.zero_grad(set_to_none=True)
    return g_old


def train_one_epoch_gated(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    controller: HRMController,
    mask_rows: list[dict],
    phase: int,
    g_old_list: list[torch.Tensor] | None,
    snapshot: dict[str, torch.Tensor],
) -> tuple[float, float, float]:
    """One epoch with governor-gated gradient updates (§17 control_update)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device).float(), y.to(device).float()

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # read-only gates from the governor (sees raw gB in .grad)
        masks = controller.compute_masks(
            model, x, y, loss, g_old_list=g_old_list, snapshot=snapshot)
        pre = {g.name: g.param.detach().clone()
               for g in controller.groups}
        optimizer.step()
        # apply gates to the real Adam update: W = pre + M o (W_adam - pre)
        # (gating raw grads is undone by Adam's per-weight normalization)
        controller.scale_update(model, masks, pre)

        if step == 0 or step == len(loader) - 1:
            stats = mask_stats(masks)
            mask_rows.append({"phase": phase, "batch": step, **stats})

        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += x.size(0)

    elapsed = time.perf_counter() - start
    return total_loss / total, 100.0 * correct / total, elapsed


def plot_retention_snapshot(
    baseline_mat: np.ndarray,
    phase2_mat: np.ndarray,
    task_names: list[str],
    out_path: Path,
) -> None:
    """Mean accuracy of all previously learned tasks per phase (both runs)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mean_previous(mat: np.ndarray) -> list[float]:
        vals = []
        for j in range(mat.shape[1]):
            diag = np.array([mat[i, j] for i in range(j + 1)
                             if not np.isnan(mat[i, j])])
            vals.append(np.nanmean(diag))
        return vals

    phases = list(range(1, len(task_names) + 1))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(phases, mean_previous(baseline_mat), marker="o", label="Phase 1 (plain)")
    ax.plot(phases, mean_previous(phase2_mat), marker="o", label="Phase 2 (HRM-governed)")
    ax.set_xlabel("Training phase")
    ax.set_ylabel("Mean accuracy of previously learned tasks (%)")
    ax.set_title("Retention snapshot: HRM governor vs plain training")
    ax.set_ylim(0, 105)
    ax.set_xticks(phases)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="EvMind Phase 2 HRM-governed stream")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline2.yaml"))
    parser.add_argument("--governor", default=DEFAULT_GOVERNOR)
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results_phase2"))
    parser.add_argument("--seed", type=int, default=None,
                        help="override the experiment seed (multi-seed verification)")
    parser.add_argument("--mode", choices=["mlp", "direct"], default="mlp",
                        help="mlp = shared gate network (phase 2); "
                             "direct = frozen one-gate-per-weight (phase 2b)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.train.seed = args.seed
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        if f.is_file():
            f.unlink()

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- same tasks/datasets as Phase 1 (identical eval sets) ---------------
    tasks = build_tasks(cfg.tasks)
    task_names = [t.name for t in tasks]
    num_tasks = len(tasks)

    datasets = {}
    for i, task in enumerate(tasks):
        datasets[(i, "train")] = generate_dataset(
            task, cfg.train_samples, cfg.train.seed, eval_split=False)
        datasets[(i, "test")] = generate_dataset(
            task, cfg.test_samples, cfg.train.seed, eval_split=True)

    # --- base model (same arch + same init seed as Phase 1) + governor ------
    model = TinyNumericTransformer(
        input_dim=cfg.model.input_dim,
        seq_len=cfg.model.seq_len,
        embedding_dim=cfg.model.embedding_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    num_params = model.count_parameters()

    groups = build_module_groups(model)
    if args.mode == "direct":
        governor = DirectGateGovernor(
            groups, init_mask=cfg.governor.init_mask).to(device)
    else:
        governor = HRMIntentGovernor(
            num_groups=len(groups),
            granularity=cfg.governor.granularity,
            hidden_dim=cfg.governor.hidden_dim,
            refine_steps=cfg.governor.refine_steps,
            init_mask=cfg.governor.init_mask,
        ).to(device)
    governor.load_state_dict(torch.load(args.governor, map_location=device))
    governor.eval()  # FROZEN governance core (§132.3): no grads, no updates
    controller = HRMController(governor, groups, device)

    optimizer = make_optimizer(model, cfg, cfg.train.learning_rate, cfg.train.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("=" * 60)
    print("PHASE 2 — HRM-GOVERNED LIVE LEARNING")
    print("=" * 60)
    print(f"Base model parameters: {num_params:,}  (identical to Phase 1)")
    print(f"Governor (frozen intent net): {governor.governor_params():,} params "
          f"controlling {sum(g.size for g in groups):,} weights "
          f"({getattr(governor, 'granularity', args.mode)}-level, "
          f"{len(groups)} modules)")
    print(f"Governor file: {args.governor}\n")

    # --- continual stream (same protocol as Phase 1) -------------------------
    accuracy_matrix = [[float("nan")] * num_tasks for _ in range(num_tasks)]
    log_rows: list[dict] = []
    mask_rows: list[dict] = []
    train_times: dict[int, float] = {}
    update_fractions: dict[int, float] = {}
    rel_changes: dict[int, float] = {}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    for phase, task in enumerate(tasks):
        task_name = task.name
        train_loader = torch.utils.data.DataLoader(
            datasets[(phase, "train")],
            batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.train.num_workers)

        snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
        g_old_list = old_task_grads(model, groups, datasets, loss_fn, device, phase)
        train_start = time.perf_counter()

        for epoch in range(cfg.train.epochs_per_task):
            loss, acc, secs = train_one_epoch_gated(
                model, train_loader, optimizer, loss_fn, device,
                controller, mask_rows, phase + 1, g_old_list, snapshot)
            log_rows.append({
                "task": task_name, "phase": phase + 1, "epoch": epoch + 1,
                "train_loss": round(loss, 5),
                "train_accuracy": round(acc, 3),
                "epoch_seconds": round(secs, 3),
            })
            print(f"  [Task {phase+1}/{num_tasks} {task_name}] epoch {epoch+1}/{cfg.train.epochs_per_task}"
                  f" loss={loss:.4f} acc={acc:.2f}%")
        train_times[phase] = time.perf_counter() - train_start
        update_fractions[phase] = measure_update_fraction(model, snapshot)
        rel_changes[phase] = measure_rel_change(model, snapshot)

        print(f"  Evaluating task(s): {', '.join(task_names[: phase + 1])}")
        for i in range(phase + 1):
            test_loader = torch.utils.data.DataLoader(
                datasets[(i, "test")], batch_size=cfg.test_samples, shuffle=False)
            acc, _, latency_ms = evaluate(model, test_loader, device, loss_fn)
            accuracy_matrix[i][phase] = round(acc, 2)
            print(f"    task {i+1:<3} ({task_names[i]:<12}) accuracy: {acc:6.2f}%")
        print()

    # --- metrics + outputs ----------------------------------------------------
    mat = np.array(accuracy_matrix, dtype=float)
    metric = compute_forgetting(mat)

    pd.DataFrame(mat, index=task_names,
                 columns=[f"after_t{i+1}" for i in range(num_tasks)]
                 ).to_csv(outdir / "task_accuracies.csv")
    pd.DataFrame({
        "task": task_names,
        "initial_accuracy": np.round(metric["initial"], 4),
        "best_accuracy": np.round(metric["best"], 4),
        "final_accuracy": np.round(metric["final"], 4),
        "forgetting": np.round(metric["forgetting"], 4),
    }).to_csv(outdir / "forgetting.csv", index=False)
    pd.DataFrame(log_rows).to_csv(outdir / "training_log.csv", index=False)
    pd.DataFrame(mask_rows).to_csv(outdir / "governor_masks.csv", index=False)
    pd.DataFrame({
        "task": task_names,
        "update_fraction_pct": [round(update_fractions[i], 3) for i in range(num_tasks)],
        "mean_rel_change": [round(rel_changes[i], 5) for i in range(num_tasks)],
    }).to_csv(outdir / "update_fraction.csv", index=False)

    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    peak_ram_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)

    run_info = {
        "run_name": cfg.run_name,
        "config": cfg.to_dict(),
        "device": device.type,
        "parameter_count": num_params,
        "governor_params": governor.governor_params(),
        "num_groups": len(groups),
        "granularity": getattr(governor, "granularity", args.mode),
        "training_time_seconds": {f"task{i+1}": round(t, 3) for i, t in train_times.items()},
        "total_training_seconds": round(sum(train_times.values()), 3),
        "inference_latency_ms_per_sample": latency_ms,
        "peak_vram_mb": peak_vram_mb,
        "peak_ram_mb": peak_ram_mb,
        "final_average_accuracy": metric["final_average_accuracy"],
        "average_forgetting": metric["average_forgetting"],
        "average_forgetting_overwritten": float(np.nanmean(metric["forgetting"][:-1])),
        "update_fractions": {f"task{i+1}": update_fractions[i] for i in range(num_tasks)},
        "mean_rel_changes": {f"task{i+1}": rel_changes[i] for i in range(num_tasks)},
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(run_info, f, indent=2)

    plot_accuracy_matrix(accuracy_matrix, task_names, outdir / "accuracy_matrix.png")
    torch.save(model.state_dict(), outdir / "final_model.pt")

    # --- comparison with Phase 1 ----------------------------------------------
    baseline_mat = None
    if Path(DEFAULT_BASELINE_CSV).exists():
        bf = pd.read_csv(DEFAULT_BASELINE_CSV, index_col=0)
        baseline_mat = bf.to_numpy(dtype=float)
        base_metric = compute_forgetting(baseline_mat)
        plot_retention_snapshot(baseline_mat, mat, task_names,
                                outdir / "comparison_retention.png")
        pd.DataFrame({
            "task": task_names,
            "forgetting_phase1": np.round(base_metric["forgetting"], 4),
            "forgetting_phase2": np.round(metric["forgetting"], 4),
            "delta": np.round(metric["forgetting"] - base_metric["forgetting"], 4),
        }).to_csv(outdir / "comparison_with_baseline.csv", index=False)

    # --- summary ----------------------------------------------------------------
    print("=" * 60)
    print("PHASE 2 — HRM-GOVERNED RESULTS")
    print(f"Parameters: {num_params:,} (+ governor {governor.governor_params():,} frozen)")
    print("=" * 60)
    for i, name in enumerate(task_names):
        print(f"\nTask {i+1} ({name}):")
        print(f"  Initial accuracy: {metric['initial'][i]:.2f}%")
        print(f"  Final accuracy:   {metric['final'][i]:.2f}%")
        print(f"  Forgetting:       {metric['forgetting'][i]:.2f}%")
        print(f"  Update fraction:  {update_fractions[i]:.2f}%   "
              f"mean rel |dW|: {rel_changes[i]:.4f}")
    print("\n" + "=" * 60)
    print(f"Average Forgetting: {metric['average_forgetting']:.2f}%")
    print(f"Average Forgetting (overwritten tasks 1..{num_tasks - 1}): "
          f"{np.nanmean(metric['forgetting'][:-1]):.2f}%")
    print(f"Final Average Accuracy: {metric['final_average_accuracy']:.2f}%")

    if baseline_mat is not None:
        print("\n--- COMPARISON vs PHASE 1 (plain training) ---")
        print(f"  Avg forgetting         : Phase1 {base_metric['average_forgetting']:.2f}%"
              f"  ->  Phase2 {metric['average_forgetting']:.2f}%")
        print(f"  Avg forgetting (1..N-1): Phase1 {np.nanmean(base_metric['forgetting'][:-1]):.2f}%"
              f"  ->  Phase2 {np.nanmean(metric['forgetting'][:-1]):.2f}%")
        print(f"  Final avg accuracy     : Phase1 {base_metric['final_average_accuracy']:.2f}%"
              f"  ->  Phase2 {metric['final_average_accuracy']:.2f}%")
        pd.DataFrame(np.round(mat, 2), index=task_names,
                     columns=[f"after_t{i+1}" for i in range(num_tasks)]
                     ).to_csv(outdir / "phase2_matrix.csv")

    print(f"Total training time: {run_info['total_training_seconds']:.1f}s")
    if peak_vram_mb:
        print(f"Peak VRAM: {peak_vram_mb} MB")
    print(f"Peak RAM: {peak_ram_mb} MB")
    print("=" * 60)
    print(f"Outputs written to {outdir}/")


if __name__ == "__main__":
    main()