"""EvMind Phase 1 baseline experiment.

Single node, single parameter space, single continual training stream.
Run with:

    .venv/bin/python -m src.experiment --config configs/baseline.yaml
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
from .metrics import compute_forgetting
from .model import TinyNumericTransformer
from .tasks import build_tasks
from .train import train_one_epoch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def set_seed(seed: int) -> None:
    """Determinism for torch/cuda/python RNGs (spec: one seed first)."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(model: torch.nn.Module, cfg, lr: float, wd: float):
    name = cfg.train.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer: {name}")


def plot_accuracy_matrix(
    matrix: list[list[float]], task_names: list[str], out_path: Path
) -> None:
    """Heatmap: rows = tasks, cols = evaluation points (after each phase)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels([f"After T{i+1}" for i in range(len(task_names))])
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names)
    ax.set_xlabel("Training phase")
    ax.set_title("Task Accuracy Matrix (baseline: sequential live training)")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center",
                        color="white" if arr[i, j] < 65 else "black")
    fig.colorbar(im, label="accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_retention_curves(
    matrix: list[list[float]], task_names: list[str], out_path: Path
) -> None:
    """Task retention vs training step (line plot)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.array(matrix, dtype=float)
    phases = np.arange(1, len(task_names) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(task_names):
        row = arr[i, : i + 1]  # task i measured from phase i onward
        ax.plot(phases[: len(row)], row, marker="o", label=name)
    ax.set_xlabel("Training phase (step)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Task Retention vs Training Step")
    ax.set_ylim(0, 105)
    ax.set_xticks(phases)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="EvMind Phase 1 baseline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--seed", type=int, default=None,
                        help="override the experiment seed (multi-seed verification)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.train.seed = args.seed
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Fresh run: clear previous artifacts so results/ mirrors the latest run.
    for f in outdir.iterdir():
        if f.is_file():
            f.unlink()

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    # --- Tasks & datasets -------------------------------------------------
    tasks = build_tasks(cfg.tasks)
    task_names = [t.name for t in tasks]
    num_tasks = len(tasks)

    datasets = {}
    for i, task in enumerate(tasks):
        datasets[(i, "train")] = generate_dataset(
            task, cfg.train_samples, cfg.train.seed, eval_split=False
        )
        datasets[(i, "test")] = generate_dataset(
            task, cfg.test_samples, cfg.train.seed, eval_split=True
        )

    # --- Model & optimizer ------------------------------------------------
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

    optimizer = make_optimizer(model, cfg, cfg.train.learning_rate, cfg.train.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("=" * 60)
    print("SINGLE NODE CONTINUAL LEARNING TEST")
    print("=" * 60)
    print(f"Model parameters: {num_params:,}")
    print(f"Tasks: {num_tasks}  Train samples/task: {cfg.train_samples:,}  "
          f"Test samples/task: {cfg.test_samples:,}")
    print(f"Optimizer: {cfg.train.optimizer} lr={cfg.train.learning_rate} "
          f"wd={cfg.train.weight_decay} batch={cfg.train.batch_size} "
          f"epochs/task={cfg.train.epochs_per_task}\n")

    # --- Continual loop ----------------------------------------------------
    # accuracy_matrix[i][j] = accuracy of task i after training phase j (NaN if not yet learned)
    accuracy_matrix = [[float("nan")] * num_tasks for _ in range(num_tasks)]
    log_rows: list[dict] = []
    train_times: dict[int, float] = {}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    for phase, task in enumerate(tasks):
        task_name = task.name
        loader_kwargs = dict(batch_size=cfg.train.batch_size, shuffle=True,
                             num_workers=cfg.train.num_workers)
        train_loader = torch.utils.data.DataLoader(datasets[(phase, "train")], **loader_kwargs)

        # ---- train on the current task (same weights, never reset) ----
        train_start = time.perf_counter()
        for epoch in range(cfg.train.epochs_per_task):
            loss, acc, secs = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            log_rows.append({
                "task": task_name,
                "phase": phase + 1,
                "epoch": epoch + 1,
                "train_loss": round(loss, 5),
                "train_accuracy": round(acc, 3),
                "epoch_seconds": round(secs, 3),
            })
            print(f"  [Task {phase+1}/{num_tasks} {task_name}] epoch {epoch+1}/{cfg.train.epochs_per_task}"
                  f" loss={loss:.4f} acc={acc:.2f}%")
        train_times[phase] = time.perf_counter() - train_start

        # ---- evaluate ALL tasks learned so far -------------------------
        print(f"  Evaluating task(s): {', '.join(task_names[: phase + 1])}")
        for i in range(phase + 1):
            test_loader = torch.utils.data.DataLoader(
                datasets[(i, "test")], batch_size=cfg.test_samples, shuffle=False
            )
            acc, _, latency_ms = evaluate(model, test_loader, device, loss_fn)
            accuracy_matrix[i][phase] = round(acc, 2)
            print(f"    task {i+1:<3} ({task_names[i]:<12}) accuracy: {acc:6.2f}%")
        print()

    # --- Metrics -----------------------------------------------------------
    mat = np.array(accuracy_matrix, dtype=float)
    metric = compute_forgetting(mat)

    # --- Save outputs ------------------------------------------------------
    df_matrix = pd.DataFrame(mat, index=task_names,
                             columns=[f"after_t{i+1}" for i in range(num_tasks)])
    df_matrix.to_csv(outdir / "task_accuracies.csv")

    df_forgetting = pd.DataFrame({
        "task": task_names,
        "initial_accuracy": np.round(metric["initial"], 4),
        "best_accuracy": np.round(metric["best"], 4),
        "final_accuracy": np.round(metric["final"], 4),
        "forgetting": np.round(metric["forgetting"], 4),
    })
    df_forgetting.to_csv(outdir / "forgetting.csv", index=False)

    df_log = pd.DataFrame(log_rows)
    df_log.to_csv(outdir / "training_log.csv", index=False)

    # peak resources
    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    peak_ram_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)

    run_info = {
        "run_name": cfg.run_name,
        "config": cfg.to_dict(),
        "device": device.type,
        "parameter_count": num_params,
        "training_time_seconds": {f"task{i+1}": round(t, 3) for i, t in train_times.items()},
        "total_training_seconds": round(sum(train_times.values()), 3),
        "inference_latency_ms_per_sample": latency_ms,
        "peak_vram_mb": peak_vram_mb,
        "peak_ram_mb": peak_ram_mb,
        "final_average_accuracy": metric["final_average_accuracy"],
        "average_forgetting": metric["average_forgetting"],
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(run_info, f, indent=2)

    plot_accuracy_matrix(accuracy_matrix, task_names, outdir / "accuracy_matrix.png")
    plot_retention_curves(accuracy_matrix, task_names, outdir / "forgetting_curve.png")
    torch.save(model.state_dict(), outdir / "final_model.pt")

    # --- Summary ------------------------------------------------------------
    print("=" * 60)
    print(f"SINGLE NODE CONTINUAL LEARNING TEST")
    print(f"Parameters: {num_params:,}")
    print("=" * 60)
    for i, name in enumerate(task_names):
        print(f"\nTask {i+1} ({name}):")
        print(f"  Initial accuracy: {metric['initial'][i]:.2f}%")
        print(f"  Final accuracy:   {metric['final'][i]:.2f}%")
        print(f"  Forgetting:       {metric['forgetting'][i]:.2f}%")
    print("\n" + "=" * 60)
    print(f"Average Forgetting: {metric['average_forgetting']:.2f}%")
    overwritten = metric["forgetting"][:-1]
    print(f"Average Forgetting (overwritten tasks 1..{num_tasks - 1}): "
          f"{np.nanmean(overwritten):.2f}%")
    print(f"Final Average Accuracy: {metric['final_average_accuracy']:.2f}%")
    print(f"Total training time: {run_info['total_training_seconds']:.1f}s")
    print(f"Inference latency: {latency_ms:.4f} ms/sample")
    if peak_vram_mb:
        print(f"Peak VRAM: {peak_vram_mb} MB")
    print(f"Peak RAM: {peak_ram_mb} MB")
    print("=" * 60)
    print(f"Outputs written to {outdir}/")


if __name__ == "__main__":
    main()