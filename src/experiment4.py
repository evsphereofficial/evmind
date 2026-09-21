"""EvMind Registry — HRM-governed live stream with Task Registry.

Identical protocol to Phase 2 (same model, seeds, tasks, epochs, eval):
the ONLY difference is the old-knowledge representation: instead of
SensitivityMemory (EWC-style accumulated importance), we use TaskRegistry
which stores explicit per-parameter footprints and region-level marks.

After each task finishes, the registry captures:
  - per-parameter footprint: |grad| * |weight| at task completion
  - region-level marks: mean importance per fixed region

During subsequent tasks, the governor receives these as 2 additional
input features so it can protect old-task territory.

Run:
    .venv/bin/python -m src.meta_pretrain_registry --config configs/registry.yaml
    .venv/bin/python -m src.experiment4 --config configs/registry.yaml
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
from .experiment import (
    make_optimizer, plot_accuracy_matrix, plot_retention_curves, set_seed,
)
from .hrm import (
    build_module_groups, compute_influence,
    HRMController, HRMIntentGovernor, mask_stats,
    measure_rel_change, measure_update_fraction,
)
from .metrics import compute_forgetting
from .model import TinyNumericTransformer
from .registry import TaskRegistry
from .tasks import build_tasks

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_GOVERNOR = str(PROJECT_ROOT / "results_registry" / "governor_pretrained.pt")
DEFAULT_BASELINE_CSV = str(PROJECT_ROOT / "results" / "task_accuracies.csv")


def capture_sensitivity(
    model: torch.nn.Module,
    groups,
    datasets: dict,
    loss_fn: torch.nn.Module,
    device: torch.device,
    task_idx: int,
) -> list[torch.Tensor]:
    """Per-weight sensitivity of ONE old task on the CURRENT params."""
    cached = datasets.get((task_idx, "test_grad"))
    if cached is None:
        task_ds = datasets[(task_idx, "test")]
        gen = torch.Generator().manual_seed(10_000 + task_idx)
        idx = torch.randperm(len(task_ds), generator=gen)[:512]
        xs = torch.stack([task_ds[j][0] for j in idx]).to(device)
        ys = torch.stack([task_ds[j][1] for j in idx]).to(device)
        datasets[(task_idx, "test_grad")] = (xs, ys)
    else:
        xs, ys = cached
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(xs), ys)
    loss.backward()
    g_t = [g.param.grad.detach().clone() for g in groups]
    model.zero_grad(set_to_none=True)
    return g_t


def train_one_epoch_registry(
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
    registry: TaskRegistry,
    close_threshold: float = 0.02,
) -> tuple[float, float, float, list[torch.Tensor]]:
    """One epoch with governor-gated gradient updates using registry features.

    Returns (loss, accuracy, elapsed, last_gradients) where last_gradients
    is the gradient list from the final batch (for registry capture).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()
    last_grads = None

    # Get registry protection signals
    param_prot, region_prot = registry.get_protection()
    registry_regions = registry.regions if registry.n > 0 else None
    registry_group_sizes = [g.size for g in controller.groups] if registry.n > 0 else None

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device).float(), y.to(device).float()

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Shadow tracker: online occupancy (non-trainable, detached)
        cur_grads = [g.param.grad for g in controller.groups]
        if registry.use_shadow:
            registry.accumulate(cur_grads)

        # Save last batch gradients for registry capture (fallback)
        last_grads = [g.detach().clone() for g in cur_grads]

        # Governor gates with registry features
        masks = list(controller.compute_masks(
            model, x, y, loss, g_old_list=g_old_list, snapshot=snapshot,
            registry_param_protection=param_prot,
            registry_region_protection=region_prot,
            registry_regions=registry_regions,
            registry_group_sizes=registry_group_sizes))
        # Hard gate for occupied weights (shared variable): never overwrite labelled weights
        if registry.n > 0 and hasattr(registry, "_occupied"):
            occ = registry.get_occupied()
            if occ.any():
                # map global occupied mask to per-group masks, force gate=0
                offset = 0
                for gi, g in enumerate(controller.groups):
                    sz = g.size
                    g_occ = occ[offset:offset+sz].reshape(g.param.shape)
                    if g_occ.any():
                        mf = masks[gi].reshape(g.param.shape)
                        mf = torch.where(g_occ, torch.zeros_like(mf), mf)
                        masks[gi] = mf.flatten()
                    offset += sz
        pre = {g.name: g.param.detach().clone() for g in controller.groups}
        optimizer.step()
        controller.scale_update(model, masks, pre)
        if close_threshold > 0.0:
            controller.zero_closed_moments(optimizer, masks, threshold=close_threshold)
        # Enforce hard closed for occupied in Adam moments as well
        if registry.n > 0 and hasattr(registry, "_occupied"):
            occ = registry.get_occupied()
            if occ.any():
                offset = 0
                for gi, g in enumerate(controller.groups):
                    sz = g.size
                    g_occ = occ[offset:offset+sz].reshape(g.param.shape)
                    if g_occ.any():
                        st = optimizer.state.get(g.param)
                        if st is not None:
                            for key in ("exp_avg", "exp_avg_sq"):
                                buf = st.get(key)
                                if buf is not None:
                                    buf.masked_fill_(g_occ, 0.0)
                    offset += sz
                # offset already handled, but keep for completeness
                pass

        if step == 0 or step == len(loader) - 1:
            stats = mask_stats(masks)
            row = {"phase": phase, "batch": step, **stats}
            mask_rows.append(row)

        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += x.size(0)

    elapsed = time.perf_counter() - start
    return total_loss / total, 100.0 * correct / total, elapsed, last_grads


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EvMind Registry — HRM-governed stream with Task Registry")
    parser.add_argument("--config",
                        default=str(PROJECT_ROOT / "configs" / "registry.yaml"))
    parser.add_argument("--governor", default=DEFAULT_GOVERNOR)
    parser.add_argument("--outdir",
                        default=str(PROJECT_ROOT / "results_registry"))
    parser.add_argument("--seed", type=int, default=None,
                        help="override the experiment seed")
    parser.add_argument("--order", type=str, default=None,
                        help="comma-separated task order permutation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.order is not None:
        wanted = [n.strip() for n in args.order.split(",")]
        by_name = {t.name: t for t in cfg.tasks}
        if set(wanted) != set(by_name) or len(wanted) != len(by_name):
            raise SystemExit(
                f"--order must be a permutation of {list(by_name)}, got {wanted}")
        cfg.tasks = [by_name[n] for n in wanted]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gov_path = Path(args.governor).resolve()
    for f in outdir.iterdir():
        # don't delete the governor input when outdir == governor dir
        if f.is_file() and f.resolve() != gov_path:
            f.unlink()

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- tasks + datasets ---
    tasks = build_tasks(cfg.tasks)
    task_names = [t.name for t in tasks]
    num_tasks = len(tasks)

    datasets = {}
    for i, task in enumerate(tasks):
        datasets[(i, "train")] = generate_dataset(
            task, cfg.train_samples, cfg.train.seed, eval_split=False)
        datasets[(i, "test")] = generate_dataset(
            task, cfg.test_samples, cfg.train.seed, eval_split=True)

    # --- base model + governor ---
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
    reg_cfg = getattr(cfg, "registry", None)
    registry_region_size = getattr(reg_cfg, "region_size", 1000) if reg_cfg else 1000
    registry_footprint = getattr(reg_cfg, "footprint_method", "grad_x_weight") if reg_cfg else "grad_x_weight"
    registry_normalize = getattr(reg_cfg, "normalize", True) if reg_cfg else True

    hmem_mode = getattr(cfg, "hmem_mode", "none")
    governor = HRMIntentGovernor(
        num_groups=len(groups),
        granularity=cfg.governor.granularity,
        hidden_dim=cfg.governor.hidden_dim,
        refine_steps=cfg.governor.refine_steps,
        init_mask=cfg.governor.init_mask,
        per_weight_feat_dim=8 + (1 if hmem_mode != "none" else 0),
        registry_feat_dim=2,  # param_protection + region_ownership
    ).to(device)
    governor.load_state_dict(
        torch.load(args.governor, map_location=device), strict=False)
    governor.eval()
    controller = HRMController(governor, groups, device)

    # --- Task Registry (replaces SensitivityMemory) ---
    reg_use_shadow = bool(getattr(reg_cfg, "use_shadow", False))
    reg_shadow_alpha = float(getattr(reg_cfg, "shadow_alpha", 0.0))
    reg_learnable = bool(getattr(reg_cfg, "learnable", False))
    reg_hid = int(getattr(reg_cfg, "recognizer_hidden", 16))
    registry = TaskRegistry(
        groups,
        region_size=registry_region_size,
        footprint_method=registry_footprint,
        normalize=registry_normalize,
        use_shadow=reg_use_shadow,
        shadow_alpha=reg_shadow_alpha,
        learnable=reg_learnable,
        recognizer_hidden=reg_hid,
    )
    if reg_learnable and registry.recognizer is not None:
        # load frozen recognizer trained jointly with governor
        registry.recognizer.to(device)
        reg_path = Path(args.governor).parent / "registry_recognizer.pt"
        alt_path = Path(args.outdir) / "registry_recognizer.pt"
        for p in [reg_path, alt_path]:
            if p.exists():
                registry.recognizer.load_state_dict(torch.load(p, map_location=device))
                registry.recognizer.to(device).eval()
                print(f"Loaded registry recognizer: {p} ({sum(p.numel() for p in registry.recognizer.parameters()):,} params)")
                break

    optimizer = make_optimizer(model, cfg, cfg.train.learning_rate, cfg.train.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    close_threshold = float(getattr(cfg.governor, "close_threshold", 0.02))

    print("=" * 60)
    print("REGISTRY — HRM-GOVERNED LIVE LEARNING")
    print("=" * 60)
    print(f"Base model parameters: {num_params:,}")
    print(f"Governor (frozen): {governor.governor_params():,} params")
    rec_info = f", recognizer={sum(p.numel() for p in registry.recognizer.parameters()):,}" if registry.recognizer else ""
    print(f"Registry: footprint={registry_footprint}, "
          f"region_size={registry_region_size}, "
          f"normalize={registry_normalize}, shadow={reg_use_shadow}, learnable={reg_learnable}{rec_info}")
    print(f"Registry regions: {registry.num_regions}")

    # --- continual stream ---
    accuracy_matrix = [[float("nan")] * num_tasks for _ in range(num_tasks)]
    log_rows: list[dict] = []
    mask_rows: list[dict] = []
    train_times: dict[int, float] = {}
    update_fractions: dict[int, float] = {}
    rel_changes: dict[int, float] = {}
    registry_snapshots: list[dict] = []

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
        if registry.use_shadow:
            registry.begin_task(device=device)
        train_start = time.perf_counter()

        for epoch in range(cfg.train.epochs_per_task):
            loss, acc, secs, last_grads = train_one_epoch_registry(
                model, train_loader, optimizer, loss_fn, device,
                controller, mask_rows, phase + 1, None, snapshot,
                registry, close_threshold=close_threshold)
            log_rows.append({
                "task": task_name, "phase": phase + 1, "epoch": epoch + 1,
                "train_loss": round(loss, 5),
                "train_accuracy": round(acc, 3),
                "epoch_seconds": round(secs, 3),
            })
            print(f"  [Task {phase+1}/{num_tasks} {task_name}] "
                  f"epoch {epoch+1}/{cfg.train.epochs_per_task} "
                  f"loss={loss:.4f} acc={acc:.2f}%")
        train_times[phase] = time.perf_counter() - train_start
        update_fractions[phase] = measure_update_fraction(model, snapshot)
        rel_changes[phase] = measure_rel_change(model, snapshot)

        # Capture this task's footprint into the registry (modular law allocation inside)
        registry.capture(model, gradients=last_grads, task_name=task_name, target_acc=98.0, headroom=0.2)
        summary = registry.summary()
        alloc = registry.allocation_summary()
        registry_snapshots.append({
            "task": task_name, "phase": phase + 1,
            "n_regions": registry.num_regions,
            "footprint_mean": summary.get("footprint_mean", 0),
            "footprint_max": summary.get("footprint_max", 0),
            "k_alloc": alloc["per_task"].get(task_name, {}).get("k_pred", 0),
            "occupied": f"{alloc['occupied_count']}/{alloc['total']}",
        })
        k_pred = alloc["per_task"].get(task_name, {}).get("k_pred", 0)
        print(f"  registry: {registry.n} task(s), "
              f"k_pred={k_pred} via law, occupied {alloc['occupied_count']}/{alloc['total']}, "
              f"footprint_mean={summary.get('footprint_mean', 0):.4f}")

        # Evaluate all learned tasks
        print(f"  Evaluating task(s): {', '.join(task_names[: phase + 1])}")
        for i in range(phase + 1):
            test_loader = torch.utils.data.DataLoader(
                datasets[(i, "test")], batch_size=cfg.test_samples, shuffle=False)
            acc, _, latency_ms = evaluate(model, test_loader, device, loss_fn)
            accuracy_matrix[i][phase] = round(acc, 2)
            print(f"    task {i+1:<3} ({task_names[i]:<12}) accuracy: {acc:6.2f}%")
        print()

    # --- metrics + outputs ---
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
        "update_fraction_pct": [round(update_fractions[i], 3)
                                for i in range(num_tasks)],
        "mean_rel_change": [round(rel_changes[i], 5)
                            for i in range(num_tasks)],
    }).to_csv(outdir / "update_fraction.csv", index=False)
    pd.DataFrame(registry_snapshots).to_csv(
        outdir / "registry_snapshots.csv", index=False)

    # Region ownership breakdown + allocation log
    region_ownership = registry.region_weights_per_task()
    pd.DataFrame(region_ownership).to_csv(
        outdir / "region_ownership.csv", index=False)
    # Save occupied mask and ownership for shared variable
    alloc_sum = registry.allocation_summary()
    occ = registry.get_occupied().cpu().numpy()
    own = registry.get_ownership().cpu().numpy()
    np.save(outdir / "occupied.npy", occ)
    np.save(outdir / "ownership.npy", own)
    pd.DataFrame([{"task": k, "k_pred": v["k_pred"], "task_id": v["task_id"]} for k, v in alloc_sum["per_task"].items()]).to_csv(outdir / "allocation.csv", index=False)

    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    peak_ram_mb = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)

    run_info = {
        "run_name": cfg.run_name,
        "config": cfg.to_dict(),
        "device": device.type,
        "parameter_count": num_params,
        "governor_params": governor.governor_params(),
        "num_groups": len(groups),
        "granularity": cfg.governor.granularity,
        "registry": {
            "footprint_method": registry_footprint,
            "region_size": registry_region_size,
            "normalize": registry_normalize,
            "num_regions": registry.num_regions,
            "tasks_captured": registry.n,
            "allocation": registry.allocation_summary(),
        },
        "training_time_seconds": {
            f"task{i+1}": round(t, 3) for i, t in train_times.items()},
        "total_training_seconds": round(sum(train_times.values()), 3),
        "peak_vram_mb": peak_vram_mb,
        "peak_ram_mb": peak_ram_mb,
        "final_average_accuracy": metric["final_average_accuracy"],
        "average_forgetting": metric["average_forgetting"],
        "average_forgetting_overwritten": float(
            np.nanmean(metric["forgetting"][:-1])),
        "update_fractions": {
            f"task{i+1}": update_fractions[i] for i in range(num_tasks)},
        "mean_rel_changes": {
            f"task{i+1}": rel_changes[i] for i in range(num_tasks)},
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(run_info, f, indent=2)

    plot_accuracy_matrix(accuracy_matrix, task_names,
                         outdir / "accuracy_matrix.png")
    plot_retention_curves(accuracy_matrix, task_names,
                          outdir / "forgetting_curve.png")
    torch.save(model.state_dict(), outdir / "final_model.pt")
    torch.save(registry, outdir / "registry.pt")

    # --- comparison with Phase 1 ---
    baseline_mat = None
    if Path(DEFAULT_BASELINE_CSV).exists():
        bf = pd.read_csv(DEFAULT_BASELINE_CSV, index_col=0)
        baseline_mat = bf.to_numpy(dtype=float)
        base_metric = compute_forgetting(baseline_mat)
        pd.DataFrame({
            "task": task_names,
            "forgetting_phase1": np.round(base_metric["forgetting"], 4),
            "forgetting_registry": np.round(metric["forgetting"], 4),
            "delta": np.round(
                metric["forgetting"] - base_metric["forgetting"], 4),
        }).to_csv(outdir / "comparison_with_baseline.csv", index=False)

    # --- summary ---
    print("=" * 60)
    print("REGISTRY — HRM-GOVERNED RESULTS")
    print(f"Parameters: {num_params:,} "
          f"(+ governor {governor.governor_params():,} frozen)")
    print(f"Registry: {registry.n} tasks captured, "
          f"{registry.num_regions} regions")
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
    print(f"Average Forgetting (overwritten): "
          f"{np.nanmean(metric['forgetting'][:-1]):.2f}%")
    print(f"Final Average Accuracy: "
          f"{metric['final_average_accuracy']:.2f}%")

    if baseline_mat is not None:
        print("\n--- COMPARISON vs PHASE 1 (plain training) ---")
        print(f"  Avg forgetting: "
              f"Phase1 {base_metric['average_forgetting']:.2f}%"
              f"  ->  Registry {metric['average_forgetting']:.2f}%")
        print(f"  Final avg accuracy: "
              f"Phase1 {base_metric['final_average_accuracy']:.2f}%"
              f"  ->  Registry {metric['final_average_accuracy']:.2f}%")

    # Region ownership summary
    print("\n--- REGION OWNERSHIP ---")
    for t, ownership in enumerate(region_ownership):
        print(f"  Task {t+1} ({task_names[t]}): {ownership}")

    print(f"\nTotal training time: "
          f"{run_info['total_training_seconds']:.1f}s")
    if peak_vram_mb:
        print(f"Peak VRAM: {peak_vram_mb} MB")
    print(f"Peak RAM: {peak_ram_mb} MB")
    print("=" * 60)
    print(f"Outputs written to {outdir}/")


if __name__ == "__main__":
    main()
