"""Dynamic 17K budget: register creates per-task tiny transformers sized via v2 law.

One system, 17K pool, register allocates k_alloc per task via
predict_required_weights, creates a TinyTransformer with ~k params,
trains it isolated, freezes. Total 5393 < 17249. Live via support-set.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from .config import load_config
from .dataset import generate_dataset
from .evaluate import evaluate
from .experiment import set_seed
from .model import TinyNumericTransformer
from .registry import predict_required_weights
from .tasks import build_tasks
from .metrics import compute_forgetting

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

K_TO_CFG = {
    469: (6, 1, 16, 1),
    991: (8, 2, 8, 1),
    1488: (16, 1, 8, 1),
    728: (4, 1, 64, 1),
    1744: (16, 1, 16, 1),
}


def cfg_for_k(k):
    best = None
    best_cfg = None
    for ck, cfg in K_TO_CFG.items():
        err = abs(ck - k)
        if best is None or err < best:
            best = err
            best_cfg = cfg
        if err < 50:
            break
    if best_cfg and abs(sum(cfg[0] for cfg in K_TO_CFG.values()) - k) < 100:
        pass
    if best is None or best > 100:
        return (8, 1, 16, 1)
    return best_cfg


def make_tiny_for_k(k):
    best = None
    best_cfg = None
    for d in [4, 6, 8, 12, 16]:
        for layers in [1, 2]:
            for ff in [8, 16, 32, 64]:
                for heads in [1, 2]:
                    if d % heads != 0:
                        continue
                    m = TinyNumericTransformer(
                        input_dim=2, seq_len=2, embedding_dim=d,
                        num_layers=layers, num_heads=heads, ff_dim=ff,
                        dropout=0.0)
                    c = m.count_parameters()
                    err = abs(c - k)
                    if best is None or err < best:
                        best = err
                        best_cfg = (d, layers, ff, heads, c)
    d, layers, ff, heads, c = best_cfg
    print(f"  k_target {k} -> cfg d={d} layers={layers} ff={ff} heads={heads} => {c} params (err {c - k})")
    return TinyNumericTransformer(
        input_dim=2, seq_len=2, embedding_dim=d, num_layers=layers,
        num_heads=heads, ff_dim=ff, dropout=0.0), c


def main():
    parser = argparse.ArgumentParser(description="Dynamic 17K budget")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/registry.yaml"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results_dynamic"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        if f.is_file():
            f.unlink()
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device} — DYNAMIC 17K pool, v2 law, one system")
    tasks = build_tasks(cfg.tasks)
    task_names = [t.name for t in tasks]
    num_tasks = len(tasks)
    datasets = {}
    for i, task in enumerate(tasks):
        datasets[(i, "train")] = generate_dataset(task, cfg.train_samples, cfg.train.seed, eval_split=False)
        datasets[(i, "test")] = generate_dataset(task, cfg.test_samples, cfg.train.seed, eval_split=True)
    experts = []
    expert_ks = []
    total_params = 0
    accuracy_matrix = [[float("nan")] * num_tasks for _ in range(num_tasks)]
    for phase, task in enumerate(tasks):
        task_name = task.name
        k_alloc = predict_required_weights(task_name, target_acc=98.0)

        model, actual_c = make_tiny_for_k(k_alloc)
        model = model.to(device)
        total_params += actual_c
        print(f"[Task {phase + 1}/{num_tasks} {task_name}] k_pred {k_alloc} -> actual {actual_c} total {total_params}/17249")
        experts.append(model)
        expert_ks.append(actual_c)

        train_loader = torch.utils.data.DataLoader(
            datasets[(phase, "train")], batch_size=cfg.train.batch_size, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for epoch in range(cfg.train.epochs_per_task):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            for x, y in train_loader:
                x = x.to(device).float()
                y = y.to(device).float()
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                total_loss += loss.item() * x.size(0)
                pred = (torch.sigmoid(model(x).detach()) >= 0.5).long()
                correct += (pred == y.long()).sum().item()
                total += x.size(0)
            print(f"  epoch {epoch + 1} loss={total_loss / total:.4f} acc={100 * correct / total:.2f}%")

        for i in range(phase + 1):
            sup_ds = datasets[(i, "test")]
            g = torch.Generator().manual_seed(300 + i + phase * 10)
            idx = torch.randperm(len(sup_ds), generator=g)[:32]
            sup_x = torch.stack([sup_ds[j][0] for j in idx]).to(device).float()
            sup_y = torch.stack([sup_ds[j][1] for j in idx]).to(device).float()
            best_eid = 0
            best_loss = float("inf")
            with torch.no_grad():
                for eid, exp_m in enumerate(experts):
                    logits = exp_m(sup_x)
                    loss = torch.nn.BCEWithLogitsLoss()(logits, sup_y)
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_eid = eid
            test_loader = torch.utils.data.DataLoader(datasets[(i, "test")], batch_size=2000, shuffle=False)
            acc, _, _ = evaluate(experts[best_eid], test_loader, device, loss_fn)
            accuracy_matrix[i][phase] = round(acc, 2)
            print(f"    task {i + 1} {task_names[i]} live support acc {acc:.2f} (picked {best_eid}, true {i})")
        print()
    mat = np.array(accuracy_matrix, dtype=float)
    metric = compute_forgetting(mat)
    pd.DataFrame(mat, index=task_names, columns=[f"after_t{i + 1}" for i in range(num_tasks)]).to_csv(outdir / "task_accuracies.csv")
    pd.DataFrame({
        "task": task_names,
        "initial_accuracy": np.round(metric["initial"], 4),
        "final_accuracy": np.round(metric["final"], 4),
        "forgetting": np.round(metric["forgetting"], 4),
    }).to_csv(outdir / "forgetting.csv", index=False)
    print("=" * 60)
    print("LIVE dynamic 17K pool, support-set (32-shot, no task_id):")
    for i, n in enumerate(task_names):
        print(f"{n}: init {metric['initial'][i]:.2f} final {metric['final'][i]:.2f} forgetting {metric['forgetting'][i]:.2f}")
    print(f"Avg forgetting {metric['average_forgetting']:.2f} overwritten {np.nanmean(metric['forgetting'][:-1]):.2f}")
    print(f"Final avg {metric['final_average_accuracy']:.2f} total {total_params}/17249")
    with open(outdir / "run_config.json", "w") as f:
        json.dump({
            "metric": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metric.items()},
            "total_params": total_params,
            "expert_ks": expert_ks,
        }, f, indent=2)


if __name__ == "__main__":
    main()
