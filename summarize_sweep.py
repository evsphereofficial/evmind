"""Collapse the 10-seed sweep into mean+/-std vs the plain baseline."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results_seeds")
BASELINE = Path("results/task_accuracies.csv")
SEEDS = [s for s in range(10)]

rows = []
for s in SEEDS:
    stream = ROOT / f"s{s}_stream"
    info = json.loads((stream / "run_config.json").read_text())
    rows.append({
        "seed": s,
        "avg_forgetting": info["average_forgetting"],
        "avg_forgetting_overwritten": info["average_forgetting_overwritten"],
        "final_avg_accuracy": info["final_average_accuracy"],
    })
df = pd.DataFrame(rows)
stats = pd.DataFrame({
    "metric": ["avg_forgetting", "avg_forgetting_overwritten", "final_avg_accuracy"],
    "mean": [df.avg_forgetting.mean(), df.avg_forgetting_overwritten.mean(), df.final_avg_accuracy.mean()],
    "std": [df.avg_forgetting.std(ddof=1), df.avg_forgetting_overwritten.std(ddof=1), df.final_avg_accuracy.std(ddof=1)],
    "min": [df.avg_forgetting.min(), df.avg_forgetting_overwritten.min(), df.final_avg_accuracy.min()],
    "max": [df.avg_forgetting.max(), df.avg_forgetting_overwritten.max(), df.final_avg_accuracy.max()],
})
df.to_csv(ROOT / "seed_summary.csv", index=False)
stats.to_csv(ROOT / "seed_stats.csv", index=False)

bf = pd.read_csv(BASELINE, index_col=0)
base_mat = bf.to_numpy(dtype=float)
from src.metrics import compute_forgetting
base_metric = compute_forgetting(base_mat)

print("per-seed (gated stream):")
for _, r in df.iterrows():
    print(f"  seed {int(r.seed):2d}: forgetting={r.avg_forgetting:6.2f}%  "
          f"overwritten={r.avg_forgetting_overwritten:6.2f}%  "
          f"final_acc={r.final_avg_accuracy:6.2f}%")
print("\nmean +/- std (10 seeds):")
for _, r in stats.iterrows():
    print(f"  {r.metric:26s} {r.mean:7.2f} +/- {r.std:5.2f}   (min {r.min:6.2f}, max {r.max:6.2f})")
print("\nvs plain baseline (seed 0):")
print(f"  avg forgetting         : {base_metric['average_forgetting']:6.2f}%  -> {df.avg_forgetting.mean():6.2f} +/- {df.avg_forgetting.std(ddof=1):5.2f}%")
print(f"  avg forgetting (1..N-1): {np.nanmean(base_metric['forgetting'][:-1]):6.2f}%  -> {df.avg_forgetting_overwritten.mean():6.2f} +/- {df.avg_forgetting_overwritten.std(ddof=1):5.2f}%")
print(f"  final avg accuracy     : {base_metric['final_average_accuracy']:6.2f}%  -> {df.final_avg_accuracy.mean():6.2f} +/- {df.final_avg_accuracy.std(ddof=1):5.2f}%")
