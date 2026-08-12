"""Collapse the fixed 5-task x 10-seed sweep into paired stats.

Per seed the governor stream (results_seeds/sN_stream) and the plain
baseline (results_seeds/base_sN) are matched 1:1 (same data seed,
same model init seed, same task order). Reports per-seed values plus
mean/std/median/min/max for both methods and the paired difference.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import compute_forgetting

ROOT = Path("results_seeds")
SEEDS = list(range(10))


def run_metrics(stream_dir: Path) -> dict:
    mat = pd.read_csv(stream_dir / "task_accuracies.csv", index_col=0).to_numpy(dtype=float)
    m = compute_forgetting(mat)
    return {
        "avg_forgetting": m["average_forgetting"],
        "avg_forgetting_overwritten": float(np.nanmean(m["forgetting"][:-1])),
        "final_avg_accuracy": m["final_average_accuracy"],
    }


rows = []
for s in SEEDS:
    gov = run_metrics(ROOT / f"s{s}_stream")
    base = run_metrics(ROOT / f"base_s{s}")
    rows.append({
        "seed": s,
        "gov_forgetting": gov["avg_forgetting"],
        "base_forgetting": base["avg_forgetting"],
        "diff_forgetting": gov["avg_forgetting"] - base["avg_forgetting"],
        "gov_forgetting_overwritten": gov["avg_forgetting_overwritten"],
        "base_forgetting_overwritten": base["avg_forgetting_overwritten"],
        "diff_forgetting_overwritten": gov["avg_forgetting_overwritten"] - base["avg_forgetting_overwritten"],
        "gov_final_accuracy": gov["final_avg_accuracy"],
        "base_final_accuracy": base["final_avg_accuracy"],
        "diff_final_accuracy": gov["final_avg_accuracy"] - base["final_avg_accuracy"],
    })
df = pd.DataFrame(rows)
df.to_csv(ROOT / "paired_fixed_order.csv", index=False)

pairs = [
    ("avg_forgetting", "base_forgetting", "gov_forgetting", "diff_forgetting"),
    ("avg_forgetting_overwritten", "base_forgetting_overwritten",
     "gov_forgetting_overwritten", "diff_forgetting_overwritten"),
    ("final_avg_accuracy", "base_final_accuracy", "gov_final_accuracy",
     "diff_final_accuracy"),
]
out = []
for label, bcol, gcol, dcol in pairs:
    b, g, d = df[bcol], df[gcol], df[dcol]
    out.append({
        "metric": label,
        "baseline_mean": b.mean(), "baseline_std": b.std(ddof=1),
        "baseline_median": b.median(), "baseline_min": b.min(), "baseline_max": b.max(),
        "governor_mean": g.mean(), "governor_std": g.std(ddof=1),
        "governor_median": g.median(), "governor_min": g.min(), "governor_max": g.max(),
        "paired_diff_mean": d.mean(), "paired_diff_std": d.std(ddof=1),
        "paired_diff_min": d.min(), "paired_diff_max": d.max(),
        "governor_wins": int((d < 0).sum()) if label != "final_avg_accuracy" else int((d > 0).sum()),
    })
stats = pd.DataFrame(out)
stats.to_csv(ROOT / "paired_fixed_order_stats.csv", index=False)

print("per-seed (paired, n=10):")
print(f"  {'seed':>4} {'base_forget':>12} {'gov_forget':>12} {'delta':>8} "
      f"{'base_final':>11} {'gov_final':>11} {'delta':>8}")
for _, r in df.iterrows():
    print(f"  {int(r.seed):>4} {r.base_forgetting:>12.2f} {r.gov_forgetting:>12.2f} "
          f"{r.diff_forgetting:>+8.2f} {r.base_final_accuracy:>11.2f} "
          f"{r.gov_final_accuracy:>11.2f} {r.diff_final_accuracy:>+8.2f}")

print("\nsummary (n=10 seeds):")
for _, r in stats.iterrows():
    print(f"  {r.metric:26s}  baseline {r.baseline_mean:7.2f} +/- {r.baseline_std:5.2f} "
          f"| governor {r.governor_mean:7.2f} +/- {r.governor_std:5.2f} "
          f"| paired diff {r.paired_diff_mean:+7.2f} +/- {r.paired_diff_std:5.2f} "
          f"(gov wins {int(r.governor_wins)}/10)")