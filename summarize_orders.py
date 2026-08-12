"""Task-order permutation experiment summary.

For each of the 10 fixed random orders x 3 seeds (paired baseline vs
governor), load the accuracy matrix and report per-task-order means plus
overall aggregate stats. 5 tasks x 10 orders x 3 seeds = 30 runs per method.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import compute_forgetting

ROOT = Path("results_orders")
ORDERS = pd.read_csv(ROOT / "permutations.tsv", sep="\t")
N_PERM, N_SEED = 10, 3


def metrics_for(stream_dir: Path) -> dict:
    mat = pd.read_csv(stream_dir / "task_accuracies.csv", index_col=0).to_numpy(dtype=float)
    m = compute_forgetting(mat)
    return {
        "avg_forgetting": m["average_forgetting"],
        "overwritten_forgetting": float(np.nanmean(m["forgetting"][:-1])),
        "final_accuracy": m["final_average_accuracy"],
    }


rows = []
for _, r in ORDERS.iterrows():
    p = int(r.perm)
    order = r.order
    for s in range(N_SEED):
        base = metrics_for(ROOT / "base" / f"p{p}_s{s}")
        gov = metrics_for(ROOT / "gov" / f"p{p}_s{s}")
        rows.append({
            "perm": p, "order": order, "seed": s,
            "base_forgetting": base["avg_forgetting"],
            "gov_forgetting": gov["avg_forgetting"],
            "diff_forgetting": gov["avg_forgetting"] - base["avg_forgetting"],
            "base_overwritten": base["overwritten_forgetting"],
            "gov_overwritten": gov["overwritten_forgetting"],
            "diff_overwritten": gov["overwritten_forgetting"] - base["overwritten_forgetting"],
            "base_final": base["final_accuracy"],
            "gov_final": gov["final_accuracy"],
            "diff_final": gov["final_accuracy"] - base["final_accuracy"],
        })
df = pd.DataFrame(rows)
df.to_csv(ROOT / "order_runs.csv", index=False)

per_perm = df.groupby(["perm", "order"])[["base_forgetting", "gov_forgetting",
                                          "base_overwritten", "gov_overwritten",
                                          "base_final", "gov_final"]].agg(["mean", "std"])
per_perm.to_csv(ROOT / "order_means.csv")

print("per-task-order (3 seeds each):")
print(f"  {'perm':>4} {'order':<76} {'base_forget':>12} {'gov_forget':>12} {'base_final':>11} {'gov_final':>11}")
for (p, order), g in df.groupby(["perm", "order"]):
    print(f"  {int(p):>4} {order:<76} "
          f"{g.base_forgetting.mean():>12.2f} {g.gov_forgetting.mean():>12.2f} "
          f"{g.base_final.mean():>11.2f} {g.gov_final.mean():>11.2f}")

print("\noverall (30 runs per method):")
for label, bcol, gcol, dcol in [
    ("avg_forgetting", "base_forgetting", "gov_forgetting", "diff_forgetting"),
    ("overwritten_forgetting", "base_overwritten", "gov_overwritten", "diff_overwritten"),
    ("final_accuracy", "base_final", "gov_final", "diff_final"),
]:
    b, g, d = df[bcol], df[gcol], df[dcol]
    print(f"  {label:22s} baseline {b.mean():7.2f} +/- {b.std(ddof=1):5.2f} | "
          f"governor {g.mean():7.2f} +/- {g.std(ddof=1):5.2f} | "
          f"paired diff {d.mean():+7.2f} +/- {d.std(ddof=1):5.2f} "
          f"(gov wins {int((d < 0).sum()) if label != 'final_accuracy' else int((d > 0).sum())}/30)")