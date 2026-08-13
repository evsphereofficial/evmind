"""h_mem (input-driven influence) experiment summary.

Ablations (per spec):
  A = no influence channel          (existing governor_390 run, results_seeds)
  B = HRM(x, raw influence)        (hmem_mode=grad)
  C = HRM(x, random map)           (hmem_mode=random)  -- control
  D = HRM(x, shuffled influence)   (hmem_mode=shuffled) -- control
  E = influence magnitude only     (hmem_mode=magnitude)

Reports:
  1. meta-level: retention gain + new-task cost from meta_evaluate
  2. live level: 5 tasks x 10 seeds, avg/overwritten forgetting + final
     accuracy (mean+/-std), paired vs baseline and vs A
  3. channel diagnostics: hmem stats + corr(hmem, gate), corr(hmem, |dW|)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import compute_forgetting

ROOT = Path("results_hmem")
SEEDS = list(range(10))


def live_metrics(d: Path) -> dict:
    mat = pd.read_csv(d / "task_accuracies.csv", index_col=0).to_numpy(dtype=float)
    m = compute_forgetting(mat)
    return {
        "avg_forgetting": m["average_forgetting"],
        "overwritten_forgetting": float(np.nanmean(m["forgetting"][:-1])),
        "final_accuracy": m["final_average_accuracy"],
    }


def meta_block(mode: str) -> dict:
    info = json.loads((ROOT / f"{mode}_governor" / "pretrain_info.json").read_text())
    ev = info.get("meta_eval", {})
    return {
        "retention_gain_pp": 100.0 * ev.get("old_task_retention_gain", float("nan")),
        "new_task_cost_pp": 100.0 * ev.get("new_task_cost", float("nan")),
        "acc_a_after_gated": 100.0 * ev.get("acc_a_gated_mean", float("nan")),
        "acc_b_gated": 100.0 * ev.get("acc_b_gated_mean", float("nan")),
    }


print("=== 1) META-LEVEL (24 paired evals; FIRST SUCCESS CRITERION) ===")
print(f"  {'mode':<10} {'ret_gain_pp':>12} {'newtask_cost':>12} "
      f"{'old_acc_gated':>14} {'new_acc_gated':>14}")
meta_rows = []
for mode in ["none", "grad", "random", "shuffled", "magnitude"]:
    if not (ROOT / f"{mode}_governor" / "pretrain_info.json").exists():
        continue
    mb = meta_block(mode)
    meta_rows.append({"mode": mode, **mb})
    print(f"  {mode:<10} {mb['retention_gain_pp']:>+12.2f} "
          f"{mb['new_task_cost_pp']:>+12.2f} "
          f"{mb['acc_a_after_gated']:>14.2f} {mb['acc_b_gated']:>14.2f}")
pd.DataFrame(meta_rows).to_csv(ROOT / "hmem_meta_summary.csv", index=False)

print("\n=== 2) LIVE 5-TASK x 10-SEED ===")
rows = []
for mode in ["none", "grad", "random", "shuffled", "magnitude"]:
    for s in SEEDS:
        d = ROOT / f"live_{mode}_s{s}"
        if not (d / "task_accuracies.csv").exists():
            if mode == "none":
                d = Path("results_seeds") / f"s{s}_stream"
            else:
                continue
        r = live_metrics(d)
        rows.append({"hmem": mode, "seed": s, **r})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "hmem_live_runs.csv", index=False)

print(f"  {'hmem':<10} {'avg_forget':>12} {'overwritten':>12} {'final_acc':>10}")
agg = []
for mode in df.hmem.unique():
    g = df[df.hmem == mode]
    agg.append({
        "hmem": mode,
        "avg_forgetting_m": g.avg_forgetting.mean(),
        "avg_forgetting_s": g.avg_forgetting.std(ddof=1),
        "overwritten_m": g.overwritten_forgetting.mean(),
        "overwritten_s": g.overwritten_forgetting.std(ddof=1),
        "final_acc_m": g.final_accuracy.mean(),
        "final_acc_s": g.final_accuracy.std(ddof=1),
    })
    print(f"  {mode:<10} {g.avg_forgetting.mean():>9.2f} "
          f"+/- {g.avg_forgetting.std(ddof=1):<5.2f} "
          f"{g.overwritten_forgetting.mean():>9.2f} "
          f"+/- {g.overwritten_forgetting.std(ddof=1):<5.2f} "
          f"{g.final_accuracy.mean():>8.2f} +/- {g.final_accuracy.std(ddof=1):<5.2f}")
pd.DataFrame(agg).to_csv(ROOT / "hmem_live_summary.csv", index=False)

if len(df[df.hmem == "grad"]) and len(df[df.hmem == "none"]):
    g = df[df.hmem == "grad"].set_index("seed")
    a = df[df.hmem == "none"].set_index("seed")
    print("\n  paired grad vs none (gov wins / 10):")
    for col in ["avg_forgetting", "overwritten_forgetting", "final_accuracy"]:
        d = g[col] - a[col]
        wins = int((d < 0).sum()) if col != "final_accuracy" else int((d > 0).sum())
        print(f"    {col:24s} diff {d.mean():+7.2f} +/- {d.std(ddof=1):5.2f} ({wins}/10)")

print("\n=== 3) CHANNEL DIAGNOSTICS (live masks CSV, last logged batch/phase) ===")
for mode in ["grad", "random", "shuffled", "magnitude"]:
    d = ROOT / f"live_{mode}_s0"
    if not (d / "governor_masks.csv").exists():
        continue
    gm = pd.read_csv(d / "governor_masks.csv")
    tail = gm[gm.phase == gm.phase.max()]
    print(f"  {mode:<10} hmem mean {tail.hmem_mean.mean():.3f} "
          f"std {tail.hmem_std.mean():.3f} max {tail.hmem_max.mean():.1f} | "
          f"corr(hmem,gate) {tail.corr_hmem_mask.mean():+.3f} | "
          f"corr(hmem,|dW|) {tail.corr_hmem_dw.mean():+.3f}")