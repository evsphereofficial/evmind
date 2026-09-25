"""Evaluate the fine-tuned EvAGI System-1 classifier (through laya.Agent,
exactly as the live app will call it).

Run: .venv/bin/python src/eval_laya_evagi.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).parent))

from evagi_system1 import (  # noqa: E402
    LAYA_DIR, build_questions, build_state, load_jsonl,
)

REPO = Path(__file__).parent.parent
DATA_DIR = REPO / "data" / "laya_evagi"


def run_rows(agent, rows, name):
    n = n_action = n_topic = n_topic_ok = 0
    confs = []
    wrong = []
    lat = []
    for r in rows:
        topics = r.get("topics") or []
        qs = build_questions(topics)
        t0 = time.perf_counter()
        res = agent.predict(r["state"], qs)
        lat.append((time.perf_counter() - t0) * 1000)
        ans = res["answers"]
        pred_action = ans["action"]["choice"]
        conf = float(ans["action"].get("answer_confidence") or 0)
        confs.append(conf)
        gold = r["gold"]
        ok_a = pred_action == gold["action"]
        n_action += ok_a
        ok_t = True
        if "topic" in ans:
            ok_t = ans["topic"]["choice"] == gold.get("topic")
            n_topic += 1
            if ok_t:
                n_topic_ok += 1
        n += 1
        if not (ok_a and ok_t):
            wrong.append((gold, pred_action, conf,
                          ans["topic"]["choice"] if "topic" in ans else "-",
                          r["state"]["user_message"],
                          r["state"]["known_memory"]))
    lat.sort()
    print(f"\n=== {name} (n={n}) ===")
    print(f"action acc: {n_action / n:.3f}   topic acc: "
          f"{(n_topic_ok / n_topic if n_topic else 1.0):.3f} ({n_topic_ok}/{n_topic})   "
          f"p50 {lat[len(lat)//2]:.0f}ms p95 {lat[int(len(lat)*.95)]:.0f}ms")
    confs.sort()
    print(f"action conf: min {confs[0]:.2f} p10 {confs[len(confs)//10]:.2f} "
          f"p50 {confs[len(confs)//2]:.2f} max {confs[-1]:.2f}")
    # confidence separation: correct vs wrong
    if wrong:
        print(f"WRONG ({len(wrong)}):")
        for g, pa, c, pt, um, mem in wrong[:20]:
            print(f"  gold={g['action']:<20} pred={pa:<20} conf={c:.2f} "
                  f"topic={pt:<22} | {um[:52]!r} mem={mem[:44]!r}")
    return n_action / n, wrong, confs


def main():
    import laya
    print(f"loading {LAYA_DIR}")
    agent = laya.Agent(LAYA_DIR, device="cuda")

    val = load_jsonl(str(DATA_DIR / "val.jsonl"))
    battery = load_jsonl(str(DATA_DIR / "battery.jsonl"))

    _, wrong_val, confs_val = run_rows(agent, val, "val (held-out)")
    acc_bat, wrong_bat, confs_bat = run_rows(agent, battery, "tricky battery")

    # threshold sweep on combined set (one predict per row)
    rows = val + battery
    results = []
    for r in rows:
        qs = build_questions(r.get("topics") or [])
        res = agent.predict(r["state"], qs)
        a = res["answers"]["action"]
        conf = float(a.get("answer_confidence") or 0)
        ok = a["choice"] == r["gold"]["action"]
        results.append((conf, ok))
    print("\n=== fallback threshold sweep (action only) ===")
    print("(fallback = send turn to System-2 LLM router instead)")
    for thr in (0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8):
        auto = [c for c, _ in results if c >= thr]
        auto_ok = sum(1 for c, ok in results if c >= thr and ok)
        cov = len(auto) / len(results)
        prec = (auto_ok / len(auto)) if auto else 1.0
        print(f"  thr={thr:.2f}  auto-coverage={cov:.3f}  auto-precision={prec:.3f}")


if __name__ == "__main__":
    main()
