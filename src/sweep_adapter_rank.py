"""Adapter capacity sweep: min rank vs answer-token length.

For each answer length, grow rank on the grid until the learn-time probe
passes -> produces the "cost of one fact" law for the instant path,
analogous to Equation V4 for the dense path.

Usage: .venv/bin/python src/sweep_adapter_rank.py
Output: research lines on stdout + results_adapter_capacity/sweep_adapter_rank.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.live_interactive_lfm2 import (
    DEVICE,
    make_probes,
    make_qa_pairs,
    get_answer,
    load_model_and_tok,
    probe_gated,
    tokenize_qa,
)
from src.instant_expert import RANK_GRID, AdapterExpert, train_instant

OUT = Path("results_adapter_capacity/sweep_adapter_rank.json")

# (kind, value, source_text) — facts at increasing answer-token lengths are
# what the capacity law is fit on; skill/knowledge included as sanity points
CASES = [
    ("fact_name", "Zara", ""),
    ("fact_code", "ZEBRA-42", "My office code is ZEBRA-42"),
    ("fact_name", "Bartholomew", ""),
    ("fact_interest", "vintage cameras and film photography", ""),
    ("fact_note", "the meeting room is on floor 3 near the east wing", ""),
    ("fact_note", "deploy key lives in the vault under rotation every 90 days", ""),
    ("fact_note", "the quarterly review deck is due the second friday of each month", ""),
    ("fact_note", "warehouse bay 12 is reserved for cold-chain shipments until further notice", ""),
    ("skill_code", "reverse a string in Python", ""),
    ("knowledge", "quantum entanglement", ""),
]


def min_rank_for(model, tok, kind, value, source, epochs=30, lr=1e-2) -> dict:
    answer = get_answer(kind, value)
    ans_tok = len(tok(answer, add_special_tokens=False)["input_ids"])
    qa = make_qa_pairs(kind, value, source)
    probes = make_probes(kind, value, source)
    data = tokenize_qa(tok, qa)
    threshold = 1.0 if kind.startswith(("fact_", "skill_")) else 0.34
    # base-model baseline: if it already answers, the row is vacuous
    base_acc = probe_gated(
        model, tok, probes, answer, kind, value, None, adapter=None,
        threshold=threshold,
    )

    row = {"kind": kind, "value": value, "ans_tok": ans_tok,
           "base_acc": round(base_acc, 3), "attempts": []}
    if base_acc >= threshold:
        row["min_rank"] = "base-knows"
        print(f"  {kind}:{value[:28]:28} ans_tok={ans_tok:2} BASE ALREADY PASSES "
              f"({base_acc:.0%}) — vacuous", flush=True)
        return row
    for rank in RANK_GRID:
        ad = AdapterExpert(model.config.hidden_size, rank).to(DEVICE)
        t0 = time.time()
        loss = train_instant(model, ad, data, epochs=epochs, lr=lr)
        train_s = time.time() - t0
        t0 = time.time()
        acc = probe_gated(
            model, tok, probes, answer, kind, value, None, adapter=ad,
            threshold=threshold,
        )
        probe_s = time.time() - t0
        hit = acc >= threshold
        row["attempts"].append(
            {"rank": rank, "loss": round(loss, 5),
             "acc": round(acc, 3), "hit": hit,
             "train_s": round(train_s, 2), "probe_s": round(probe_s, 2)}
        )
        print(
            f"  {kind}:{value[:28]:28} ans_tok={ans_tok:2} rank={rank:4} "
            f"loss={loss:.4f} probe={acc:.0%} train={train_s:.2f}s "
            f"{'HIT' if hit else ''}",
            flush=True,
        )
        if hit:
            row["min_rank"] = rank
            row["train_s"] = round(train_s, 2)
            break
    else:
        row["min_rank"] = None
    return row


def main():
    model, tok, *_ = load_model_and_tok()
    rows = []
    t_start = time.time()
    for kind, value, src in CASES:
        print(f"\n== {kind}:{value} ==", flush=True)
        rows.append(min_rank_for(model, tok, kind, value, src))
    wall = time.time() - t_start

    ok = [r for r in rows
          if isinstance(r.get("min_rank"), int) and r["kind"].startswith("fact_")]
    print("\n=== capacity law ===")
    print(f"{'kind':12} {'ans_tok':>7} {'min_rank':>8} {'params':>9}")
    for r in rows:
        mr = r.get("min_rank")
        params = 2 * 2048 * mr if isinstance(mr, int) else 0
        print(
            f"{r['kind']:12} {r['ans_tok']:7} {str(mr):>8} {params:9}"
        )
    # fit: rank ~ a * ans_tok + b over successes
    if len(ok) >= 2:
        xs = [r["ans_tok"] for r in ok]
        ys = [r["min_rank"] for r in ok]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        varx = sum((x - mx) ** 2 for x in xs)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        a = cov / varx if varx else 0.0
        b = my - a * mx
        ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys) or 1.0
        r2 = 1 - ss_res / ss_tot
        print(f"\nfit: rank_min = {a:.1f} * ans_tok + {b:.1f}   R^2={r2:.3f}")
        out = {"fit": {"a": a, "b": b, "r2": r2}, "wall_s": round(wall, 1),
               "rows": rows}
    else:
        out = {"fit": None, "wall_s": round(wall, 1), "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"saved -> {OUT}  (wall {wall:.0f}s)")


if __name__ == "__main__":
    main()
