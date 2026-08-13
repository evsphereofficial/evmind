#!/usr/bin/env bash
set -euo pipefail
MODES="${MODES:-grad random shuffled magnitude}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
for mode in $MODES; do
  GOV="results_hmem/${mode}_governor/governor_pretrained.pt"
  [ -f "$GOV" ] || { echo "skip $mode: $GOV missing"; continue; }
  for s in $SEEDS; do
    echo "=== LIVE hmem=$mode seed $s ==="
    .venv/bin/python -m src.experiment2 --config configs/baseline2.yaml \
      --seed "$s" --hmem-mode "$mode" --governor "$GOV" \
      --outdir "results_hmem/live_${mode}_s${s}" 2>&1 | tail -3
  done
done
echo "=== HMEM LIVE SWEEP COMPLETE ==="
