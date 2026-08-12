#!/usr/bin/env bash
# 10-seed live 5-task stream with the single 390-step-trained governor.
set -euo pipefail
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
GOV="${GOV:-results_seeds/governor_390.pt}"
mkdir -p results_seeds
for s in $SEEDS; do
  echo "=== STREAM SEED $s ==="
  .venv/bin/python -m src.experiment2 \
    --config configs/baseline2.yaml --seed "$s" \
    --governor "$GOV" \
    --outdir "results_seeds/s${s}_stream" 2>&1 | tail -18
done
echo "=== LIVE SEED SWEEP COMPLETE ==="
