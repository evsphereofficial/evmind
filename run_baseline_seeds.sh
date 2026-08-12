#!/usr/bin/env bash
set -euo pipefail
for s in ${SEEDS:-0 1 2 3 4 5 6 7 8 9}; do
  echo "=== BASELINE SEED $s ==="
  .venv/bin/python -m src.experiment --config configs/baseline.yaml \
    --seed "$s" --outdir "results_seeds/base_s${s}" 2>&1 | tail -4
done
echo "=== BASELINE SEED SWEEP COMPLETE ==="
