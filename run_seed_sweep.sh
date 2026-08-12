#!/usr/bin/env bash
# Full-pipeline 10-seed sweep: per seed, meta-pretrain the horizon-matched
# governor, then run the gated 5-task live stream with that governor.
set -euo pipefail
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
OUTROOT="${OUTROOT:-results_seeds}"
mkdir -p "$OUTROOT"
for s in $SEEDS; do
  echo "=== SEED $s ==="
  .venv/bin/python -m src.meta_pretrain \
    --config configs/baseline2.yaml --seed "$s" \
    --outdir "$OUTROOT/s${s}_governor" 2>&1 | tail -6
  .venv/bin/python -m src.experiment2 \
    --config configs/baseline2.yaml --seed "$s" \
    --governor "$OUTROOT/s${s}_governor/governor_pretrained.pt" \
    --outdir "$OUTROOT/s${s}_stream" 2>&1 | tail -25
done
echo "=== SWEEP COMPLETE ==="
