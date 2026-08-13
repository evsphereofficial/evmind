#!/usr/bin/env bash
set -euo pipefail
for mode in grad random shuffled magnitude; do
  echo "=== HMEM META $mode ==="
  .venv/bin/python -m src.meta_pretrain --config configs/baseline2.yaml \
    --hmem-mode "$mode" --outdir "results_hmem/${mode}_governor" 2>&1 | tail -4
done
echo "=== HMEM META COMPLETE ==="
