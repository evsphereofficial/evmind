#!/usr/bin/env bash
set -euo pipefail
for mode in random shuffled magnitude; do
  GOV="results_hmem/${mode}_governor/governor_pretrained.pt"
  if [ -f "$GOV" ]; then
    echo "=== HMEM META $mode already done, skipping ==="
    continue
  fi
  echo "=== HMEM META $mode ==="
  .venv/bin/python -m src.meta_pretrain --config configs/baseline2.yaml \
    --hmem-mode "$mode" --outdir "results_hmem/${mode}_governor" 2>&1 | tail -4
done
echo "=== HMEM META COMPLETE ==="
