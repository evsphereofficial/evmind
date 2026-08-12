#!/usr/bin/env bash
set -euo pipefail
PERMS_TSV="results_orders/permutations.tsv"
GOV="${GOV:-results_seeds/governor_390.pt}"
SEEDS="${SEEDS:-0 1 2}"
mkdir -p results_orders/base results_orders/gov
while IFS=$'\t' read -r perm order; do
  [ "$perm" = "perm" ] && continue
  for s in $SEEDS; do
    echo "=== PERM $perm (${order//,/ -> }) SEED $s ==="
    .venv/bin/python -m src.experiment --config configs/baseline.yaml \
      --seed "$s" --order "$order" \
      --outdir "results_orders/base/p${perm}_s${s}" > /dev/null 2>&1
    .venv/bin/python -m src.experiment2 --config configs/baseline2.yaml \
      --seed "$s" --order "$order" --governor "$GOV" \
      --outdir "results_orders/gov/p${perm}_s${s}" 2>&1 | tail -3
  done
done < "$PERMS_TSV"
echo "=== ORDER SWEEP COMPLETE ==="
