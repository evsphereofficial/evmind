#!/usr/bin/env bash
set -euo pipefail
MODES="${MODES:-grad magnitude}"
while IFS=$'\t' read -r perm order; do
  [ "$perm" = "perm" ] && continue
  for mode in $MODES; do
    GOV="results_hmem/${mode}_governor/governor_pretrained.pt"
    [ -f "$GOV" ] || continue
    for s in 0 1 2; do
      echo "=== ORDER $perm hmem=$mode seed $s ==="
      .venv/bin/python -m src.experiment2 --config configs/baseline2.yaml \
        --seed "$s" --order "$order" --hmem-mode "$mode" --governor "$GOV" \
        --outdir "results_orders/hmem_${mode}/p${perm}_s${s}" 2>&1 | tail -3
    done
  done
done < results_orders/permutations.tsv
echo "=== HMEM ORDER SWEEP COMPLETE ==="
