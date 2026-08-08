#!/usr/bin/env bash
set -e

if [[ -n "${SMS:-}" ]]; then
  make SMS="${SMS}"
else
  make
fi

ITERATIONS=${ITERATIONS:-100}
WEIGHT_SPARSITY=${WEIGHT_SPARSITY:-0.5}

declare -a SHAPES=(
  "4096 4096"
  "4096 11008"
  "5120 5120"
  "5120 13824"
  "7168 7168"
  "7168 28672"
)

declare -a ACT_SPARSITIES=(0.3 0.5 0.7)

for shape in "${SHAPES[@]}"; do
  read -r M K <<< "${shape}"
  for ACT_SPARSITY in "${ACT_SPARSITIES[@]}"; do
    echo "========================================"
    echo "Hybrid ablation: M=${M}, K=${K}, act_sparsity=${ACT_SPARSITY}, weight_sparsity=${WEIGHT_SPARSITY}"
    echo "========================================"
    ./test_hybrid_ablation "${M}" "${K}" "${ACT_SPARSITY}" "${WEIGHT_SPARSITY}" "${ITERATIONS}"
  done
done
