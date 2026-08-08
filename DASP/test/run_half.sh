#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
mkdir -p data

# Expected naming convention for matrix files:
# test/matrices_generated/<rows>x<cols>_d<density>.mtx
# Examples:
#   test/matrices/4096x4096_d30.mtx
#   test/matrices/4096x11008_d50.mtx
#   test/matrices/7168x28672_d70.mtx
#
# This matches the plotting cases:
#   4096x4096, 4096x11008
#   5120x5120, 5120x13824
#   7168x7168, 7168x28672
# with densities 30%, 50%, 70%.

MATRIX_DIR="${1:-test/matrices}"

shapes=(
  "5120x5120"
)

densities=("60" "35")

found_any=0
for shape in "${shapes[@]}"; do
  for density in "${densities[@]}"; do
    matrix_path="${MATRIX_DIR}/${shape}_d${density}.mtx"
    if [[ ! -f "${matrix_path}" ]]; then
      echo "Skipping missing matrix: ${matrix_path}"
      continue
    fi

    found_any=1
    echo
    echo "============================================================"
    echo "Running DASP fp16 on ${shape} at density ${density}%"
    echo "Matrix: ${matrix_path}"
    echo "============================================================"
    ./spmv_half "${matrix_path}"
  done
done

if [[ "${found_any}" -eq 0 ]]; then
  echo "No matching matrices found under ${MATRIX_DIR}"
  echo "Expected files like: ${MATRIX_DIR}/4096x4096_d30.mtx"
  echo "You can create them with: python3 generate_mtx.py"
  exit 1
fi
