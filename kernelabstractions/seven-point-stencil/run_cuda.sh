#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
SIZE_CONFIGS=("128" "256" "512" "1024")
NUM_TIMES="${NUM_TIMES:-100}"
PRECISION="${PRECISION:-64}"

echo "seven-point-stencil - KernelAbstractions, CUDA"

for size in "${SIZE_CONFIGS[@]}"; do
  "$JULIA_BIN" --project="$SCRIPT_DIR/env/cuda" -O3 --check-bounds=no \
    "$SCRIPT_DIR/seven-point-stencil.jl" \
    --backend cuda -s "$size" -n "$NUM_TIMES" -p "$PRECISION"
done
