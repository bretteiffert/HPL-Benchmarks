#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
NUM_TIMES="${NUM_TIMES:-1000}"
PRECISION="${PRECISION:-64}"

echo "babelstream - KernelAbstractions, CUDA"

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

for size in "${SIZE_CONFIGS[@]}"; do
  "$JULIA_BIN" --project="$SCRIPT_DIR/env/cuda" -O3 --check-bounds=no \
    "$SCRIPT_DIR/babelstream.jl" \
    --backend cuda -s "$size" -n "$NUM_TIMES" -p "$PRECISION"
done
