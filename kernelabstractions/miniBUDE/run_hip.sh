#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
DECK="${DECK:-$SCRIPT_DIR/../../data/miniBUDE/bm1}"
NUM_TIMES="${NUM_TIMES:-8}"
PRECISION="${PRECISION:-64}"
POSES="${POSES:-0}"
WGSIZE="${WGSIZE:-32}"

echo "miniBUDE - KernelAbstractions, HIP" >&2

PPWI_CONFIGS=(
  "1"
  "2"
  "4"
  "8"
  "16"
  "32"
  "64"
  "128"
)

if [[ -n "${PPWI:-}" ]]; then
  PPWI_CONFIGS=("$PPWI")
fi

if [[ ! -d "$DECK" ]]; then
  echo "Deck directory not found: $DECK" >&2
  exit 1
fi

for ppwi in "${PPWI_CONFIGS[@]}"; do
  "$JULIA_BIN" --project="$SCRIPT_DIR/env/hip" -O3 --check-bounds=no \
    "$SCRIPT_DIR/miniBUDE.jl" --backend hip --deck "$DECK" \
    -n "$NUM_TIMES" -p "$PRECISION" --poses "$POSES" \
    --ppwi "$ppwi" --wgsize "$WGSIZE" --csv
done
