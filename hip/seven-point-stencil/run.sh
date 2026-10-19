#!/usr/bin/env bash
# Seven-point stencil (HIP) sweep: 15 block shapes on a 1024^3 grid.
#
#   bash run.sh [extra args passed to every run]
#
# Builds once first (build output on stderr). In --csv / --csv-raw modes the
# CSV header is printed once for the whole sweep, so
#   bash run.sh --csv > results.csv
# gives one clean file. Status lines go to stderr. No environment variables
# are needed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "stencil (HIP): building" >&2
make -C "$SCRIPT_DIR" >&2

binary="$SCRIPT_DIR/laplacian_kernel_hip"
GRID=1024

csv=0
for arg in "$@"; do
  case "$arg" in
    --csv|--csv-raw) csv=1 ;;
  esac
done

BLOCK_CONFIGS=(
  "64,1,1"   "128,1,1"  "256,1,1"  "512,1,1"  "1024,1,1"
  "32,4,1"   "64,4,1"   "128,4,1"  "256,4,1"
  "32,8,1"   "64,8,1"   "128,8,1"
  "16,16,1"  "32,16,1"  "64,16,1"
)

first=1
for cfg in "${BLOCK_CONFIGS[@]}"; do
  echo "stencil (HIP): -s $GRID --block-size $cfg $*" >&2
  if [ "$csv" -eq 1 ] && [ "$first" -eq 0 ]; then
    "$binary" -s "$GRID" --block-size "$cfg" "$@" | tail -n +2
  else
    "$binary" -s "$GRID" --block-size "$cfg" "$@"
  fi
  first=0
done
