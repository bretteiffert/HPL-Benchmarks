#!/usr/bin/env bash
# Runs the Mojo seven-point stencil (compiled by `mojo run` on each run) on a
# 1024^3 grid for 15 block shapes, 100 timed iterations each.
#
#   bash run.sh <cuda|hip> [extra args passed to every run]
#
# Example: bash run.sh cuda --float --csv > results.csv
# Status lines and compiler output go to stderr; in the CSV modes the CSV
# header is printed once across the sweep.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "usage: bash run.sh <cuda|hip> [stencil args...]" >&2
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi
BACKEND="$1"
shift
case "$BACKEND" in
  cuda|hip) ;;
  *)
    echo "error: unknown backend '$BACKEND' (expected cuda or hip)" >&2
    usage
    exit 1
    ;;
esac

CSV_MODE=0
for arg in "$@"; do
  case "$arg" in
    --csv|--csv-raw) CSV_MODE=1 ;;
  esac
done

run_one() {
  pixi run --manifest-path "$SCRIPT_DIR" mojo run -D "BACKEND=$BACKEND" \
    "$SCRIPT_DIR/main.mojo" "$@"
}

GRID=1024

BLOCK_CONFIGS=(
  "64,1,1"   "128,1,1"  "256,1,1"  "512,1,1"  "1024,1,1"
  "32,4,1"   "64,4,1"   "128,4,1"  "256,4,1"
  "32,8,1"   "64,8,1"   "128,8,1"
  "16,16,1"  "32,16,1"  "64,16,1"
)

first=1
for blk in "${BLOCK_CONFIGS[@]}"; do
  echo "stencil - Mojo, ${BACKEND^^}: -s $GRID --block-size $blk" >&2
  if [ "$CSV_MODE" -eq 1 ] && [ "$first" -eq 0 ]; then
    run_one -s "$GRID" --block-size "$blk" -n 100 "$@" | tail -n +2
  else
    run_one -s "$GRID" --block-size "$blk" -n 100 "$@"
  fi
  first=0
done
