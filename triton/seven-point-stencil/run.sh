#!/usr/bin/env bash
# Seven-point stencil (Triton) sweep: bash run.sh <cuda|hip> [args...]
# Extra arguments are passed to every run, e.g. bash run.sh cuda --float --csv > results.csv
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() { echo "usage: bash run.sh <cuda|hip> [args...]" >&2; }

if [ $# -lt 1 ]; then
  echo "error: missing backend (valid: cuda, hip)" >&2
  usage
  exit 1
fi
backend="$1"
shift
case "$backend" in
  cuda|hip) ;;
  *) echo "error: unknown backend '$backend' (valid: cuda, hip)" >&2; usage; exit 1 ;;
esac

csv=0
for arg in "$@"; do
  case "$arg" in --csv|--csv-raw) csv=1 ;; esac
done

GRID=1024

BLOCK_CONFIGS=(
  "64,1,1"   "128,1,1"  "256,1,1"  "512,1,1"  "1024,1,1"
  "32,4,1"   "64,4,1"   "128,4,1"  "256,4,1"
  "32,8,1"   "64,8,1"   "128,8,1"
  "16,16,1"  "32,16,1"  "64,16,1"
)

echo "stencil - Triton, $backend" >&2
echo "installing the '$backend' environment" >&2
pixi install --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" 1>&2

first=1
for block in "${BLOCK_CONFIGS[@]}"; do
  echo "running -s $GRID --block-size $block $*" >&2
  if [ "$csv" -eq 1 ] && [ "$first" -eq 0 ]; then
    # CSV header once across the sweep.
    pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" \
      python "$SCRIPT_DIR/main.py" -s "$GRID" --block-size "$block" "$@" | tail -n +2
  else
    pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" \
      python "$SCRIPT_DIR/main.py" -s "$GRID" --block-size "$block" "$@"
  fi
  first=0
done
