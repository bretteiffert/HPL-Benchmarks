#!/usr/bin/env bash
# BabelStream (Triton) sweep: bash run.sh <cuda|hip> [args...]
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

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

echo "babelstream - Triton, $backend" >&2
echo "installing the '$backend' environment" >&2
pixi install --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" 1>&2

first=1
for size in "${SIZE_CONFIGS[@]}"; do
  echo "running -s $size -n 1000 $*" >&2
  if [ "$csv" -eq 1 ] && [ "$first" -eq 0 ]; then
    # CSV header once across the sweep.
    FMAD=1 pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" \
      python "$SCRIPT_DIR/main.py" -s "$size" -n 1000 "$@" | tail -n +2
  else
    FMAD=1 pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" \
      python "$SCRIPT_DIR/main.py" -s "$size" -n 1000 "$@"
  fi
  first=0
done
