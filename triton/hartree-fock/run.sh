#!/usr/bin/env bash
# Hartree-Fock (Triton) sweep: bash run.sh <cuda|hip> [args...]
# Extra arguments are passed to every run, e.g. bash run.sh cuda --float --csv > results.csv
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../data/hartree-fock"

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
  "he16"
  "he32"
  "he64"
  "he128"
  "he256"
)

echo "hartree-fock - Triton, $backend" >&2
echo "installing the '$backend' environment" >&2
pixi install --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" 1>&2

first=1
for size in "${SIZE_CONFIGS[@]}"; do
  input="$INPUT_DIR/$size"
  echo "running $size -n 10 $*" >&2
  if [ "$csv" -eq 1 ] && [ "$first" -eq 0 ]; then
    # CSV header once across the sweep.
    pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" \
      python "$SCRIPT_DIR/main.py" "$input" -n 10 "$@" | tail -n +2
  else
    pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" \
      python "$SCRIPT_DIR/main.py" "$input" -n 10 "$@"
  fi
  first=0
done
