#!/usr/bin/env bash
# Runs the Mojo BabelStream benchmark (compiled by `mojo run` on each run) for
# three array sizes with 1,000 iterations each.
#
#   bash run.sh <cuda|hip> [extra args passed to every run]
#
# Example: bash run.sh cuda --float --csv > results.csv
# Status lines and compiler output go to stderr; in the CSV modes the CSV
# header is printed once across the sweep.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "usage: bash run.sh <cuda|hip> [babelstream args...]" >&2
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

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

first=1
for size in "${SIZE_CONFIGS[@]}"; do
  echo "babelstream - Mojo, ${BACKEND^^}: -s $size -n 1000" >&2
  if [ "$CSV_MODE" -eq 1 ] && [ "$first" -eq 0 ]; then
    run_one -s "$size" -n 1000 "$@" | tail -n +2
  else
    run_one -s "$size" -n 1000 "$@"
  fi
  first=0
done
