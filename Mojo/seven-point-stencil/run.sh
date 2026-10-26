#!/usr/bin/env bash
# Runs the Mojo seven-point stencil with the pinned 64x4x1 block shape.
#
#   bash run.sh <cuda|hip> [extra args passed to every run]
#
# Example: bash run.sh cuda --float --csv > results.csv
# Status lines and compiler output go to stderr.
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

run_one() {
  pixi run --manifest-path "$SCRIPT_DIR" mojo run -D "BACKEND=$BACKEND" \
    "$SCRIPT_DIR/main.mojo" "$@"
}

echo "stencil - Mojo, ${BACKEND^^}: -s 1024 --block-size 64,4,1" >&2
run_one -s 1024 --block-size 64,4,1 -n 100 "$@"
