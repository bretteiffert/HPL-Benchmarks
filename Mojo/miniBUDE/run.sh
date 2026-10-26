#!/usr/bin/env bash
# Runs the Mojo miniBUDE benchmark on bm1 with pinned PPWI 1 and work-group 256.
#
#   bash run.sh <cuda|hip> [extra args passed to the run]
#
# Example: bash run.sh cuda --csv > results.csv
# Status lines and compiler output go to stderr.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../data/miniBUDE/bm1"

usage() {
  echo "usage: bash run.sh <cuda|hip> [minibude args...]" >&2
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

echo "miniBUDE - Mojo, ${BACKEND^^}: bm1, --float -p 1 -w 256 --warmup 2 -i 8" >&2
pixi run --manifest-path "$SCRIPT_DIR" mojo run -D "BACKEND=$BACKEND" \
  "$SCRIPT_DIR/main.mojo" --deck "$INPUT_DIR" --float -p 1 -w 256 --warmup 2 -i 8 "$@"
