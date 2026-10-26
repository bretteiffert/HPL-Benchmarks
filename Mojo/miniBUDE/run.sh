#!/usr/bin/env bash
# Runs the Mojo miniBUDE benchmark (compiled by `mojo run`) on the bm1 deck,
# sweeping every ppwi (1..128) at work-group size 32 (CUDA) or 64 (HIP) in one
# run.
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
  cuda) WGSIZE=32 ;;
  hip) WGSIZE=64 ;;
  *)
    echo "error: unknown backend '$BACKEND' (expected cuda or hip)" >&2
    usage
    exit 1
    ;;
esac

echo "miniBUDE - Mojo, ${BACKEND^^}: bm1, -p all -w $WGSIZE" >&2
pixi run --manifest-path "$SCRIPT_DIR" mojo run -D "BACKEND=$BACKEND" \
  "$SCRIPT_DIR/main.mojo" --deck "$INPUT_DIR" -p all -w "$WGSIZE" "$@"
