#!/usr/bin/env bash
# miniBUDE (CUDA) sweep: every PPWI (-p all) at work-group size 32, one run.
#
#   bash run.sh [extra args passed to the run]
#
# The deck is read from ../../data/miniBUDE/bm1 next to this folder. Builds
# first (build output on stderr). Status lines go to stderr, so
#   bash run.sh --csv > results.csv
# gives one clean file. No environment variables are needed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../data/miniBUDE/bm1"
WGSIZE=32

echo "minibude (CUDA): building" >&2
make -C "$SCRIPT_DIR" >&2

echo "minibude (CUDA): --deck bm1 -p all -w $WGSIZE $*" >&2
"$SCRIPT_DIR/bude-cuda" --deck "$INPUT_DIR" -p all -w "$WGSIZE" "$@"
