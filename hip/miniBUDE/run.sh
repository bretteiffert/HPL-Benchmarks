#!/usr/bin/env bash
# miniBUDE (HIP) sweep: every PPWI (-p all) at work-group size 64, one run.
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
WGSIZE=64

echo "minibude (HIP): building" >&2
make -C "$SCRIPT_DIR" >&2

echo "minibude (HIP): --deck bm1 -p all -w $WGSIZE $*" >&2
"$SCRIPT_DIR/bude-hip" --deck "$INPUT_DIR" -p all -w "$WGSIZE" "$@"
