#!/usr/bin/env bash
# miniBUDE (CUDA): one pinned PPWI 1, work-group 256 run on bm1.
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
echo "minibude (CUDA): building" >&2
make -C "$SCRIPT_DIR" >&2

echo "minibude (CUDA): --deck bm1 --float -p 1 -w 256 --warmup 2 -i 8 $*" >&2
"$SCRIPT_DIR/bude-cuda" --deck "$INPUT_DIR" --float -p 1 -w 256 --warmup 2 -i 8 "$@"
