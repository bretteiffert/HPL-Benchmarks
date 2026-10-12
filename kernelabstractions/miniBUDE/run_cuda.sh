#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
DECK="${DECK:-}"
NUM_TIMES="${NUM_TIMES:-8}"
PRECISION="${PRECISION:-64}"
POSES="${POSES:-0}"
PPWI="${PPWI:-1}"
WGSIZE="${WGSIZE:-1}"

if [[ -z "$DECK" ]]; then
  echo "Set DECK to a miniBUDE deck directory." >&2
  exit 1
fi

"$JULIA_BIN" --project="$SCRIPT_DIR/env/cuda" -O3 --check-bounds=no \
  "$SCRIPT_DIR/miniBUDE.jl" --backend cuda --deck "$DECK" \
  -n "$NUM_TIMES" -p "$PRECISION" --poses "$POSES" \
  --ppwi "$PPWI" --wgsize "$WGSIZE"
