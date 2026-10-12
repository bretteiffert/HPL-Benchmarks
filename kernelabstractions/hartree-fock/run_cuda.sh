#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
INPUT="${INPUT:-}"
NUM_TIMES="${NUM_TIMES:-10}"
PRECISION="${PRECISION:-64}"
if [[ -z "$INPUT" ]]; then
  echo "Set INPUT to a Hartree-Fock input file." >&2
  exit 1
fi
"$JULIA_BIN" --project="$SCRIPT_DIR/env/cuda" -O3 --check-bounds=no \
  "$SCRIPT_DIR/hartree-fock.jl" "$INPUT" \
  --backend cuda -n "$NUM_TIMES" -p "$PRECISION"
