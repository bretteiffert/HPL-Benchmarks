#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
INPUT_DIR="${INPUT_DIR:-$SCRIPT_DIR/../../data/hartree-fock}"
NUM_TIMES="${NUM_TIMES:-10}"
PRECISION="${PRECISION:-64}"

echo "hartree-fock - KernelAbstractions, HIP" >&2

SIZE_CONFIGS=(
  "he16"
  "he32"
  "he64"
  "he128"
  "he256"
)

if [[ -n "${INPUT:-}" ]]; then
  INPUTS=("$INPUT")
else
  INPUTS=()
  for size in "${SIZE_CONFIGS[@]}"; do
    INPUTS+=("$INPUT_DIR/$size")
  done
fi

for input in "${INPUTS[@]}"; do
  if [[ ! -f "$input" ]]; then
    echo "Input file not found: $input" >&2
    exit 1
  fi

  "$JULIA_BIN" --project="$SCRIPT_DIR/env/hip" -O3 --check-bounds=no \
    "$SCRIPT_DIR/hartree-fock.jl" "$input" \
    --backend hip --csv -n "$NUM_TIMES" -p "$PRECISION"
done
