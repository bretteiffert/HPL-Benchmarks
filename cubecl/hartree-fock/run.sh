#!/usr/bin/env bash
# Runs the CubeCL Hartree-Fock benchmark across the configured inputs.
#
# Usage: ./run.sh <cuda|hip> [extra benchmark args...]
#   e.g. ./run.sh cuda --csv > results.csv
#        ./run.sh hip --float
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../data/hartree-fock"

backend="${1:-}"
case "$backend" in
  cuda|hip) shift ;;
  *)
    echo "usage: $0 <cuda|hip> [benchmark args...]" >&2
    exit 1
    ;;
esac

SIZE_CONFIGS=(
  "he16"
  "he32"
  "he64"
  "he128"
  "he256"
)

# Cargo reads .cargo/config.toml from the working directory, not from
# --manifest-path. Use GNU host compilers for native Rust dependencies rather
# than any compiler injected by the NVIDIA HPC SDK.
cd "$SCRIPT_DIR"
export CC=gcc CXX=g++

# Status lines go to stderr so `--csv` output can be redirected cleanly.
echo "hartree-fock - CubeCL, ${backend^^}" >&2

# Build once up front so compiler output doesn't interleave with results.
cargo build --release --no-default-features --features "$backend" >&2

csv_header_done=0
for size in "${SIZE_CONFIGS[@]}"; do
  out="$(cargo run --quiet --release --no-default-features --features "$backend" -- \
    "$INPUT_DIR/$size" -n 10 --block-size 256 "$@")"
  # In CSV modes, keep a single header row across inputs.
  if [[ " $* " == *" --csv "* || " $* " == *" --csv-raw "* ]]; then
    if (( csv_header_done )); then
      printf '%s\n' "$out" | tail -n +2
    else
      printf '%s\n' "$out"
      csv_header_done=1
    fi
  else
    printf '%s\n\n' "$out"
  fi
done
