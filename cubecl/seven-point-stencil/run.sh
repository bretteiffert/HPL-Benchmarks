#!/usr/bin/env bash
# Runs the CubeCL seven-point stencil across the configured block shapes.
#
# Usage: ./run.sh <cuda|hip> [extra benchmark args...]
#   e.g. ./run.sh cuda --csv > results.csv
#        ./run.sh hip --float
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

backend="${1:-}"
case "$backend" in
  cuda|hip) shift ;;
  *)
    echo "usage: $0 <cuda|hip> [benchmark args...]" >&2
    exit 1
    ;;
esac

GRID=1024

BLOCK_CONFIGS=(
  "64,1,1"   "128,1,1"  "256,1,1"  "512,1,1"  "1024,1,1"
  "32,4,1"   "64,4,1"   "128,4,1"  "256,4,1"
  "32,8,1"   "64,8,1"   "128,8,1"
  "16,16,1"  "32,16,1"  "64,16,1"
)

# Status lines go to stderr so `--csv` output can be redirected cleanly.
echo "Seven-point stencil - CubeCL, ${backend^^}" >&2

# Build once up front so compiler output doesn't interleave with results.
cargo build --release --manifest-path "$SCRIPT_DIR/Cargo.toml" \
  --no-default-features --features "$backend" >&2

csv_header_done=0
for block in "${BLOCK_CONFIGS[@]}"; do
  out="$(cargo run --quiet --release --manifest-path "$SCRIPT_DIR/Cargo.toml" \
    --no-default-features --features "$backend" -- \
    -s "$GRID" --block-size "$block" "$@")"
  # In CSV modes, keep a single header row across block shapes.
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
