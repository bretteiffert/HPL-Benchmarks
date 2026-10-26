#!/usr/bin/env bash
# Runs the CubeCL seven-point stencil with the pinned 64x4x1 block shape.
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

# Status lines go to stderr so `--csv` output can be redirected cleanly.
echo "Seven-point stencil - CubeCL, ${backend^^}" >&2

# Build once up front so compiler output doesn't interleave with results.
cargo build --release --manifest-path "$SCRIPT_DIR/Cargo.toml" \
  --no-default-features --features "$backend" >&2

cargo run --quiet --release --manifest-path "$SCRIPT_DIR/Cargo.toml" \
  --no-default-features --features "$backend" -- \
  -s 1024 --block-size 64,4,1 "$@"
