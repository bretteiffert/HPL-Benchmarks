#!/usr/bin/env bash
# Runs CubeCL miniBUDE with pinned PPWI 1 and work-group 256 on bm1.
#
# Usage: ./run.sh <cuda|hip> [extra benchmark args...]
#   e.g. ./run.sh cuda --csv > results.csv
#        ./run.sh hip --float -w 64,128,256
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DECK="$SCRIPT_DIR/../../data/miniBUDE/bm1"

backend="${1:-}"
case "$backend" in
  cuda|hip) shift ;;
  *)
    echo "usage: $0 <cuda|hip> [benchmark args...]" >&2
    exit 1
    ;;
esac

# Status lines go to stderr so `--csv` output can be redirected cleanly.
echo "miniBUDE - CubeCL, ${backend^^}" >&2

# Build first so compiler output doesn't interleave with results.
cargo build --release --manifest-path "$SCRIPT_DIR/Cargo.toml" \
  --no-default-features --features "$backend" >&2

cargo run --quiet --release --manifest-path "$SCRIPT_DIR/Cargo.toml" \
  --no-default-features --features "$backend" -- \
  --deck "$DECK" --float -p 1 -w 256 --warmup 2 -i 8 "$@"
