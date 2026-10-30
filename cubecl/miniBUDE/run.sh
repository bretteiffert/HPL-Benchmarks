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

# Cargo reads .cargo/config.toml from the working directory, not from
# --manifest-path. Use GNU host compilers for native Rust dependencies rather
# than any compiler injected by the NVIDIA HPC SDK.
cd "$SCRIPT_DIR"
export CC=gcc CXX=g++

# Status lines go to stderr so `--csv` output can be redirected cleanly.
echo "miniBUDE - CubeCL, ${backend^^}" >&2

# Build first so compiler output doesn't interleave with results.
cargo build --release --no-default-features --features "$backend" >&2

cargo run --quiet --release --no-default-features --features "$backend" -- \
  --deck "$DECK" --float -p 1 -w 256 --warmup 2 -i 8 "$@"
