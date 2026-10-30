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

# Cargo reads .cargo/config.toml from the working directory, not from
# --manifest-path, so run from the project dir. The exports are a fallback in
# case the config is missing: native deps (ring, liblzma, llvm-sys) must be
# built with GNU tools, not nvc from the NVIDIA HPC SDK. Only this script's
# environment changes.
cd "$SCRIPT_DIR"
export CC=gcc CXX=g++

# Status lines go to stderr so `--csv` output can be redirected cleanly.
echo "Seven-point stencil - CubeCL, ${backend^^}" >&2

# Build once up front so compiler output doesn't interleave with results.
cargo build --release --no-default-features --features "$backend" >&2

cargo run --quiet --release --no-default-features --features "$backend" -- \
  -s 1024 --block-size 64,4,1 "$@"