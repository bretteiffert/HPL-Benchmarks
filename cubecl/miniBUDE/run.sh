#!/usr/bin/env bash
# Runs the CubeCL miniBUDE benchmark over every poses-per-work-item value at
# the backend's native work-group size. The sweep happens inside one run, so
# --csv output is a single table.
#
# Usage: ./run.sh <cuda|hip> [extra benchmark args...]
#   e.g. ./run.sh cuda --csv > results.csv
#        ./run.sh hip --float -w 64,128,256
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DECK="$SCRIPT_DIR/../../data/miniBUDE/bm1"

backend="${1:-}"
case "$backend" in
  cuda) shift; WGSIZE=32 ;;  # NVIDIA warp size
  hip)  shift; WGSIZE=64 ;;  # AMD wavefront size
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

# Later flags win, so extra arguments (e.g. -w 64,128) override these defaults.
cargo run --quiet --release --manifest-path "$SCRIPT_DIR/Cargo.toml" \
  --no-default-features --features "$backend" -- \
  --deck "$DECK" -p all -w "$WGSIZE" "$@"
