#!/usr/bin/env bash
# Runs the CubeCL BabelStream benchmark across the configured array sizes.
#
# Usage: ./run.sh <cuda|hip> [extra benchmark args...]
#   e.g. ./run.sh cuda --csv > results.csv
#        ./run.sh hip --float --block-size 256
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

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

# Status lines go to stderr so `--csv` output can be redirected cleanly.
echo "babelstream - CubeCL, ${backend^^}" >&2

# Build once up front so compiler output doesn't interleave with results.
cargo build --release --manifest-path "$SCRIPT_DIR/Cargo.toml" \
  --no-default-features --features "$backend" >&2

csv_header_done=0
for size in "${SIZE_CONFIGS[@]}"; do
  out="$(cargo run --quiet --release --manifest-path "$SCRIPT_DIR/Cargo.toml" \
    --no-default-features --features "$backend" -- \
    -s "$size" -n 1000 "$@")"
  # In CSV modes, keep a single header row across sizes.
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
