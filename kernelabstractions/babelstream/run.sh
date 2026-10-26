#!/usr/bin/env bash
# BabelStream (KernelAbstractions) sweep.
#
#   bash run.sh <cuda|hip> [extra args passed to every run]
#
# Example: bash run.sh cuda --float --csv > results.csv
set -euo pipefail

usage() { echo "usage: bash run.sh <cuda|hip> [args...]" >&2; }

if [[ $# -lt 1 ]]; then
  usage; exit 1
fi
backend="$1"; shift
case "$backend" in
  cuda|hip) ;;
  *) echo "error: unknown backend '$backend' (valid: cuda, hip)" >&2; usage; exit 1 ;;
esac

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JULIA="${JULIA_BIN:-julia}"
PROJECT="$ROOT/env/$backend"

csv=0
for a in "$@"; do
  [[ "$a" == "--csv" || "$a" == "--csv-raw" ]] && csv=1
done

# Build once: install the pinned Manifest and precompile (output to stderr).
echo "run.sh: preparing $PROJECT" >&2
"$JULIA" --project="$PROJECT" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' 1>&2

SIZES=(65536 1048576 33554432)

first=1
for size in "${SIZES[@]}"; do
  echo "run.sh: $backend, array size $size" >&2
  cmd=("$JULIA" --project="$PROJECT" -O3 --check-bounds=no "$ROOT/main.jl"
       -s "$size" -n 1000 "$@")
  if [[ $csv -eq 1 && $first -eq 0 ]]; then
    "${cmd[@]}" | tail -n +2        # CSV header only once across the sweep
  else
    "${cmd[@]}"
  fi
  first=0
done
