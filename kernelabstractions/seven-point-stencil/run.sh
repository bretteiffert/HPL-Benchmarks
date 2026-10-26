#!/usr/bin/env bash
# Seven-point stencil (KernelAbstractions) sweep: 15 workgroup shapes on a
# 1024³ grid.
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

# Every shape has 1024 work-items.
BLOCKS=(
  1024,1,1 512,2,1 256,4,1 128,8,1 64,16,1 32,32,1
  512,1,2 256,2,2 128,4,2 64,8,2 32,16,2
  256,1,4 128,2,4 64,4,4 32,8,4
)

csv=0
for a in "$@"; do
  if [[ "$a" == "--csv" || "$a" == "--csv-raw" ]]; then csv=1; fi
done

# Build once: install the pinned Manifest and precompile (output to stderr).
echo "run.sh: preparing $PROJECT" >&2
"$JULIA" --project="$PROJECT" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' 1>&2

first=1
for block in "${BLOCKS[@]}"; do
  echo "run.sh: $backend, grid 1024, block $block" >&2
  cmd=("$JULIA" --project="$PROJECT" -O3 --check-bounds=no "$ROOT/main.jl"
       -s 1024 -n 100 --block-size "$block" "$@")
  if [[ $csv -eq 1 && $first -eq 0 ]]; then
    "${cmd[@]}" | tail -n +2        # CSV header only once across the sweep
  else
    "${cmd[@]}"
  fi
  first=0
done
