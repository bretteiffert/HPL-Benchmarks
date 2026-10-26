#!/usr/bin/env bash
# Seven-point stencil (KernelAbstractions), pinned 64x4x1 workgroup.
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

# Build once: install the pinned Manifest and precompile (output to stderr).
echo "run.sh: preparing $PROJECT" >&2
"$JULIA" --project="$PROJECT" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' 1>&2

echo "run.sh: $backend, grid 1024, block 64,4,1" >&2
"$JULIA" --project="$PROJECT" -O3 --check-bounds=no "$ROOT/main.jl" \
  -s 1024 -n 100 --block-size 64,4,1 "$@"
