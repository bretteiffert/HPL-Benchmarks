#!/usr/bin/env bash
# miniBUDE (KernelAbstractions): pinned PPWI 1 and work-group 256 on bm1.
#
#   bash run.sh <cuda|hip> [extra args passed to the run]
#
# Example: bash run.sh hip --float --csv > results.csv
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
DECK="$ROOT/../../data/miniBUDE/bm1"

if [[ ! -d "$DECK" ]]; then
  echo "error: deck directory not found: $DECK" >&2; exit 1
fi

# Build once: install the pinned Manifest and precompile (output to stderr).
echo "run.sh: preparing $PROJECT" >&2
"$JULIA" --project="$PROJECT" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' 1>&2

echo "run.sh: $backend, deck bm1, float32, ppwi 1, wgsize 256" >&2
"$JULIA" --project="$PROJECT" -O3 --check-bounds=no "$ROOT/main.jl" \
  --deck "$DECK" --float -p 1 -w 256 --warmup 2 -i 8 "$@"
