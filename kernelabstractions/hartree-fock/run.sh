#!/usr/bin/env bash
# Hartree-Fock (KernelAbstractions) sweep over he16 .. he256.
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
DATA="$ROOT/../../data/hartree-fock"
INPUTS=(he16 he32 he64 he128 he256)

for name in "${INPUTS[@]}"; do
  if [[ ! -f "$DATA/$name" ]]; then
    echo "error: input file not found: $DATA/$name" >&2; exit 1
  fi
done

csv=0
for a in "$@"; do
  if [[ "$a" == "--csv" || "$a" == "--csv-raw" ]]; then csv=1; fi
done

# Build once: install the pinned Manifest and precompile (output to stderr).
echo "run.sh: preparing $PROJECT" >&2
"$JULIA" --project="$PROJECT" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' 1>&2

first=1
for name in "${INPUTS[@]}"; do
  echo "run.sh: $backend, input $name" >&2
  cmd=("$JULIA" --project="$PROJECT" -O3 --check-bounds=no "$ROOT/main.jl"
       "$DATA/$name" -n 10 "$@")
  if [[ $csv -eq 1 && $first -eq 0 ]]; then
    "${cmd[@]}" | tail -n +2        # CSV header only once across the sweep
  else
    "${cmd[@]}"
  fi
  first=0
done
