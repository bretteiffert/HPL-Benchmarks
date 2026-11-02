#!/usr/bin/env bash
# Hartree-Fock (KernelAbstractions) sweep: he16, he32, he64, he128, he256 at -n 10.
#
#   bash run.sh <cuda|hip> [extra args passed to every run]
#
# Inputs are read from ../../data/hartree-fock next to this folder. Installs
# the backend's Julia packages first (output on stderr). In --csv /
# --csv-raw modes the CSV header is printed once for the whole sweep, so
#   bash run.sh hip --csv > results.csv
# gives one clean file. Status lines go to stderr. No environment variables
# are needed.

set -euo pipefail

usage="usage: bash run.sh <cuda|hip> [extra args passed to every run]"
if [ "$#" -lt 1 ]; then
  echo "error: missing backend; $usage" >&2
  exit 1
fi
backend="$1"
shift
case "$backend" in
  cuda|hip) ;;
  *) echo "error: unknown backend '$backend' (valid: cuda, hip); $usage" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../data/hartree-fock"

if ! command -v julia >/dev/null 2>&1; then
  echo "error: julia not found on the PATH; install Julia 1.10 or newer" >&2
  exit 1
fi

# Install and precompile the backend's packages (quick when already done).
echo "hartree-fock (KernelAbstractions, $backend): installing packages" >&2
julia --project="$SCRIPT_DIR/env/$backend" --startup-file=no \
  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' >&2

run_one() {
  julia --project="$SCRIPT_DIR/env/$backend" --startup-file=no --history-file=no -O3 \
    "$SCRIPT_DIR/main.jl" "$@"
}

csv=0
for arg in "$@"; do
  case "$arg" in
    --csv|--csv-raw) csv=1 ;;
  esac
done

SIZE_CONFIGS=(
  "he16"
  "he32"
  "he64"
  "he128"
  "he256"
)

first=1
for size in "${SIZE_CONFIGS[@]}"; do
  input="$INPUT_DIR/$size"
  echo "hartree-fock (KernelAbstractions, $backend): $size -n 10 $*" >&2
  if [ "$csv" -eq 1 ] && [ "$first" -eq 0 ]; then
    run_one "$input" -n 10 --block-size 256 "$@" | tail -n +2
  else
    run_one "$input" -n 10 --block-size 256 "$@"
  fi
  first=0
done
