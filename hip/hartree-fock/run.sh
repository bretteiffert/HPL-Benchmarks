#!/usr/bin/env bash
# Hartree-Fock (HIP) sweep: he16, he32, he64, he128, he256 at -n 10.
#
#   bash run.sh [extra args passed to every run]
#
# Inputs are read from ../../data/hartree-fock next to this folder. Builds
# once first (build output on stderr). In --csv / --csv-raw modes the CSV
# header is printed once for the whole sweep, so
#   bash run.sh --csv > results.csv
# gives one clean file. Status lines go to stderr. No environment variables
# are needed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../data/hartree-fock"

echo "hartree-fock (HIP): building" >&2
make -C "$SCRIPT_DIR" >&2

binary="$SCRIPT_DIR/hartree-fock-hip"

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
  echo "hartree-fock (HIP): $size -n 10 $*" >&2
  if [ "$csv" -eq 1 ] && [ "$first" -eq 0 ]; then
    "$binary" "$input" -n 10 --block-size 256 "$@" | tail -n +2
  else
    "$binary" "$input" -n 10 --block-size 256 "$@"
  fi
  first=0
done
