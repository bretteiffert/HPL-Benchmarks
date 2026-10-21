#!/usr/bin/env bash
# BabelStream (HIP) sweep: 65,536 / 1,048,576 / 33,554,432 elements at -n 1000.
#
#   bash run.sh [extra args passed to every run]
#
# Builds once first (build output on stderr). In --csv / --csv-raw modes the
# CSV header is printed once for the whole sweep, so
#   bash run.sh --csv > results.csv
# gives one clean file. Status lines go to stderr. No environment variables
# are needed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "babelstream (HIP): building" >&2
make -C "$SCRIPT_DIR" >&2

binary="$SCRIPT_DIR/babelstream_hip"

csv=0
for arg in "$@"; do
  case "$arg" in
    --csv|--csv-raw) csv=1 ;;
  esac
done

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

first=1
for size in "${SIZE_CONFIGS[@]}"; do
  echo "babelstream (HIP): -s $size -n 1000 $*" >&2
  if [ "$csv" -eq 1 ] && [ "$first" -eq 0 ]; then
    "$binary" -s "$size" -n 1000 "$@" | tail -n +2
  else
    "$binary" -s "$size" -n 1000 "$@"
  fi
  first=0
done
