#!/usr/bin/env bash
# miniBUDE (Triton): bash run.sh <cuda|hip> [args...]
# One run uses the pinned PPWI 1, work-group 256 configuration.
# Extra arguments are passed through, e.g. bash run.sh cuda --float --csv > results.csv
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../data/miniBUDE/bm1"

usage() { echo "usage: bash run.sh <cuda|hip> [args...]" >&2; }

if [ $# -lt 1 ]; then
  echo "error: missing backend (valid: cuda, hip)" >&2
  usage
  exit 1
fi
backend="$1"
shift
case "$backend" in
  cuda|hip) ;;
  *) echo "error: unknown backend '$backend' (valid: cuda, hip)" >&2; usage; exit 1 ;;
esac

echo "miniBUDE - Triton, $backend" >&2
echo "installing the '$backend' environment" >&2
pixi install --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" 1>&2

echo "running --deck bm1 --float -p 1 -w 256 --warmup 2 -i 8 $*" >&2
pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" \
  python "$SCRIPT_DIR/main.py" --deck "$INPUT_DIR" --float -p 1 -w 256 --warmup 2 -i 8 "$@"
