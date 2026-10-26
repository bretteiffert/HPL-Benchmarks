#!/usr/bin/env bash
# Seven-point stencil (Triton), pinned 64x4x1 configuration.
# Extra arguments are passed to every run, e.g. bash run.sh cuda --float --csv > results.csv
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

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

echo "stencil - Triton, $backend" >&2
echo "installing the '$backend' environment" >&2
pixi install --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" 1>&2

echo "running -s 1024 --block-size 64,4,1 $*" >&2
pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" -e "$backend" \
  python "$SCRIPT_DIR/main.py" -s 1024 --block-size 64,4,1 "$@"
