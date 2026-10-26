#!/usr/bin/env bash
# Seven-point stencil (CUDA): pinned 64x4x1 block on a 1024^3 grid.
#
#   bash run.sh [extra args passed to every run]
#
# Builds once first; build and status output go to stderr.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "stencil (CUDA): building" >&2
make -C "$SCRIPT_DIR" >&2

binary="$SCRIPT_DIR/laplacian_kernel"
echo "stencil (CUDA): -s 1024 --block-size 64,4,1 $*" >&2
"$binary" -s 1024 --block-size 64,4,1 "$@"
