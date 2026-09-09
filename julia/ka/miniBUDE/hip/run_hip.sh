#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../../../data/miniBUDE/bm1"

WGSIZE=64

echo "miniBUDE - KA, HIP"

PPWI_CONFIGS=(
"1"
"2"
"4"
"8"
"16"
"32"
"64"
"128"
)

for ppwi in "${PPWI_CONFIGS[@]}"; do

STENCIL_GPU=amdgpu julia --project="$SCRIPT_DIR" -O3 \
    "$SCRIPT_DIR/miniBUDE.jl" --deck "$INPUT_DIR" -w $WGSIZE -p "$ppwi"

done