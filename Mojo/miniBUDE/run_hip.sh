#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../data/miniBUDE/bm1"

WGSIZE=64

echo "miniBUDE - Mojo, HIP"

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

pixi run mojo "$SCRIPT_DIR/miniBUDE.mojo" \
    --deck "$INPUT_DIR" -w $WGSIZE -p "$ppwi"

done