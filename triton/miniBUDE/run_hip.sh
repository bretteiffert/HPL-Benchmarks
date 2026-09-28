#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../data/miniBUDE/bm1"

WGSIZE=64

echo "miniBUDE - Triton, HIP"

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

pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" -e hip \
    python "$SCRIPT_DIR/miniBUDE.py" \
    --deck "$INPUT_DIR" -w $WGSIZE -p "$ppwi"

done
