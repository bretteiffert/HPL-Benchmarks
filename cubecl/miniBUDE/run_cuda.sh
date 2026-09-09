#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../data/miniBUDE"

WGSIZE=32

echo "miniBUDE - CubeCL, CUDA"

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

cargo run --manifest-path "$SCRIPT_DIR/Cargo.toml" --release --features cuda -- --deck "$INPUT_DIR" -p "$ppwi" -w "$WGSIZE"

done