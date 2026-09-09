#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../../../data/hartree-fock"

echo "hartree-fock - KA, CUDA"

SIZE_CONFIGS=(
"he16"
"he32"
"he64"
"he128"
"he256"
)

for size in "${SIZE_CONFIGS[@]}"; do
    input="$INPUT_DIR/$size"

    STENCIL_GPU=cuda julia --project="$SCRIPT_DIR" -O3 \
    "$SCRIPT_DIR/hartree-fock.jl" "$input" --csv --iters=10
    
done