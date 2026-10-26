#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/roofline/babelstream}"
NCU_BIN="${NCU_BIN:-ncu}"
SKIP_CUBECL_BUILD="${SKIP_CUBECL_BUILD:-0}"

ARRAY_SIZE="${ARRAY_SIZE:-33554432}"
ITERATIONS="${ITERATIONS:-10}"
DEVICE="${DEVICE:-0}"
BLOCK_SIZE=256

COMMON_ARGS=(
  -s "$ARRAY_SIZE"
  -n "$ITERATIONS"
  --device "$DEVICE"
  --block-size "$BLOCK_SIZE"
  --no-energy
)

show_help() {
  cat <<'EOF'
Usage:
  ./run_cuda_roofline.sh --babelstream
  ./run_cuda_roofline.sh --stencil
  ./run_cuda_roofline.sh --hartree-fock
  ./run_cuda_roofline.sh --miniBUDE
  ./run_cuda_roofline.sh --all

Environment overrides:
  ARRAY_SIZE=33554432
  ITERATIONS=10
  DEVICE=0
  NCU_BIN=ncu
  OUTPUT_DIR=output/roofline/babelstream
  SKIP_CUBECL_BUILD=0       # set to 1 to reuse an existing release binary

BabelStream runs FP16, FP32, and FP64 as separate roofline profiles.
EOF
}

profile() {
  local name="$1"
  shift

  echo "roofline: $name" >&2
  "$NCU_BIN" \
    --target-processes all \
    --section SpeedOfLight_RooflineChart \
    --force-overwrite \
    --export "$OUTPUT_DIR/$name" \
    "$@"
}

run_babelstream() {
  mkdir -p "$OUTPUT_DIR"

  echo "Preparing native CUDA BabelStream" >&2
  make -C "$ROOT/cuda/babelstream" >&2

  echo "Preparing Triton CUDA environment" >&2
  pixi install --manifest-path "$ROOT/triton/babelstream/pixi.toml" -e cuda >&2

  echo "Preparing Mojo CUDA environment" >&2
  pixi install --manifest-path "$ROOT/Mojo/babelstream" >&2

  echo "Preparing CubeCL CUDA binary" >&2
  if [[ "$SKIP_CUBECL_BUILD" == 1 ]]; then
    if [[ ! -x "$ROOT/cubecl/babelstream/target/release/babelstream" ]]; then
      echo "error: SKIP_CUBECL_BUILD=1 but the CubeCL release binary does not exist" >&2
      exit 1
    fi
    echo "Reusing existing CubeCL release binary" >&2
  else
    cargo build --release \
      --manifest-path "$ROOT/cubecl/babelstream/Cargo.toml" \
      --no-default-features --features cuda >&2
  fi

  echo "Preparing KernelAbstractions CUDA environment" >&2
  julia --project="$ROOT/kernelabstractions/babelstream/env/cuda" \
    -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' >&2

  local precision_spec precision precision_flag
  for precision_spec in fp16:--half fp32:--float fp64:--double; do
    IFS=: read -r precision precision_flag <<< "$precision_spec"
    local run_args=("${COMMON_ARGS[@]}" "$precision_flag")

    profile "babelstream_cuda_$precision" \
      "$ROOT/cuda/babelstream/babelstream_cuda" \
      "${run_args[@]}"

    profile "babelstream_triton_cuda_$precision" \
      pixi run --manifest-path "$ROOT/triton/babelstream/pixi.toml" -e cuda \
      python "$ROOT/triton/babelstream/main.py" \
      "${run_args[@]}"

    profile "babelstream_mojo_cuda_$precision" \
      pixi run --manifest-path "$ROOT/Mojo/babelstream" \
      mojo run -D BACKEND=cuda "$ROOT/Mojo/babelstream/main.mojo" \
      "${run_args[@]}"

    profile "babelstream_cubecl_cuda_$precision" \
      "$ROOT/cubecl/babelstream/target/release/babelstream" \
      "${run_args[@]}"

    profile "babelstream_kernelabstractions_cuda_$precision" \
      julia --project="$ROOT/kernelabstractions/babelstream/env/cuda" \
      -O3 --check-bounds=no "$ROOT/kernelabstractions/babelstream/main.jl" \
      "${run_args[@]}"
  done
}

run_stencil() {
  # TODO: add pinned NVIDIA seven-point-stencil roofline commands.
  echo "roofline: seven-point-stencil commands not added yet" >&2
}

run_hartreefock() {
  # TODO: add pinned NVIDIA Hartree-Fock roofline commands.
  echo "roofline: Hartree-Fock commands not added yet" >&2
}

run_minibude() {
  # TODO: add pinned NVIDIA miniBUDE roofline commands.
  echo "roofline: miniBUDE commands not added yet" >&2
}

if [[ $# -ne 1 ]]; then
  show_help >&2
  exit 1
fi

case "$1" in
  --babelstream)
    run_babelstream
    ;;
  --stencil)
    run_stencil
    ;;
  --hartree-fock)
    run_hartreefock
    ;;
  --miniBUDE)
    run_minibude
    ;;
  --all)
    run_stencil
    run_babelstream
    run_hartreefock
    run_minibude
    ;;
  --help|-h)
    show_help
    ;;
  *)
    echo "error: unknown option '$1'" >&2
    show_help >&2
    exit 1
    ;;
esac
