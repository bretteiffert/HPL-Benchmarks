SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "babelstream - KernelAbstractions, CUDA"

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

for size in "${SIZE_CONFIGS[@]}"; do
  STENCIL_GPU=cuda julia --project="$SCRIPT_DIR" -O3 --check-bounds=no \
    "$SCRIPT_DIR/babelstream.jl" \
    -s "$size" -n 1000
done