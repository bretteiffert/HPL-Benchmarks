SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "babelstream - Triton, CUDA"

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

for size in "${SIZE_CONFIGS[@]}"; do
  FMAD=1 pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" -e cuda \
    python "$SCRIPT_DIR/babelstream.py" \
    -s "$size" -n 1000
done
