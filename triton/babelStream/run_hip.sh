SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "babelstream - Triton, HIP"

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

for size in "${SIZE_CONFIGS[@]}"; do
  FMAD=1 pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" -e hip \
    python "$SCRIPT_DIR/babelstream.py" \
    -s "$size" -n 1000
done
