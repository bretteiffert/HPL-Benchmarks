SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

make -C "$SCRIPT_DIR" clean
make -C "$SCRIPT_DIR"

echo "babelstream - Native, CUDA"

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

for size in "${SIZE_CONFIGS[@]}"; do
  "$SCRIPT_DIR/babelstream_cuda" \
    -s "$size" -n 1250
done
