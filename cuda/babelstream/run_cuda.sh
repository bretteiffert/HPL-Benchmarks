SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "babelstream - Native, CUDA"

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

for size in "${SIZE_CONFIGS[@]}"; do
  "$SCRIPT_DIR/babelstream_cuda" \
    -s "$size" -n 1000
done