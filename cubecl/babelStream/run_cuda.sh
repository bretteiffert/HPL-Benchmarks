# Runs the CubeCL BabelStream CUDA benchmark across the configured array sizes.
# BS_SM_COUNT must be set to the target GPU's streaming-multiprocessor count because the benchmark does not query it.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "babelstream - CubeCL, CUDA"

: "${BS_SM_COUNT:?set BS_SM_COUNT to your device SM/CU count}"

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

for size in "${SIZE_CONFIGS[@]}"; do
  BS_SM_COUNT="$BS_SM_COUNT" \
    cargo run --release --manifest-path "$SCRIPT_DIR/Cargo.toml" \
      --no-default-features --features cuda -- \
      -s "$size" -n 1000
done
