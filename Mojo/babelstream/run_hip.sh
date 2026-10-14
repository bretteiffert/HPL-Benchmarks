# Runs the Mojo BabelStream benchmark for three array sizes on HIP with 1,000 iterations each.
# BS_SM_COUNT must already contain the AMD device's positive CU count because the benchmark derives its dot grid from it.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "babelstream - Mojo, HIP"

: "${BS_SM_COUNT:?set BS_SM_COUNT to your device SM/CU count}"

SIZE_CONFIGS=(
  "65536"
  "1048576"
  "33554432"
)

for size in "${SIZE_CONFIGS[@]}"; do
  BS_SM_COUNT="$BS_SM_COUNT" pixi run --manifest-path "$SCRIPT_DIR" mojo "$SCRIPT_DIR/babelstream.mojo" \
    -s "$size" -n 1000
done
