SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "Seven-point stencil - CubeCL, CUDA"

GRID=1024

BLOCK_CONFIGS=(
  "64 1 1"   "128 1 1"  "256 1 1"  "512 1 1"  "1024 1 1"
  "32 4 1"   "64 4 1"   "128 4 1"  "256 4 1"
  "32 8 1"   "64 8 1"   "128 8 1"
  "16 16 1"  "32 16 1"  "64 16 1"
)

for cfg in "${BLOCK_CONFIGS[@]}"; do
  read -r bx by bz <<< "$cfg"
  cargo run --manifest-path $SCRIPT_DIR/Cargo.toml --release --features cuda -- \
    "$GRID" "$GRID" "$GRID" "$bx" "$by" "$bz";
done
