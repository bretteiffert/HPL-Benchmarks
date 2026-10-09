# Seven-point Stencil - CubeCL

## Setup

Install Rust, the vendor toolkit and driver required by the selected backend, and set vendor path.

```bash
export CUDA_PATH=[path-to-cuda]
export HIP_PATH=[path-to-rocm]
```

## Configuration

Precision is selected at compile time in `src/main.rs`

```rust
type Precision = f64;
const PRECISION_STR: &str = "double"; // default
```

or

```rust
type Precision = f32;
const PRECISION_STR: &str = "single";
```

## NVIDIA CUDA

Run one

```bash
cargo run --release --no-default-features --features cuda -- 1024 1024 1024 1024 1 1
```

Run the sweep

```bash
bash run_cuda.sh
```

## AMD ROCm/HIP

Run one

```bash
cargo run --release --no-default-features --features hip -- 1024 1024 1024 512 1 1
```

Run the sweep

```bash
bash run_hip.sh
```
