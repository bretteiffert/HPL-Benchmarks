# miniBUDE - CubeCL

## Setup

Install Rust, the vendor toolkit and driver required by the selected backend, and set vendor path.

```bash
export CUDA_PATH=[path-to-cuda]
export HIP_PATH=[path-to-rocm]
```

## Configuration

Precision is selected at compile time in `src/main.rs`

```rust
const PRECISION_BITS: usize = 64; // fp64, the default
const PRECISION_BITS: usize = 32; // fp32
```

## NVIDIA CUDA

Run one

```bash
cargo run --release --no-default-features --features cuda -- --deck ../../data/miniBUDE/bm1 -p 64 -w 32
```

Run the sweep

```bash
bash run_cuda.sh
```

## AMD ROCm/HIP

Run one

```bash
cargo run --release --no-default-features --features hip -- --deck ../../data/miniBUDE/bm1 -p 64 -w 64
```

Run the sweep

```bash
bash run_hip.sh
```
