# Hartree-Fock - CubeCL

## Setup

Install Rust, the vendor toolkit and driver required by the selected backend, and set vendor path.

```bash
export CUDA_PATH=[path-to-cuda]
export HIP_PATH=[path-to-rocm]
```

## Configuration

Precision is selected at compile time in `src/main.rs`:

```rust
const PRECISION: u32 = 64; // fp64, the default
const PRECISION: u32 = 32; // fp32
```

## NVIDIA CUDA

Run one:

```bash
cargo run --release --no-default-features --features cuda -- ../../data/hartree-fock/he64 --csv --iters=10
```

Run the `he16`, `he32`, `he64`, `he128`, and `he256` input sweep:

```bash
bash run_cuda.sh
```

## AMD ROCm/HIP

Run one

```bash
cargo run --release --no-default-features --features hip -- ../../data/hartree-fock/he64 --csv --iters=10
```

Run the sweep

```bash
bash run_hip.sh
```
