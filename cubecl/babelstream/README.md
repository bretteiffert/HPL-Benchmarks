# BabelStream - CubeCL

## Setup

Install Rust, the vendor toolkit and driver required by the selected backend, and set vendor path.

```bash
export CUDA_PATH=[path-to-cuda]
export HIP_PATH=[path-to-rocm]
```

## Configuration

Precision is selected at compile time in `src/main.rs`:

```rust
const USE_FLOAT: bool = false; // fp64, the default
const USE_FLOAT: bool = true;  // fp32
```

Set `BS_SM_COUNT` to the NVIDIA streaming-multiprocessor count or AMD compute-unit count. The Dot kernel launches four workgroups per SM/CU; an individual run warns and falls back to 64 when the variable is absent, while both sweep scripts require it.

```bash
export BS_SM_COUNT=132
```

## NVIDIA CUDA

Run one

```bash
BS_SM_COUNT="$BS_SM_COUNT" cargo run --release --no-default-features --features cuda -- -s 33554432 -n 1000
```

Run the array-size sweep

```bash
bash run_cuda.sh
```

## AMD ROCm/HIP

Run one

```bash
BS_SM_COUNT="$BS_SM_COUNT" cargo run --release --no-default-features --features hip -- -s 33554432 -n 1000
```

Run the sweep

```bash
bash run_hip.sh
```
