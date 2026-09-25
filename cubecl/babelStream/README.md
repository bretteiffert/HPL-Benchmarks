# BabelStream - CubeCL

This CubeCL implementation measures Copy, Mul, Add, Triad, and Dot memory-bandwidth kernels on NVIDIA and AMD GPUs. It uses device-profiled kernel times, reports correctness metrics after the run, and selects the intended GPU backend through Cargo features. Run all commands from `cubecl/babelStream`.

## Setup

Install Rust and the vendor toolkit and driver required by the selected backend. Dependencies are pinned to CubeCL 0.10.0 and are fetched by Cargo when the benchmark is built.

Before building either backend, update or remove the `cubecl-hip-sys` patch in `Cargo.toml`; it currently points to the machine-local path `/home/35e/cubecl-hip-sys/crates/cubecl-hip-sys`.

## Configuration

Precision is selected at compile time in `src/main.rs`:

```rust
const USE_FLOAT: bool = false; // fp64, the default
const USE_FLOAT: bool = true;  // fp32
```

The `--float` option is a no-op and does not override `USE_FLOAT`. Array sizes must be multiples of the fixed 1024-thread workgroup size, and `-n` must be at least 2 because the first timing sample is excluded from the summary.

Set `BS_SM_COUNT` to the NVIDIA streaming-multiprocessor count or AMD compute-unit count. The Dot kernel launches four workgroups per SM/CU; an individual run warns and falls back to 64 when the variable is absent, while both sweep scripts require it.

```bash
export BS_SM_COUNT=132
```

Optional `BS_GPU_NAME` and `BS_DRIVER` values affect reported metadata only. `FMAD` defaults to `1`, but both source branches currently evaluate the same multiply-add expression, so it does not guarantee or disable contraction.

## NVIDIA CUDA

Run one configuration:

```bash
BS_SM_COUNT="$BS_SM_COUNT" cargo run --release --no-default-features --features cuda -- -s 33554432 -n 1000
```

Run the array-size sweep (`65536`, `1048576`, and `33554432` elements):

```bash
bash run_cuda.sh
```

## AMD ROCm/HIP

Run one configuration:

```bash
BS_SM_COUNT="$BS_SM_COUNT" cargo run --release --no-default-features --features hip -- -s 33554432 -n 1000
```

Run the same array-size sweep:

```bash
bash run_hip.sh
```
