# Hartree-Fock - CubeCL

This CubeCL benchmark builds the two-electron Fock contribution for Hartree-Fock inputs on NVIDIA and AMD GPUs. It uses a fixed 256-thread workgroup, device-profiled timings, and atomic updates to the Fock matrix. Run all commands from `cubecl/hartree-fock`.

## Setup

Install Rust and the vendor toolkit and driver required by the selected backend.

## Configuration

Precision is selected at compile time in `src/main.rs`:

```rust
const PRECISION: u32 = 64; // fp64, the default
const PRECISION: u32 = 32; // fp32
```

`LEGACY_SQRTF` defaults to `false`, which uses native-precision square roots. Setting it to `true` reproduces the legacy behavior that converts the relevant square roots through fp32.

The first positional argument is the input path. In normal output mode, a CPU fp64 correctness check runs by default for inputs with at most 128 atoms. Use `--check` to force it for larger inputs, `--no-check` to disable it, or `--max-check-atoms=N` to change the automatic limit. CSV mode reports timing samples and does not run or print the CPU check. `--iters=N` controls the number of CSV timing iterations and defaults to 10.

The kernel uses atomic additions in the selected precision, so the target backend and GPU must support the corresponding atomics. The implementation also rejects inputs whose launch would exceed CubeCL's 32-bit global index space.

## NVIDIA CUDA

Run one input in CSV timing mode:

```bash
cargo run --release --no-default-features --features cuda -- ../../data/hartree-fock/he64 --csv --iters=10
```

Run the `he16`, `he32`, `he64`, `he128`, and `he256` input sweep:

```bash
bash run_cuda.sh
```

## AMD ROCm/HIP

Run one input in CSV timing mode:

```bash
cargo run --release --no-default-features --features hip -- ../../data/hartree-fock/he64 --csv --iters=10
```

Run the same input-size sweep:

```bash
bash run_hip.sh
```
