# miniBUDE - CubeCL

This CubeCL implementation evaluates the miniBUDE molecular-docking energy kernel on NVIDIA and AMD GPUs. It supports individual work configurations and poses-per-work-item (PPWI) sweeps, with device-profiled timings and numerical comparison metrics. Run all commands from `cubecl/miniBUDE`.

## Setup

Install Rust and the vendor toolkit and driver required by the selected backend. Dependencies are pinned to CubeCL 0.10.0 and are fetched by Cargo when the benchmark is built. The benchmark data is included under `../../data/miniBUDE`.

Before building either backend, update or remove the `cubecl-hip-sys` patch in `Cargo.toml`; it currently points to the machine-local path `/home/35e/cubecl-hip-sys/crates/cubecl-hip-sys`.

## Configuration

Precision is selected at compile time in `src/main.rs`; there is no precision command-line option:

```rust
const PRECISION_BITS: usize = 64; // fp64, the default
const PRECISION_BITS: usize = 32; // fp32
```

The default input deck is `../../data/miniBUDE/bm1`. Select another self-contained deck with `--deck DIR`. The sweep scripts explicitly use `../../data/miniBUDE`, whose top-level files also form a complete deck.

Use `-p`/`--ppwi` for one or more comma-separated PPWI values from `1,2,4,8,16,32,64,128`, and `-w`/`--wgsize` for one or more comma-separated workgroup sizes. The pose count must be at least and exactly divisible by `ppwi * wgsize`; invalid combinations are skipped. fp32 benchmark runs first compute an fp64 differential reference using the first runnable configuration. Comparison against the deck's low-precision `ref_energies.out` is informational.

## NVIDIA CUDA

Run one configuration using the CUDA sweep's 32-thread workgroup:

```bash
cargo run --release --no-default-features --features cuda -- --deck ../../data/miniBUDE/bm1 -p 64 -w 32
```

Run the PPWI sweep from 1 through 128 with workgroup size 32:

```bash
bash run_cuda.sh
```

## AMD ROCm/HIP

Run one configuration using the HIP sweep's 64-thread workgroup:

```bash
cargo run --release --no-default-features --features hip -- --deck ../../data/miniBUDE/bm1 -p 64 -w 64
```

Run the PPWI sweep from 1 through 128 with workgroup size 64:

```bash
bash run_hip.sh
```
