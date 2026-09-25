# Seven-point Stencil - CubeCL

This CubeCL benchmark applies a three-dimensional seven-point Laplacian on NVIDIA and AMD GPUs. It profiles 100 kernel iterations, calculates effective memory bandwidth, and checks the interior output against the analytical value 3. Run all commands from `cubecl/seven-point-stencil`.

## Setup

Install Rust and the vendor toolkit and driver required by the selected backend. Dependencies are pinned to CubeCL 0.10.0 and are fetched by Cargo when the benchmark is built.

Before building either backend, update or remove the `cubecl-hip-sys` patch in `Cargo.toml`; it currently points to the machine-local path `/home/35e/cubecl-hip-sys/crates/cubecl-hip-sys`.

## Configuration

Precision is selected at compile time in `src/main.rs`. Keep the type and output label consistent:

```rust
type Precision = f64;
const PRECISION_STR: &str = "double"; // default
```

or:

```rust
type Precision = f32;
const PRECISION_STR: &str = "single";
```

The positional arguments are `nx ny nz blk_x blk_y blk_z`. The final three values define the Laplacian workgroup, and their product must be supported by the target GPU. Initialization separately uses a fixed 1024-thread one-dimensional workgroup.

Both scripts sweep the same workgroups, from `64 1 1` through configurations containing 1024 threads, on a `1024 x 1024 x 1024` grid. The grid allocates two full device arrays: 8 GiB in fp32 or 16 GiB in fp64, excluding runtime overhead. Use smaller dimensions if the GPU does not have enough memory. Set `GPU_NAME` to customize the label in output; it defaults to `unknown-gpu`.

## NVIDIA CUDA

Run one configuration:

```bash
cargo run --release --no-default-features --features cuda -- 1024 1024 1024 1024 1 1
```

Run the workgroup sweep:

```bash
bash run_cuda.sh
```

## AMD ROCm/HIP

Run one configuration:

```bash
cargo run --release --no-default-features --features hip -- 1024 1024 1024 512 1 1
```

Run the same workgroup sweep:

```bash
bash run_hip.sh
```
