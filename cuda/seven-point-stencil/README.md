# Seven-Point Stencil - Native CUDA

This benchmark applies a seven-point finite-difference Laplacian to a three-dimensional grid on an NVIDIA GPU. Run all commands from `cuda/seven-point-stencil`.

## Prerequisites and build

The build requires the CUDA toolkit (`nvcc`), `nvidia-smi`, and a C++17-capable host compiler. The makefile detects the compute capability of the first GPU reported by `nvidia-smi` and compiles for that architecture.

```sh
make
```

This creates `./laplacian_kernel`.

## Precision and configuration

Precision is selected at compile time in `laplacian.cu`. Keep the arithmetic type and output label consistent:

```cpp
using precision = double;
char precision_str[] = "double";
```

Use `float` and `"single"` for fp32. Rebuild after changing either value.

The positional arguments are `nx ny nz blk_x blk_y blk_z`. The block dimensions form one CUDA workgroup, so their product must not exceed the device's maximum threads per block. Initialization always uses a separate 1024-thread block, so the GPU must support 1024 threads per block even when the stencil block is smaller. Each invocation performs one correctness run followed by 100 timed kernel runs.

The default and sweep grid is `1024 x 1024 x 1024`. Its two main arrays require about 16 GiB in fp64 or 8 GiB in fp32, excluding CUDA runtime overhead; use smaller dimensions if device memory is limited.

## Run one configuration

Run the 1024-cubed grid with a `1024 x 1 x 1` workgroup:

```sh
./laplacian_kernel 1024 1024 1024 1024 1 1
```

Append `--csv` as the seventh argument to emit timing rows as CSV.

## Run the sweep

The supplied sweep holds the grid at `1024 x 1024 x 1024` and tests the script-defined block configurations, from `64 x 1 x 1` through `64 x 16 x 1`:

```sh
bash run_cuda.sh
```
