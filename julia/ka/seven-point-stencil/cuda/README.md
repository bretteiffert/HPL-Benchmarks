# Seven-point Stencil - Julia KernelAbstractions CUDA

This directory runs a seven-point Laplacian stencil on NVIDIA GPUs through CUDA.jl and KernelAbstractions.jl. Run the commands below from this directory.

## Setup

Install the Julia dependencies from the local project:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Configuration

Select the CUDA backend with `STENCIL_GPU=cuda`. The checked-in `laplacian.jl` defaults to `Float32`:

```julia
const precision = Float32
const precision_str = "float"
```

To use `Float64`, change both values to `Float64` and `"double"` so the arithmetic and output label remain consistent.

## Run one

The six positional values are the three grid dimensions followed by the three workgroup dimensions:

```bash
STENCIL_GPU=cuda julia --project=. -O3 --check-bounds=no laplacian.jl 1024 1024 1024 1024 1 1
```

Initialization always uses a separate 1024-thread workgroup, so the GPU must support 1024 threads per workgroup even when the stencil workgroup is smaller.

## Run sweep

The supplied script runs a 1024 x 1024 x 1024 grid over its workgroup configurations:

```bash
bash ./run_cuda.sh
```

## Caveats

- A 1024 x 1024 x 1024 run allocates two full GPU arrays: 8 GiB for `Float32` or 16 GiB for `Float64`, excluding runtime overhead.
- The benchmark reports 100 event-timed stencil iterations for each configuration.
