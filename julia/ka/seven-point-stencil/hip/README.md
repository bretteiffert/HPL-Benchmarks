# Seven-point Stencil - Julia KernelAbstractions HIP

This directory runs a seven-point Laplacian stencil on AMD GPUs through ROCm, AMDGPU.jl, and KernelAbstractions.jl. Run the commands below from this directory.

## Setup

Install the Julia dependencies from the local project:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Configuration

Select the AMD GPU backend with `STENCIL_GPU=amdgpu`. The checked-in `laplacian.jl` defaults to `Float64`:

```julia
const precision = Float64
const precision_str = "double"
```

To use `Float32`, change both values to `Float32` and `"float"` so the arithmetic and output label remain consistent.

## Run one

The six positional values are the three grid dimensions followed by the three workgroup dimensions:

```bash
STENCIL_GPU=amdgpu julia --project=. -O3 --check-bounds=no laplacian.jl 1024 1024 1024 1024 1 1
```

Initialization always uses a separate 1024-thread workgroup, so the GPU must support 1024 threads per workgroup even when the stencil workgroup is smaller.

## Run sweep

The supplied script runs a 1024 x 1024 x 1024 grid over its workgroup configurations:

```bash
bash ./run_hip.sh
```

## Caveats

- A 1024 x 1024 x 1024 run allocates two full GPU arrays: 8 GiB for `Float32` or 16 GiB for `Float64`, excluding runtime overhead.
- The benchmark reports 100 event-timed stencil iterations for each configuration.
