# BabelStream - Julia KernelAbstractions

Run a single configuration or the supplied size sweep on NVIDIA and AMD GPUs. Run all commands from `julia/ka/babelStream`.

## Configuration

Select precision in both `cuda/babelstream.jl` and `hip/babelstream.jl`:

```julia
const USE_FLOAT = false # Float64
const USE_FLOAT = true  # Float32
```

The `--float` option is currently a no-op; precision is selected only by `USE_FLOAT`. Array sizes must be multiples of 1024.

Select the backend with `STENCIL_GPU=cuda` for NVIDIA or `STENCIL_GPU=amdgpu` for AMD. Use the matching project below because each project contains only its vendor GPU package.

## Setup

```bash
julia --project=cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=hip -e 'using Pkg; Pkg.instantiate()'
```

## NVIDIA CUDA

Run one configuration:

```bash
STENCIL_GPU=cuda julia --project=cuda -O3 --check-bounds=no cuda/babelstream.jl -s 33554432 -n 1000
```

Run the size sweep:

```bash
source cuda/run_cuda.sh
```

## AMD ROCm

Run one configuration:

```bash
STENCIL_GPU=amdgpu julia --project=hip -O3 --check-bounds=no hip/babelstream.jl -s 33554432 -n 1000
```

Run the size sweep:

```bash
source hip/run_hip.sh
```
