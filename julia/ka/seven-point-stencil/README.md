# Seven-point Stencil - Julia KernelAbstractions

## Configuration

Select precision in both `cuda/laplacian.jl` and `hip/laplacian.jl`.

```julia
const precision = Float64
const precision_str = "double"
```

Select the backend with `STENCIL_GPU=cuda` for NVIDIA or `STENCIL_GPU=amdgpu` for AMD. Use the matching project below because each project contains only its vendor GPU package.

## Setup

```bash
julia --project=cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=hip -e 'using Pkg; Pkg.instantiate()'
```

## NVIDIA CUDA

Run one

```bash
STENCIL_GPU=cuda julia --project=cuda -O3 --check-bounds=no cuda/laplacian.jl 1024 1024 1024 1024 1 1
```

Run the sweep

```bash
source cuda/run_cuda.sh
```

## AMD ROCm

Run one

```bash
STENCIL_GPU=amdgpu julia --project=hip -O3 --check-bounds=no hip/laplacian.jl 1024 1024 1024 1024 1 1
```

Run the sweep

```bash
source hip/run_hip.sh
```
