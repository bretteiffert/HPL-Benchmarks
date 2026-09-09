# Seven-point Stencil - Julia KernelAbstractions

Run a single configuration or the supplied workgroup sweep on NVIDIA and AMD GPUs. Run all commands from `julia/ka/seven-point-stencil`.

## Configuration

Select precision in both `cuda/laplacian.jl` and `hip/laplacian.jl`, keeping the type and label consistent:

```julia
const precision = Float64
const precision_str = "double"
```

or:

```julia
const precision = Float32
const precision_str = "float"
```

The checked-in CUDA launcher uses `Float32`, while the HIP launcher uses `Float64`; align them before comparing backends. A 1024 x 1024 x 1024 run allocates two full GPU arrays, requiring 8 GiB for `Float32` or 16 GiB for `Float64`, excluding runtime overhead.

Select the backend with `STENCIL_GPU=cuda` for NVIDIA or `STENCIL_GPU=amdgpu` for AMD. Use the matching project below because each project contains only its vendor GPU package.

## Setup

```bash
julia --project=cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=hip -e 'using Pkg; Pkg.instantiate()'
```

## NVIDIA CUDA

Run one configuration:

```bash
STENCIL_GPU=cuda julia --project=cuda -O3 --check-bounds=no cuda/laplacian.jl 1024 1024 1024 1024 1 1
```

Run the workgroup sweep:

```bash
source cuda/run_cuda.sh
```

## AMD ROCm

Run one configuration:

```bash
STENCIL_GPU=amdgpu julia --project=hip -O3 --check-bounds=no hip/laplacian.jl 1024 1024 1024 1024 1 1
```

Run the workgroup sweep:

```bash
source hip/run_hip.sh
```
