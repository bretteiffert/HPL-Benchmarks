# miniBUDE - Julia KernelAbstractions

Run a single configuration or the supplied poses-per-work-item sweep on NVIDIA and AMD GPUs. Run all commands from `julia/ka/miniBUDE`.

## Configuration

Select precision in both `cuda/miniBUDE.jl` and `hip/miniBUDE.jl`:

```julia
const PRECISION_BITS = 64 # Float64
const PRECISION_BITS = 32 # Float32
```

Select the backend with `STENCIL_GPU=cuda` for NVIDIA or `STENCIL_GPU=amdgpu` for AMD. Use the matching project below because each project contains only its vendor GPU package.

## Setup

```bash
julia --project=cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=hip -e 'using Pkg; Pkg.instantiate()'
```

The commands use the included `bm1` deck at `../../../data/miniBUDE/bm1`.

## NVIDIA CUDA

Run one configuration:

```bash
STENCIL_GPU=cuda julia --project=cuda -O3 cuda/miniBUDE.jl --deck ../../../data/miniBUDE/bm1 -w 32 -p 64
```

Run the poses-per-work-item sweep:

```bash
source cuda/run_cuda.sh
```

## AMD ROCm

Run one configuration:

```bash
STENCIL_GPU=amdgpu julia --project=hip -O3 hip/miniBUDE.jl --deck ../../../data/miniBUDE/bm1 -w 64 -p 64
```

Run the poses-per-work-item sweep:

```bash
source hip/run_hip.sh
```
