# Hartree-Fock - Julia KernelAbstractions

Run a single input or the supplied size sweep on NVIDIA and AMD GPUs. Run all commands from `julia/ka/hartree-fock`.

## Configuration

Select precision for both backends in the shared `../../hf_host.jl` source:

```julia
const HF_PRECISION = 64 # Float64
const HF_PRECISION = 32 # Float32
```

Select the backend with `STENCIL_GPU=cuda` for NVIDIA or `STENCIL_GPU=amdgpu` for AMD. Use the matching project below because each project contains only its vendor GPU package.

## Setup

```bash
julia --project=cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=hip -e 'using Pkg; Pkg.instantiate()'
```

The benchmark performs a `Float64` startup test of device-side `SpecialFunctions.erf`; this requires a CUDA.jl or AMDGPU.jl version that provides the corresponding SpecialFunctions device override.

## NVIDIA CUDA

Run one input:

```bash
STENCIL_GPU=cuda julia --project=cuda -O3 cuda/hartree-fock.jl ../../../data/hartree-fock/he64 --csv --iters=10
```

Run the input-size sweep:

```bash
source cuda/run_cuda.sh
```

## AMD ROCm

Run one input:

```bash
STENCIL_GPU=amdgpu julia --project=hip -O3 hip/hartree-fock.jl ../../../data/hartree-fock/he64 --csv --iters=10
```

Run the input-size sweep:

```bash
source hip/run_hip.sh
```
