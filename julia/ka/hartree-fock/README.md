# Hartree-Fock - Julia KernelAbstractions

## Configuration

Select precision for both backends in the shared `../../hf_host.jl` source:

```julia
const HF_PRECISION = 64 # Float64
const HF_PRECISION = 32 # Float32
```

Select the backend with `STENCIL_GPU=cuda` for NVIDIA or `STENCIL_GPU=amdgpu` for AMD.
## Setup

```bash
julia --project=cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=hip -e 'using Pkg; Pkg.instantiate()'
```

## NVIDIA CUDA

Run one

```bash
STENCIL_GPU=cuda julia --project=cuda -O3 cuda/hartree-fock.jl ../../../data/hartree-fock/he64 --csv --iters=10
```

Run the sweep

```bash
source cuda/run_cuda.sh
```

## AMD ROCm

Run one

```bash
STENCIL_GPU=amdgpu julia --project=hip -O3 hip/hartree-fock.jl ../../../data/hartree-fock/he64 --csv --iters=10
```

Run the sweep

```bash
source hip/run_hip.sh
```
