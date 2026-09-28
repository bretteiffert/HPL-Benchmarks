# Hartree-Fock - Julia KernelAbstractions HIP

This directory runs the two-electron Hartree-Fock Fock-build benchmark on AMD GPUs through ROCm, AMDGPU.jl, and KernelAbstractions.jl. Run the commands below from this directory.

## Setup

Install the Julia dependencies from the local project:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The commands read inputs from `../../../../data/hartree-fock`.

## Configuration

Select the AMD GPU backend with `STENCIL_GPU=amdgpu`. Precision is shared with the other Julia Hartree-Fock implementations and defaults to `Float64` in `../../../hf_host.jl`:

```julia
const HF_PRECISION = 64
```

Set this constant to `32` to use `Float32`.

## Run one

```bash
STENCIL_GPU=amdgpu julia --project=. -O3 hartree-fock.jl ../../../../data/hartree-fock/he64 --csv --iters=10
```

## Run sweep

The supplied script runs `he16`, `he32`, `he64`, `he128`, and `he256` with 10 timed iterations each:

```bash
bash ./run_hip.sh
```

## Caveats

- Startup runs a device-side `Float64` `SpecialFunctions.erf` self-test. AMDGPU.jl and SpecialFunctions.jl must provide the corresponding device override.
- The kernel uses 256-thread workgroups and atomic updates to the Fock matrix.
