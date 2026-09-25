# miniBUDE - Julia KernelAbstractions HIP

This directory runs the miniBUDE molecular-docking benchmark on AMD GPUs through ROCm, AMDGPU.jl, and KernelAbstractions.jl. Run the commands below from this directory.

## Setup

Install the Julia dependencies from the local project:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The commands use the repository's included `bm1` deck at `../../../../data/miniBUDE/bm1`.

## Configuration

Select the AMD GPU backend with `STENCIL_GPU=amdgpu`. The checked-in `miniBUDE.jl` defaults to `Float64`:

```julia
const PRECISION_BITS = 64
```

Set this constant to `32` to use `Float32`; precision is not selected by a command-line option.

## Run one

```bash
STENCIL_GPU=amdgpu julia --project=. -O3 miniBUDE.jl --deck ../../../../data/miniBUDE/bm1 -w 64 -p 64
```

## Run sweep

The supplied script uses workgroup size 64 and sweeps supported poses-per-work-item values from 1 through 128:

```bash
bash ./run_hip.sh
```

## Caveats

- A configuration is skipped unless the deck's pose count is at least `wgsize * ppwi` and is divisible by that product.
- `Float32` runs first compute an in-process `Float64` reference pass for differential checks.
