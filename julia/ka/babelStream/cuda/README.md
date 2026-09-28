# BabelStream - Julia KernelAbstractions CUDA

This directory runs the BabelStream memory-bandwidth kernels on NVIDIA GPUs through CUDA.jl and KernelAbstractions.jl. Run the commands below from this directory.

## Setup

Install the Julia dependencies from the local project:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Configuration

Select the CUDA backend with `STENCIL_GPU=cuda`. The checked-in `babelstream.jl` uses `Float64` because it sets:

```julia
const USE_FLOAT = false
```

Set this constant to `true` to use `Float32`.

## Run one

```bash
STENCIL_GPU=cuda julia --project=. -O3 --check-bounds=no babelstream.jl -s 33554432 -n 1000
```

## Run sweep

The supplied script runs array sizes 65536, 1048576, and 33554432 with 1000 iterations each:

```bash
bash ./run_cuda.sh
```

## Caveats

- Array sizes must be multiples of 1024.
- The `--float` option is accepted but does not change precision; edit `USE_FLOAT` instead.
