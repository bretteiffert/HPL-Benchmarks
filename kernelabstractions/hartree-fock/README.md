# Hartree-Fock - KernelAbstractions

## Setup

Install Julia 1.13 (the Manifests were resolved with 1.13.1) and the vendor driver for the selected backend. CUDA.jl downloads its CUDA runtime as an artifact; AMDGPU.jl uses the system ROCm and finds it through `ROCM_PATH`.

    export ROCM_PATH=[path-to-rocm]

Each backend has its own environment, so one checkout serves both vendors:

    env/cuda   KernelAbstractions + CUDA.jl + Atomix + SpecialFunctions
    env/hip    KernelAbstractions + AMDGPU.jl + Atomix + SpecialFunctions

The environment you activate selects the backend. Install it once:

    julia --project=env/cuda -e 'using Pkg; Pkg.instantiate()'
    julia --project=env/hip  -e 'using Pkg; Pkg.instantiate()'

Energy is measured through ../../dependencies/energy.jl, which loads libkernelenergy.so from the same folder, so build that library first. Runs with `--no-energy` need neither.

Inputs are read from ../../data/hartree-fock (he16 … he256).

## Configuration

Precision is selected at run time with a flag:

    --double   # fp64, the default
    --float    # fp32

There is no `--half`: the 1e-12 screening thresholds are below the float16 range.

Other options: `-n` timed iterations after one untimed run (10), `--block-size 256` (the pinned value), `--check` / `--no-check` / `--max-check-atoms` (fp64 CPU reference, automatic up to 128 atoms), `--device`, `--list`, `--csv`, `--csv-raw`, `--no-energy`. The pinned configuration uses native-precision square roots. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    julia --project=env/cuda -O3 --check-bounds=no main.jl ../../data/hartree-fock/he128 -n 10

Run the sweep

    bash run.sh cuda

## AMD ROCm/HIP

Run one

    julia --project=env/hip -O3 --check-bounds=no main.jl ../../data/hartree-fock/he128 -n 10

Run the sweep

    bash run.sh hip

Extra arguments are passed to every run of the sweep, for example `bash run.sh cuda --float --csv > results.csv`.
