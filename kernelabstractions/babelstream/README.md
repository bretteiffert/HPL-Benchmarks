# BabelStream - KernelAbstractions

## Setup

Install Julia 1.13 (the Manifests were resolved with 1.13.1) and the vendor driver for the selected backend. CUDA.jl downloads its CUDA runtime as an artifact; AMDGPU.jl uses the system ROCm and finds it through `ROCM_PATH`.

    export ROCM_PATH=[path-to-rocm]

Each backend has its own environment, so one checkout serves both vendors:

    env/cuda   KernelAbstractions + CUDA.jl
    env/hip    KernelAbstractions + AMDGPU.jl

The environment you activate selects the backend. Install it once:

    julia --project=env/cuda -e 'using Pkg; Pkg.instantiate()'
    julia --project=env/hip  -e 'using Pkg; Pkg.instantiate()'

Energy is measured through ../../dependencies/energy.jl, which loads libkernelenergy.so from the same folder, so build that library first. Runs with `--no-energy` need neither.

## Configuration

Precision is selected at run time with a flag:

    --double   # fp64, the default
    --float    # fp32
    --half     # fp16

Other options: `-s` array size (33554432), `-n` iterations (1000; the first is a warm-up and is excluded from statistics, CSV and energy), `--block-size 256` (the pinned value for every kernel), `--device`, `--list`, `--csv`, `--csv-raw`, `--no-energy`, `--mibibytes`. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    julia --project=env/cuda -O3 --check-bounds=no main.jl -s 33554432 -n 1000

Run the sweep

    bash run.sh cuda

## AMD ROCm/HIP

Run one

    julia --project=env/hip -O3 --check-bounds=no main.jl -s 33554432 -n 1000

Run the sweep

    bash run.sh hip

Extra arguments are passed to every run of the sweep, for example `bash run.sh cuda --float --csv > results.csv`.
