# miniBUDE - KernelAbstractions

## Setup

Install Julia 1.13 (the Manifests were resolved with 1.13.1) and the vendor driver for the selected backend. CUDA.jl downloads its CUDA runtime as an artifact; AMDGPU.jl uses the system ROCm and finds it through `ROCM_PATH`.

    export ROCM_PATH=[path-to-rocm]

Each backend has its own environment, so one checkout serves both vendors:

    env/cuda   KernelAbstractions + CUDA.jl + SpecialFunctions
    env/hip    KernelAbstractions + AMDGPU.jl + SpecialFunctions

The environment you activate selects the backend. Install it once:

    julia --project=env/cuda -e 'using Pkg; Pkg.instantiate()'
    julia --project=env/hip  -e 'using Pkg; Pkg.instantiate()'

Energy is measured through ../../dependencies/energy.jl, which loads libkernelenergy.so from the same folder, so build that library first. Runs with `--no-energy` need neither.

The default deck is ../../data/miniBUDE/bm1; `--deck DIR` selects another.

## Configuration

Precision is selected at run time with a flag:

    --float    # fp32, the default; one untimed fp64 pass is run as the reference
    --double   # optional fp64 mode

The launch is pinned to `-p 1 -w 256`; other values are rejected. Other options include `-i` timed iterations (8), `--warmup` (2), `-n` poses (all), `--deck`, `-o`, `-r`, `-d`/`--device`, `--list`, `--csv`, `--csv-raw`, and `--no-energy`.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    julia --project=env/cuda -O3 --check-bounds=no main.jl --float -p 1 -w 256

Run the canonical benchmark

    bash run.sh cuda

## AMD ROCm/HIP

Run one

    julia --project=env/hip -O3 --check-bounds=no main.jl --float -p 1 -w 256

Run the canonical benchmark

    bash run.sh hip

Extra arguments are passed through, for example `bash run.sh cuda --csv > results.csv`.
