# Seven-Point Stencil - KernelAbstractions

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
    --half     # fp16, only up to 105 points per side

Other options: `-s L` cubic grid (1024) or `--grid X,Y,Z`, `-n` timed iterations after one untimed run (100), `--block-size X[,Y[,Z]]` (1024,1,1), `--device`, `--list`, `--csv`, `--csv-raw`, `--no-energy`. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver.

Bandwidth counts the theoretical traffic: every point except the 8 corners and 12 edges read once, every interior point written once. Correctness checks ∇²u = 3 at the interior points of the untimed run, on the device.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    julia --project=env/cuda -O3 --check-bounds=no main.jl -s 1024 -n 100

Run the sweep

    bash run.sh cuda

## AMD ROCm/HIP

Run one

    julia --project=env/hip -O3 --check-bounds=no main.jl -s 1024 -n 100

Run the sweep

    bash run.sh hip

Extra arguments are passed to every run of the sweep, for example `bash run.sh cuda --float --csv > results.csv`.
