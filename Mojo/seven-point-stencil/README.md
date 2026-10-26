# Seven-point Stencil - Mojo

## Setup

Install pixi, then the Mojo 1.2.0 / MAX 26.7 toolchain from the project's `pixi.toml`, plus the vendor driver for the selected backend (NVIDIA driver for CUDA, ROCm for HIP), and set the vendor path.

    pixi install
    export CUDA_PATH=[path-to-cuda]
    export HIP_PATH=[path-to-rocm]

Energy is measured through `energy.py`, a Python wrapper around libkernelenergy.so. It loads ../../dependencies/libkernelenergy.so (or `$DSLBENCH_ENERGY_LIB` if that is set), so build that library first, or run with `--no-energy`.

The backend is chosen when Mojo compiles the program, with `-D BACKEND=cuda` or `-D BACKEND=hip`. `mojo run` compiles and runs in one step, so there is no separate build.

## Configuration

Precision is selected at run time with a flag:

    --double   # fp64, the default
    --float    # fp32
    --half     # fp16, only for small grids (up to 105 points per side for a cube)

Other options: `-s L` cubic grid (1024) or `--grid X,Y,Z`, `--block-size 64,4,1` (the pinned 256-thread shape), `-n` timed iterations (100, after 1 untimed run), `--device INDEX`, `--list`, `--mibibytes`, `--csv`, `--csv-raw`, `--no-energy`. Optional environment variables `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    pixi run mojo run -D BACKEND=cuda main.mojo -s 1024 --block-size 64,4,1

Run the pinned configuration

    bash run.sh cuda

## AMD ROCm/HIP

Run one

    pixi run mojo run -D BACKEND=hip main.mojo -s 1024 --block-size 64,4,1

Run the pinned configuration

    bash run.sh hip

Extra arguments are passed to the run, for example `bash run.sh cuda --float --csv > results.csv`.
