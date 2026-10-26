# Hartree-Fock - Mojo

## Setup

Install pixi, then the Mojo 1.2.0 / MAX 26.7 toolchain from the project's `pixi.toml` (the environment targets `linux-64`), plus the vendor driver for the selected backend (NVIDIA driver for CUDA, ROCm for HIP), and set the vendor path.

    pixi install
    export CUDA_PATH=[path-to-cuda]
    export HIP_PATH=[path-to-rocm]

Energy is measured through `energy.py`, a Python wrapper around libkernelenergy.so. It loads ../../dependencies/libkernelenergy.so (or `$DSLBENCH_ENERGY_LIB` if that is set), so build that library first, or run with `--no-energy`.

The backend is chosen when Mojo compiles the program, with `-D BACKEND=cuda` or `-D BACKEND=hip`. `mojo run` compiles and runs in one step, so there is no separate build.

## Configuration

Precision is selected at run time with a flag:

    --double   # fp64, the default
    --float    # fp32

There is no `--half`: the 1e-12 screening thresholds are below float16 range.

Other options: `-n` timed iterations (10, after 1 untimed run), `--block-size` (256), `--legacy-sqrt` (Boys-function square roots in float32), `--check` / `--no-check` / `--max-check-atoms N` (the float64 CPU reference runs by default up to 128 atoms), `--device INDEX`, `--list`, `--csv`, `--csv-raw`, `--no-energy`. Optional environment variables `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    pixi run mojo run -D BACKEND=cuda main.mojo ../../data/hartree-fock/he64 -n 10

Run the sweep

    bash run.sh cuda

## AMD ROCm/HIP

Run one

    pixi run mojo run -D BACKEND=hip main.mojo ../../data/hartree-fock/he64 -n 10

Run the sweep

    bash run.sh hip

Extra arguments are passed to every run of the sweep, for example `bash run.sh cuda --float --csv > results.csv`. The sweep runs the CPU reference for he16 to he128; add `--no-check` to skip it.
