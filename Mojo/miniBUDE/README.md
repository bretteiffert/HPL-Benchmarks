# miniBUDE - Mojo

## Setup

Install pixi, then the Mojo 1.2.0 / MAX 26.7 toolchain from the project's `pixi.toml`, plus the vendor driver for the selected backend (NVIDIA driver for CUDA, ROCm for HIP), and set the vendor path.

    pixi install
    export CUDA_PATH=[path-to-cuda]
    export HIP_PATH=[path-to-rocm]

Energy is measured through `energy.py`, a Python wrapper around libkernelenergy.so. It loads ../../dependencies/libkernelenergy.so (or `$DSLBENCH_ENERGY_LIB` if that is set), so build that library first, or run with `--no-energy`.

The backend is chosen when Mojo compiles the program, with `-D BACKEND=cuda` or `-D BACKEND=hip`. `mojo run` compiles and runs in one step, so there is no separate build.

## Configuration

Precision is selected at run time with a flag:

    --float    # fp32, the default in this port
    --double   # fp64, see below

`--double` exists but does not work with Mojo 1.2: the kernel uses float64 `sin`/`cos`, which Mojo 1.2 rejects on NVIDIA at compile time and cannot lower on AMD. The float64 path is therefore only compiled when you add `-D FP64` (`mojo run -D BACKEND=cuda -D FP64 ...`); with it, float32 runs also get a float64 reference pass on the device. Without it, the float64 reference for the accuracy metrics comes from `--golden FILE` (for example the CUDA build's float64 `-o` output).

The launch is pinned to `-p 1 -w 256`; other values are rejected. Other options include `-i` timed launches (8), `--warmup` (2), `-n` poses (all), `--deck DIR` (../../data/miniBUDE/bm1), `-r` energies to print (8), `-o FILE`, `--golden FILE`, `--device INDEX`, `--list`, `--csv`, `--csv-raw`, and `--no-energy`.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    pixi run mojo run -D BACKEND=cuda main.mojo --deck ../../data/miniBUDE/bm1 --float -w 256 -p 1

Run the canonical benchmark

    bash run.sh cuda

## AMD ROCm/HIP

Run one

    pixi run mojo run -D BACKEND=hip main.mojo --deck ../../data/miniBUDE/bm1 --float -w 256 -p 1

Run the canonical benchmark

    bash run.sh hip

Extra arguments are passed through, for example `bash run.sh cuda --csv > results.csv`.
