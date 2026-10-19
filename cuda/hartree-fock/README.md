# Hartree-Fock - Native CUDA

## Setup

Install the CUDA toolkit (`nvcc`, CUDA 11 or newer for C++17), a C++17 host compiler (`g++` by default), and the NVIDIA driver, and set the vendor path.

    export CUDA_PATH=[path-to-cuda]

The build links libkernelenergy.so from ../../dependencies, so build that library first. Inputs are read from ../../data/hartree-fock.

    make                  # detects the GPU architecture with nvidia-smi
    make ARCH=90          # or name it, e.g. on a build node without a GPU

This creates `./hartree-fock-cuda`. The host compiler is pinned to `g++` (`make HOST_CXX=...` to change it), so an exported `CC=nvc` does not leak in. `--double` needs `atomicAdd` on double, so the GPU must be compute capability 6.0 or newer.

## Configuration

Precision is selected at run time with a flag:

    --double   # fp64, the default
    --float    # fp32

Other options: `-n NUM` timed builds after one untimed build (default 10), `--legacy-sqrt` (Boys-function square roots in fp32 whatever the precision, as the old mixed-precision build did), `--block-size N` (default 256), `--device INDEX`, `--list`, `--check` / `--no-check` / `--max-check-atoms N` (the fp64 CPU reference runs automatically up to 128 atoms), `--csv`, `--csv-raw`, `--no-energy`. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver; no environment variables are required.

Timed builds add into the same Fock matrix, as in the original benchmark; the 2e- energy checksum and the correctness check use the untimed first build.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    ./hartree-fock-cuda ../../data/hartree-fock/he64 -n 10

Run the sweep

    bash run.sh

Extra arguments are passed to every run of the sweep, for example `bash run.sh --float --csv > results.csv`.
