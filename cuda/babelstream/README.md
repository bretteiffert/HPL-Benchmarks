# BabelStream - Native CUDA

## Setup

Install the CUDA toolkit (`nvcc`, CUDA 11 or newer for C++17), a C++17 host compiler (`g++` by default), and the NVIDIA driver, and set the vendor path.

    export CUDA_PATH=[path-to-cuda]

The build links libkernelenergy.so from ../../dependencies, so build that library first.

    make                  # detects the GPU architecture with nvidia-smi
    make ARCH=90          # or name it, e.g. on a build node without a GPU

This creates `./babelstream_cuda`. The host compiler is pinned to `g++` (`make HOST_CXX=...` to change it), so an exported `CC=nvc` does not leak in.

## Configuration

Precision is selected at run time with a flag:

    --double   # fp64, the default
    --float    # fp32
    --half     # fp16; values decay below the fp16 range after a few hundred iterations, see the correctness notes

Other options: `-s SIZE` elements per array (default 33,554,432), `-n NUM` iterations (default 1000, the first is warm-up), `--block-size N` (default 1024, a power of two), `--device INDEX`, `--list`, `--csv`, `--csv-raw`, `--no-energy`, `--mibibytes` (text report only). `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver; no environment variables are required.

`FMAD` is a build-time switch (`make EXTRA_FLAGS=-DFMAD=0`). nvcc contracts `a*b+c` into an FMA by default, so it is normally a no-op.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    ./babelstream_cuda -s 33554432 -n 1000

Run the sweep

    bash run.sh

Extra arguments are passed to every run of the sweep, for example `bash run.sh --float --csv > results.csv`.
