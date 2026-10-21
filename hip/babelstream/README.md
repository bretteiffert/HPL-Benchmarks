<<<<<<< HEAD
# BabelStream - Native HIP

## Setup

Install ROCm (HIP 5.x or newer, `hipcc`) and the AMD GPU driver, and set the vendor path.

    export HIP_PATH=[path-to-rocm]

The build links libkernelenergy.so from ../../dependencies, so build that library first.

    make                  # detects the GPU architecture with amdgpu-arch or rocminfo
    make ARCH=gfx90a      # or name it, e.g. on a build node without a GPU

This creates `./babelstream_hip`. `HIPCC` defaults to `$HIP_PATH/bin/hipcc` (HIP_PATH defaults to /opt/rocm).

## Configuration

Precision is selected at run time with a flag:

    --double   # fp64, the default
    --float    # fp32
    --half     # fp16; values decay below the fp16 range after a few hundred iterations, see the correctness notes

Other options: `-s SIZE` elements per array (default 33,554,432), `-n NUM` iterations (default 1000, the first is warm-up), `--block-size N` (default 1024, a power of two), `--device INDEX`, `--list`, `--csv`, `--csv-raw`, `--no-energy`, `--mibibytes` (text report only). `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver; no environment variables are required.

`FMAD` is a build-time switch (`make EXTRA_FLAGS=-DFMAD=0`). hipcc contracts `a*b+c` into an FMA by default, so it is normally a no-op.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## AMD ROCm/HIP

Run one

    ./babelstream_hip -s 33554432 -n 1000

Run the sweep

    bash run.sh

Extra arguments are passed to every run of the sweep, for example `bash run.sh --float --csv > results.csv`.
=======
# BabelStream - HIP

## Configuration

Precision is selected at compile time in `main.cpp`

```cpp
use float = false // 64 bit
```


## Build

The makefile uses `amd-smi` to detect the GPU architecture and requires `hipcc` from ROCm.

```sh
make
```

## Run one

```sh
./babelstream_hip -s 33554432 -n 1000
```

## Run the sweep

```sh
source run_hip.sh
```
>>>>>>> main
