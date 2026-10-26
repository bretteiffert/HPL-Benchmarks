# miniBUDE - Native CUDA

## Setup

Install the CUDA toolkit (`nvcc`, CUDA 11 or newer for C++17), a C++17 host compiler (`g++` by default), and the NVIDIA driver, and set the vendor path.

    export CUDA_PATH=[path-to-cuda]

The build links libkernelenergy.so from ../../dependencies, so build that library first. The deck is read from ../../data/miniBUDE/bm1 by default.

    make                  # detects the GPU architecture with nvidia-smi
    make ARCH=90          # or name it, e.g. on a build node without a GPU

This creates `./bude-cuda`. The host compiler is pinned to `g++` (`make HOST_CXX=...` to change it), so an exported `CC=nvc` does not leak in. The original build options still work through `EXTRA_FLAGS`: `-DUSE_SHARED` (forcefield in shared memory), `-DNO_RESTRICT`, `-DMANAGED` or `-DPAGEFAULT` (allocation mode).

## Configuration

Precision is selected at run time with a flag:

    --float    # fp32, the default; one untimed fp64 pass is run as the reference
    --double   # optional fp64 mode

The canonical launch configuration is pinned to `-p 1 -w 256`; other PPWI or work-group values are rejected. Other options keep the upstream names: `--deck DIR`, `-i` timed iterations (default 8), `--warmup N` (default 2), `-n` poses (0 for all), `-d` device index or name substring, `-o` output energies, and `-r` energies shown in the report. Also `--list`, `--csv`, `--csv-raw`, `--no-energy`. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    ./bude-cuda --deck ../../data/miniBUDE/bm1 --float -p 1 -w 256

Run the canonical benchmark

    bash run.sh

Extra arguments are passed through, for example `bash run.sh --csv > results.csv`.
