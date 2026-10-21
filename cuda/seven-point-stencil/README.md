# Seven-Point Stencil - Native CUDA

## Setup

Install the CUDA toolkit (`nvcc`, CUDA 11 or newer for C++17), a C++17 host compiler (`g++` by default), and the NVIDIA driver, and set the vendor path.

    export CUDA_PATH=[path-to-cuda]

The build links libkernelenergy.so from ../../dependencies, so build that library first.

    make                  # detects the GPU architecture with nvidia-smi
    make ARCH=90          # or name it, e.g. on a build node without a GPU

This creates `./laplacian_kernel`. The host compiler is pinned to `g++` (`make HOST_CXX=...` to change it), so an exported `CC=nvc` does not leak in.

## Configuration

Precision is selected at run time with a flag:

    --double   # fp64, the default
    --float    # fp32
    --half     # fp16; up to 105 points per side, where the centre coefficient fits in fp16

Other options: `-s L` for an L × L × L grid (default 1024) or `--grid X,Y,Z`, `--block-size X[,Y[,Z]]` (default 1024,1,1), `-n NUM` timed runs after one untimed run (default 100), `--device INDEX`, `--list`, `--csv`, `--csv-raw`, `--no-energy`. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver; no environment variables are required.

Bandwidth uses the theoretical traffic: every point is read except the 8 corners and 12 edges, and every interior point is written. The correctness check compares the interior with the exact Laplacian 3.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    ./laplacian_kernel -s 1024 --block-size 1024,1,1

Run the sweep

    bash run.sh

Extra arguments are passed to every run of the sweep, for example `bash run.sh --float --csv > results.csv`.
