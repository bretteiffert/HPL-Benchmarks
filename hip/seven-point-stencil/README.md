# Seven-Point Stencil - Native HIP

## Setup

Install ROCm (HIP 5.x or newer, `hipcc`) and the AMD GPU driver, and set the vendor path.

    export HIP_PATH=[path-to-rocm]

The build links libkernelenergy.so from ../../dependencies, so build that library first.

    make                  # detects the GPU architecture with amdgpu-arch or rocminfo
    make ARCH=gfx90a      # or name it, e.g. on a build node without a GPU

This creates `./laplacian_kernel_hip`. `HIPCC` defaults to `$HIP_PATH/bin/hipcc` (HIP_PATH defaults to /opt/rocm).

## Configuration

Precision is selected at run time with a flag:

    --double   # fp64, the default
    --float    # fp32
    --half     # fp16; up to 105 points per side, where the centre coefficient fits in fp16

Other options: `-s L` for an L × L × L grid (default 1024) or `--grid X,Y,Z`, `--block-size 64,4,1` (the pinned 256-thread shape), `-n NUM` timed runs after one untimed run (default 100), `--device INDEX`, `--list`, `--csv`, `--csv-raw`, `--no-energy`. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver; no environment variables are required.

Bandwidth uses the theoretical traffic: every point is read except the 8 corners and 12 edges, and every interior point is written. The correctness check compares the interior with the exact Laplacian 3.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## AMD ROCm/HIP

Run one

    ./laplacian_kernel_hip -s 1024 --block-size 64,4,1

Build and run the pinned configuration

    bash run.sh

Extra arguments are passed to the run, for example `bash run.sh --float --csv > results.csv`.
