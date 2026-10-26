# Hartree-Fock - Native HIP

## Setup

Install ROCm (HIP 5.x or newer, `hipcc`) and the AMD GPU driver, and set the vendor path.

    export HIP_PATH=[path-to-rocm]

The build links libkernelenergy.so from ../../dependencies, so build that library first. Inputs are read from ../../data/hartree-fock.

    make                  # detects the GPU architecture with amdgpu-arch or rocminfo
    make ARCH=gfx90a      # or name it, e.g. on a build node without a GPU

This creates `./hartree-fock-hip`. `HIPCC` defaults to `$HIP_PATH/bin/hipcc` (HIP_PATH defaults to /opt/rocm).

## Configuration

Precision is selected at run time with a flag:

    --double   # fp64, the default
    --float    # fp32

Other options: `-n NUM` timed builds after one untimed build (default 10), `--block-size 256` (the pinned value), `--device INDEX`, `--list`, `--check` / `--no-check` / `--max-check-atoms N` (the fp64 CPU reference runs automatically up to 128 atoms), `--csv`, `--csv-raw`, `--no-energy`. The pinned configuration uses native-precision square roots. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver; no environment variables are required.

Timed builds add into the same Fock matrix, as in the original benchmark; the 2e- energy checksum and the correctness check use the untimed first build.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## AMD ROCm/HIP

Run one

    ./hartree-fock-hip ../../data/hartree-fock/he64 -n 10

Run the sweep

    bash run.sh

Extra arguments are passed to every run of the sweep, for example `bash run.sh --float --csv > results.csv`.
