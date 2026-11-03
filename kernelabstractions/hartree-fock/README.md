# Hartree-Fock - KernelAbstractions

## Setup

Install Julia 1.10 or newer (`julia` on the PATH), the vendor toolkit and driver required by the selected backend, and set the vendor path.

    export CUDA_PATH=[path-to-cuda]
    export ROCM_PATH=[path-to-rocm]

CUDA.jl downloads a matching CUDA runtime as a Julia artifact unless told to use the local toolkit; AMDGPU.jl uses the ROCm installation in `ROCM_PATH` (default /opt/rocm). The program loads libkernelenergy.so from ../../dependencies, so build that library first. Inputs are read from ../../data/hartree-fock.

The backend is chosen by the Julia environment the program runs in: `env/cuda` (CUDA.jl) or `env/hip` (AMDGPU.jl), each with KernelAbstractions 0.10, Atomix and SpecialFunctions. Install one once (needs network access):

    julia --project=env/cuda -e 'using Pkg; Pkg.instantiate()'   # or
    julia --project=env/hip  -e 'using Pkg; Pkg.instantiate()'

`run.sh` does this for you. There is nothing to build: Julia compiles the kernel during the untimed first Fock build, so compilation is never in the timings.

## Configuration

Precision is selected at run time with a flag:

    --double   # fp64, the default
    --float    # fp32

Other options: `-n NUM` timed builds after one untimed build (default 30), `--block-size 256` (the pinned value), `--device INDEX`, `--list`, `--check` / `--no-check` / `--max-check-atoms N` (the fp64 CPU reference runs automatically up to 128 atoms), `--csv`, `--csv-raw`, `--no-energy`. The pinned configuration uses native-precision square roots. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver; no environment variables are required.

Timed builds add into the same Fock matrix, as in the original benchmark; the 2e- energy checksum and the correctness check use the untimed first build.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

    julia --project=env/cuda -O3 main.jl ../../data/hartree-fock/he64 -n 30

Run the sweep

    bash run.sh cuda

## AMD ROCm/HIP

Run one

    julia --project=env/hip -O3 main.jl ../../data/hartree-fock/he64 -n 30

Run the sweep

    bash run.sh hip

Extra arguments are passed to every run of the sweep, for example `bash run.sh hip --float --csv > results.csv`.
