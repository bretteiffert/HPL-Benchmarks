# Hartree-Fock - Native CUDA

This benchmark builds a Hartree-Fock Fock matrix from helium-system inputs on an NVIDIA GPU. Run all commands from `cuda/hartree-fock`.

## Prerequisites and build

The build requires the CUDA toolkit (`nvcc`), `nvidia-smi`, and a CUDA-capable host C++ compiler. The makefile detects the compute capability of the first GPU reported by `nvidia-smi` and compiles for that architecture. The default fp64 kernel uses native double-precision `atomicAdd`, which requires compute capability 6.0 or newer.

```sh
make
```

This creates `./hartree-fock-cuda`. Input files are under `../../data/hartree-fock`.

## Precision and configuration

Precision is selected at compile time in `hartree-fock.cu`:

```cpp
#define PRECISION 64
```

Set it to `32` for fp32 and rebuild. `LEGACY_SQRTF` separately controls the legacy fp32 square-root path; its default is `0`, which uses the selected arithmetic type. The kernel uses a fixed 256-thread workgroup.

The first positional argument is the input file. `--iters=N` controls CSV timing iterations and defaults to 10. Without `--csv`, the program reports its checksum and runs an untimed fp64 CPU comparison by default for inputs of at most 128 atoms; `--no-check` disables it and `--check` forces it for larger inputs. CPU checks are not emitted in CSV mode.

## Run one configuration

Run the `he64` input and emit 10 timing rows:

```sh
./hartree-fock-cuda ../../data/hartree-fock/he64 --csv --iters=10
```

## Run the sweep

The supplied sweep runs `he16`, `he32`, `he64`, `he128`, and `he256` in CSV mode:

```sh
bash run_cuda.sh
```

The script passes `--input 10`, but the executable has no `--input` option and silently ignores both arguments. The sweep therefore uses the executable's default of 10 iterations. Use `--iters=10` for an explicit iteration setting in manual commands.

In CSV mode, an untimed launch populates the Fock buffer before timing, and timed launches accumulate into that same buffer rather than clearing it between iterations; the reported rows measure kernel execution time only.
