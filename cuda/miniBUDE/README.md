# miniBUDE - Native CUDA

This benchmark evaluates the miniBUDE molecular docking energy kernel on an NVIDIA GPU. Run all commands from `cuda/miniBUDE`.

## Prerequisites and build

The build requires the CUDA toolkit (`nvcc`), `nvidia-smi`, and a C++17-capable host compiler. The makefile detects the compute capability of the first GPU reported by `nvidia-smi` and compiles for that architecture.

```sh
make
```

This creates `./bude-cuda`. The supplied commands use the `bm1` deck at `../../data/miniBUDE/bm1`.

## Precision and configuration

Precision is selected at compile time in `main.cpp`:

```cpp
#define PRECISION_BITS 64
```

Set it to `32` for fp32 and rebuild. The `-p` option selects poses per work-item (PPWI), not precision. Supported PPWI values are `1,2,4,8,16,32,64,128`.

Use `-w` or `--wgsize` for the CUDA workgroup size, `-i` or `--iter` for measured iterations, and `--deck` for the input directory. The pose count must be at least and divisible by `PPWI * wgsize`; invalid combinations are skipped. An fp32 run also computes one fp64 differential reference pass.

Use `./bude-cuda --help` for all options and `./bude-cuda --list` to list devices.

## Run one configuration

Run the `bm1` deck with PPWI 64 and a 32-thread workgroup:

```sh
./bude-cuda --deck ../../data/miniBUDE/bm1 -w 32 -p 64
```

## Run the sweep

The supplied sweep uses the `bm1` deck and a 32-thread workgroup while testing PPWI values `1, 2, 4, 8, 16, 32, 64, and 128`:

```sh
bash run_cuda.sh
```
