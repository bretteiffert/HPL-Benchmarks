# BabelStream - Native CUDA

This benchmark runs the BabelStream Copy, Mul, Add, Triad, and Dot kernels on an NVIDIA GPU. Run all commands from `cuda/babelStream`.

## Prerequisites and build

The build requires the CUDA toolkit (`nvcc`), `nvidia-smi`, and a C++11-capable host compiler. The makefile detects the compute capability of the first GPU reported by `nvidia-smi` and compiles for that architecture.

```sh
make
```

This creates `./babelstream_cuda`.

## Precision and configuration

Double precision (fp64) is the default. Pass `--float` to select single precision (fp32). The main runtime options are:

- `-s SIZE` or `--arraysize SIZE`: elements per array; `SIZE` must be a multiple of the fixed 1024-thread workgroup size.
- `-n NUM` or `--numtimes NUM`: repetitions of each kernel; `NUM` must be at least 2 because reporting discards the first sample.
- `--device INDEX`: CUDA device index.
- `--mibibytes`: report binary rather than decimal bandwidth units.
- `--csv`: emit timing rows as CSV.

Use `./babelstream_cuda --help` for usage information and `./babelstream_cuda --list` to list devices.

## Run one configuration

Run fp64 with 33,554,432 elements per array and 1,000 repetitions:

```sh
./babelstream_cuda -s 33554432 -n 1000
```

Add `--float` for fp32.

## Run the sweep

The supplied sweep runs fp64 with 1,000 repetitions at array sizes 65,536, 1,048,576, and 33,554,432:

```sh
bash run_cuda.sh
```
