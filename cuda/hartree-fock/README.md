# Hartree-Fock - Native CUDA

## Prerequisites and build

The build requires the CUDA toolkit (`nvcc`) and a C++ host compiler. The makefile detects the GPU and compiles for that architecture.

```sh
make
```

## Precision and configuration

Precision is selected at compile time in `hartree-fock.cu`:

```cpp
#define PRECISION 64
```

## Run one configuration

```sh
./hartree-fock-cuda ../../data/hartree-fock/he64 --csv --iters=10
```

## Run the sweep

```sh
bash run_cuda.sh
```
