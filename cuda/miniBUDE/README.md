# miniBUDE - Native CUDA

## Prerequisites and build

The build requires the CUDA toolkit (`nvcc`) and a C++ host compiler. The makefile detects the GPU and compiles for that architecture.
```sh
make
```

## Precision and configuration

Precision is selected at compile time in `main.cpp`:

```cpp
#define PRECISION_BITS 64
```

## Run one

```sh
./bude-cuda --deck ../../data/miniBUDE/bm1 -w 32 -p 64
```

## Run the sweep

```sh
bash run_cuda.sh
```
