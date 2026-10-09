# BabelStream - Native CUDA

## Prerequisites and build

The build requires the CUDA toolkit (`nvcc`) and a C++ host compiler. The makefile detects the GPU and compiles for that architecture.

```sh
make
```

This creates `./babelstream_cuda`.

## Precision and configuration

Precision is selected at compile time in `main.cpp`:

```cpp
use float = false // 64 bit
```

## Run one

```sh
./babelstream_cuda -s 33554432 -n 1000
```

## Run the sweep

```sh
bash run_cuda.sh
```
