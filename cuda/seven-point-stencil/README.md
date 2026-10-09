# Seven-Point Stencil - Native CUDA

## Prerequisites and build

The build requires the CUDA toolkit (`nvcc`) and a C++ host compiler. The makefile detects the GPU and compiles for that architecture.

```sh
make
```

## Precision and configuration

Precision is selected at compile time in `laplacian.cu`

```cpp
using precision = double;
char precision_str[] = "double";
```

## Run one configuration

```sh
./laplacian_kernel 1024 1024 1024 1024 1 1
```

## Run the sweep

```sh
bash run_cuda.sh
```
