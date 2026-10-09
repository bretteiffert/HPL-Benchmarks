# Seven-point Stencil - HIP

## Configuration

Precision is selected at compile time in `laplacian.cpp`

```cpp
using precision = double;
char precision_str[] = "double";
```

## Build and run

The makefile uses `amd-smi` to detect the GPU architecture and requires `hipcc` from ROCm.

```sh
make
```

## Run one

```sh
./laplacian_kernel 1024 1024 1024 512 1 1
```

## Run the sweep

```sh
source run_hip.sh
```
