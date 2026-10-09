# miniBUDE - HIP

## Configuration

Precision is selected at compile time in `main.cpp`:

```cpp
#define PRECISION_BITS 64
```

## Build

The makefile uses `amd-smi` to detect the GPU architecture and requires `hipcc` from ROCm.

```sh
make
```

## Run one

```sh
./bude-hip --deck ../../data/miniBUDE/bm1 -p 64 -w 64
```

## Run the sweep

```sh
source run_hip.sh
```
