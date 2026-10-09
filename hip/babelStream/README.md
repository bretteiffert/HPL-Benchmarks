# BabelStream - HIP

## Configuration

Precision is selected at compile time in `main.cpp`

```cpp
use float = false // 64 bit
```


## Build

The makefile uses `amd-smi` to detect the GPU architecture and requires `hipcc` from ROCm.

```sh
make
```

## Run one

```sh
./babelstream_hip -s 33554432 -n 1000
```

## Run the sweep

```sh
source run_hip.sh
```
