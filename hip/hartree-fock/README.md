# Hartree-Fock - HIP

## Configuration

Precision is selected at compile time in `hartree-fock.cc`:

```cpp
#define PRECISION 64
```

## Build

The makefile uses `amd-smi` to detect the GPU architecture and requires `hipcc` from ROCm.

```sh
make
```

## Run one

```sh
./hartree-fock-hip ../../data/hartree-fock/he64 --csv --iters=10
```

## Run the sweep

```sh
source run_hip.sh
```
