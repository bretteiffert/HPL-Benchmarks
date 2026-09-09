## miniBUDE - HIP

Run a single customized configuration or a poses-per-work-item sweep on an AMD GPU with ROCm/HIP.

### Configuration

Set `PRECISION_BITS` in `main.cpp` to `64` or `32`. The included sweep uses the `bm1` deck, workgroups of 64 threads, and PPWI values from 1 through 128.

### Build and run

The makefile uses `amd-smi` to detect the GPU architecture and requires `hipcc` from ROCm.

```sh
make
```

Run one configuration:

```sh
./bude-hip --deck ../../data/miniBUDE/bm1 -p 64 -w 64
```

Run the supplied PPWI sweep:

```sh
source run_hip.sh
```
