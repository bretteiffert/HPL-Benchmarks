## Seven-point Stencil - HIP

Run a single customized configuration or a workgroup sweep on an AMD GPU with ROCm/HIP.

### Configuration

Set the precision in `laplacian.cpp`. Use `using precision = double;` and `precision_str = "double"`, or `using precision = float;` and `precision_str = "single"`.

### Build and run

The makefile uses `amd-smi` to detect the GPU architecture and requires `hipcc` from ROCm.

```sh
make
```

Run one configuration:

```sh
./laplacian_kernel 1024 1024 1024 512 1 1
```

Run the supplied workgroup sweep:

```sh
source run_hip.sh
```

The default `1024^3` grid requires about 16 GiB for the two double-precision arrays, excluding runtime overhead.
