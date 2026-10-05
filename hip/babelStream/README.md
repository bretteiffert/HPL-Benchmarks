## BabelStream - HIP

Run a single customized configuration or a parameter sweep on an AMD GPU with ROCm/HIP.

### Configuration

Set `use_float` in `main.cpp` to `false` for double precision or `true` for single precision. The `--float` option also selects single precision at runtime.

### Build and run

The makefile uses `amd-smi` to detect the GPU architecture and requires `hipcc` from ROCm.

```sh
make
```

Run one configuration:

```sh
./babelstream_hip -s 33554432 -n 1000
```

Run the supplied size sweep:

```sh
source run_hip.sh
```
