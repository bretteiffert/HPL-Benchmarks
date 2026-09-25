## Hartree-Fock - HIP

Run a single customized configuration or an input-size sweep on an AMD GPU with ROCm/HIP.

### Configuration

Set `PRECISION` in `hartree-fock.cc` to `64` or `32`.

### Build and run

The makefile uses `amd-smi` to detect the GPU architecture and requires `hipcc` from ROCm.

```sh
make
```

Run one configuration:

```sh
./hartree-fock-hip ../../data/hartree-fock/he64 --csv --iters=10
```

Run the supplied input-size sweep:

```sh
source run_hip.sh
```

The sweep currently passes an unrecognized `--input` option; the executable ignores it and uses its default of 10 iterations.
