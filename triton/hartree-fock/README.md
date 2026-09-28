## Hartree-Fock - Triton
Run a single input or the provided CUDA and HIP sweeps. PyTorch selects the backend from the active Pixi environment; ROCm also uses the `torch.cuda` API. Run the commands below from `triton/hartree-fock`.

#### Setup
The included `pixi.toml` targets `linux-64` with Python 3.12 and provides separate environments for the official PyTorch 2.13.0 CUDA 13.0 and ROCm 7.1 wheels. Both environments use the matching Triton 3.7.1 runtime.

Install the environment for the backend on the current machine:
```sh
pixi install -e cuda
# or
pixi install -e hip
```

##### Set configurations
Set `PRECISION = 32` or `PRECISION = 64` near the top of `hartree-fock.py`; fp64 is the default. `LEGACY_SQRTF` separately controls legacy fp32 square-root behavior.

The source resolves Triton's version-dependent device `pow` and `erf` APIs from libdevice, `tl.math`, or `tl`; an import error means the resolver must be updated for the installed Triton version. CPU correctness checks run by default only through 128 atoms and are omitted in `--csv` mode.

#### CUDA
Run one:
```sh
pixi run -e cuda python hartree-fock.py ../../data/hartree-fock/he64 --csv --iters=10
```

Run the input-size sweep:
```sh
bash run_cuda.sh
```

#### HIP
Run one:
```sh
pixi run -e hip python hartree-fock.py ../../data/hartree-fock/he64 --csv --iters=10
```

Run the input-size sweep:
```sh
bash run_hip.sh
```
