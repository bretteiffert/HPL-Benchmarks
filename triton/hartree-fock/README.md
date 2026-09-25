## Hartree-Fock - Triton
Run a single input or the provided CUDA and HIP sweeps. These commands assume existing `triton-cuda` and `triton-hip` Mamba environments with the corresponding PyTorch build and Triton installed. PyTorch selects the backend from the active environment; ROCm also uses the `torch.cuda` API.

##### Set configurations
Set `PRECISION = 32` or `PRECISION = 64` near the top of `hartree-fock.py`; fp64 is the default. `LEGACY_SQRTF` separately controls legacy fp32 square-root behavior.

The source resolves Triton's version-dependent device `pow` and `erf` APIs from libdevice, `tl.math`, or `tl`; an import error means the resolver must be updated for the installed Triton version. CPU correctness checks run by default only through 128 atoms and are omitted in `--csv` mode.

#### CUDA
Run one:
```sh
mamba run -n triton-cuda python hartree-fock.py ../../data/hartree-fock/he64 --csv --iters=10
```

Run the input-size sweep:
```sh
bash run_cuda.sh
```

#### HIP
Run one:
```sh
mamba run -n triton-hip python hartree-fock.py ../../data/hartree-fock/he64 --csv --iters=10
```

Run the input-size sweep:
```sh
bash run_hip.sh
```
