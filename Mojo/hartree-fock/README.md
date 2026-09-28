## Hartree-Fock - Mojo
Run the benchmark on an NVIDIA CUDA or AMD ROCm/HIP system. The MAX runtime selects the available GPU backend, and inputs are under `../../data/hartree-fock`.

#### Setup
The Pixi environment targets `linux-64`. Install the pinned Mojo and MAX versions from this directory:
```sh
pixi install
```

##### Set configurations
Set `comptime PRECISION = 32` or `64` in `hartree-fock.mojo`; precision is selected at compile time. The kernel uses a fixed 256-thread workgroup.

#### CUDA
Run one:
```sh
pixi run mojo hartree-fock.mojo ../../data/hartree-fock/he64 --csv --iters=10
```

#### ROCm/HIP
Run one:
```sh
pixi run mojo hartree-fock.mojo ../../data/hartree-fock/he64 --csv --iters=10
```

`run_cuda.sh` and `run_hip.sh` are intended to sweep `he16`, `he32`, `he64`, `he128`, and `he256`, but neither works as checked in: each calls `pixi run` without the required `mojo` launcher. Until the scripts are corrected, run the equivalent sweep manually on the appropriate backend:
```sh
for size in he16 he32 he64 he128 he256; do
  pixi run mojo hartree-fock.mojo "../../data/hartree-fock/$size" --csv --iters=10
done
```
