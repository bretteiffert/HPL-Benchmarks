# Hartree–Fock - KernelAbstractions

## Setup

```bash
julia --project=env/cuda -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

or
```bash
julia --project=env/hip -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

## Run

The input file is the first argument. Backend selection is per invocation:

```bash
julia --project=env/cuda -O3 --check-bounds=no hartree-fock.jl ../../data/hartree-fock/he128 --backend cuda -n 10
```

or
```bash
julia --project=env/hip -O3 --check-bounds=no hartree-fock.jl ../../data/hartree-fock/he128 --backend hip -n 10
```

Supported options:

```text
--backend cuda|hip
-p 16|32|64
-n NUM
--csv
--check
--no-check
-h
```

The default arithmetic is Float64. Use `-p 32` for Float32 or `-p 16` for the
experimental Float16 path. The default backend is CUDA. The selected runtime
uses its default visible GPU; there is no device-index argument.

## Sweep scripts

```bash
INPUT=../../data/hartree-fock/he128 bash run_cuda.sh
```

or
```bash
INPUT=../../data/hartree-fock/he128 bash run_hip.sh
```

CSV columns:

```text
backend,GPU,precision,sqrt_mode,natoms,ngauss,exec_time_ms
```
