# BabelStream - KernelAbstractions

## Setup

```bash
julia --project=env/hip -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

## Run

Select the backend per invocation:

```bash
julia --project=env/cuda -O3 --check-bounds=no babelstream.jl \
  --backend cuda -s 33554432 -n 1000 -p 64

julia --project=env/hip -O3 --check-bounds=no babelstream.jl \
  --backend hip -s 33554432 -n 1000 -p 64
```

Supported options:

```text
--backend cuda|hip
-s SIZE
-n NUM
-p 16|32|64
--csv
-h
```

Defaults are `cuda`, `33554432` elements, `1000` iterations, and Float64.
Precision is selected at runtime.

## CUDA sweep

```bash
bash run_cuda.sh
```

Override precision or iteration count when needed:

```bash
PRECISION=64 NUM_TIMES=100 bash run_cuda.sh
```

## HIP sweep

```bash
bash run_hip.sh
```

Override precision or iteration count when needed:

```bash
PRECISION=32 NUM_TIMES=100 bash run_hip.sh
```

## CSV output

Add `--csv` to write performance rows to stdout with these columns:

```text
backend,GPU,precision,vec_size,routine,BW_GBs
```

Correctness diagnostics are written to stderr:

```bash
julia --project=. babelstream.jl --backend cuda -s 1048576 -n 10 --csv
```
