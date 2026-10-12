# Seven-Point Stencil - KernelAbstractions

## Setup

From this directory, instantiate the environment for the GPU being used:

```bash
julia --project=env/cuda -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
julia --project=env/hip -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

## Run

The benchmark uses the default GPU visible to the selected vendor runtime.
There is no device-index argument and no `STENCIL_GPU` environment variable.

```bash
julia --project=env/cuda -O3 --check-bounds=no seven-point-stencil.jl \
  --backend cuda -s 256 -n 100 -p 32

julia --project=env/hip -O3 --check-bounds=no seven-point-stencil.jl \
  --backend hip -s 256 -n 100 -p 32
```

Options:

```text
--backend cuda|hip
-s SIZE
-n NUM
-p 16|32|64
--csv
-h, --help
```

`-s` sets all three grid dimensions (`SIZE × SIZE × SIZE`). `-n` sets the
number of timed stencil iterations. Precision is selected at runtime; Float16
is experimental and depends on backend and device support.

## Run scripts

Each script runs a small size sweep. Override the defaults with environment
variables:

```bash
SIZE=512 NUM_TIMES=100 PRECISION=64 bash run_cuda.sh
SIZE=512 NUM_TIMES=100 PRECISION=32 bash run_hip.sh
```

## CSV output

Add `--csv` to write performance rows to stdout with these columns:

```text
backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs
```

Correctness diagnostics are written to stderr:

```bash
julia --project=env/cuda seven-point-stencil.jl \
  --backend cuda -s 256 -n 10 --csv > results.csv
```
