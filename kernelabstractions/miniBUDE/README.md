# miniBUDE - KernelAbstractions

## Setup

From this directory, generate each environment's manifest:

```bash
julia --project=env/cuda -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
julia --project=env/hip -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

The default deck is `../../data/miniBUDE/bm1` relative to this benchmark.
Override it with `--deck` when needed.

## Run

```bash
julia --project=env/cuda -O3 --check-bounds=no miniBUDE.jl \
  --backend cuda -n 8 -p 64

julia --project=env/hip -O3 --check-bounds=no miniBUDE.jl \
  --backend hip -n 8 -p 64
```

Options:

```text
--backend cuda|hip
-n NUM                 Timed kernel iterations
-p 16|32|64            Precision
--poses NUM             Number of poses; 0 uses all deck poses
--ppwi LIST|all         Poses per work-item
--wgsize LIST           Work-group sizes
--deck DIR              Input deck directory
--out PATH              Save energies
--rows NUM              Print energy rows
--csv
-h
```

Float32 and Float64 are normal modes. Float16 is experimental and depends on
device support.

## Sweep scripts

```bash
DECK=../../data/miniBUDE/bm1 bash run_cuda.sh
DECK=../../data/miniBUDE/bm1 bash run_hip.sh
```

Optional overrides:

```bash
DECK=../../data/miniBUDE/bm1 PRECISION=32 NUM_TIMES=20 bash run_cuda.sh
```

CSV output contains precision, PPWI, work-group, timing, throughput, and
correctness columns.
