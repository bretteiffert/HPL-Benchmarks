# Seven-point Stencil - CubeCL

## Setup

Install Rust, the vendor toolkit and driver required by the selected backend, and set vendor path.

```bash
export CUDA_PATH=[path-to-cuda]
export HIP_PATH=[path-to-rocm]
```

The build links `libkernelenergy.so` from `../../dependencies`, so build that library first.

If your environment sets `CC=nvc` (NVIDIA HPC SDK), nothing extra is needed: `.cargo/config.toml` builds native dependencies with gcc for this project only.

## Configuration

Precision is selected at run time with a flag:

```bash
--double   # fp64, the default
--float    # fp32
--half     # fp16, grids up to 105 per side only: larger grids overflow fp16's range
```

Other options:

```bash
-s, --size L             # L x L x L grid (default 1024)
--grid X,Y,Z             # non-cubic grid instead
--block-size 64,4,1       # pinned 256-unit cube shape for the Laplacian
-n, --numtimes NUM       # timed iterations (default 100), after one untimed warm-up run
--device INDEX           # device to run on (default 0); --list shows them
--csv                    # CSV to stdout: one summary row
--csv-raw                # CSV to stdout: one row per timed iteration
--no-energy              # skip energy measurement
--mibibytes              # report MiB instead of MB (CSV stays GB/s)
```

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

```bash
cargo run --release --no-default-features --features cuda -- -s 1024 --block-size 64,4,1
```

Run the pinned configuration

```bash
bash run.sh cuda
```

## AMD ROCm/HIP

Run one

```bash
cargo run --release --no-default-features --features hip -- -s 1024 --block-size 64,4,1
```

Run the pinned configuration

```bash
bash run.sh hip
```

Extra arguments are passed to the run, for example `bash run.sh cuda --float --csv > results.csv`.
