# miniBUDE - CubeCL

## Setup

Install Rust 1.95 or newer, the vendor toolkit and driver required by the selected backend, and set vendor path.

```bash
export CUDA_PATH=[path-to-cuda]
export HIP_PATH=[path-to-rocm]
```

The build links `libkernelenergy.so` from `../../dependencies`, so build that library first. Decks are read from `../../data/miniBUDE`.

If your environment sets `CC=nvc` (NVIDIA HPC SDK), nothing extra is needed: `.cargo/config.toml` builds native dependencies with gcc for this project only.

## Configuration

Precision is selected at run time with a flag:

```bash
--float    # fp32, the default; first computes energies once in fp64 for comparison
--double   # optional fp64 mode
```

The launch configuration is pinned:

```bash
-p, --ppwi 1         # one pose per work-item
-w, --wgsize 256     # 256 work-items per group
```

Other values are rejected, and the pose count must be divisible by 256.

Other options:

```bash
--deck DIR           # input deck (default ../../data/miniBUDE/bm1)
-n, --poses N        # use the first N poses (default 0 = all)
-i, --iter I         # timed iterations per combination (default 8)
--warmup W           # untimed iterations before them (default 2)
-r, --rows N         # pose energies to show (default 8)
-o, --out PATH       # save all pose energies to PATH
-d, --device INDEX   # device to run on (default 0); --list shows them
--csv                # CSV to stdout: one summary row per combination
--csv-raw            # CSV to stdout: one row per combination per timed iteration
--no-energy          # skip energy measurement
```

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

```bash
cargo run --release --no-default-features --features cuda -- --deck ../../data/miniBUDE/bm1 --float -p 1 -w 256
```

Run the canonical benchmark

```bash
bash run.sh cuda
```

## AMD ROCm/HIP

Run one

```bash
cargo run --release --no-default-features --features hip -- --deck ../../data/miniBUDE/bm1 --float -p 1 -w 256
```

Run the canonical benchmark

```bash
bash run.sh hip
```

Extra arguments are passed through, for example `bash run.sh cuda --csv > results.csv`.
