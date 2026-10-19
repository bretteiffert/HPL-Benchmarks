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
--double   # fp64, the default
--float    # fp32; first computes the energies once in fp64 and reports the difference
```

One run sweeps every combination of poses per work-item and work-group size:

```bash
-p, --ppwi LIST      # e.g. 1,4,16 or all (1 to 128); default 1
-w, --wgsize LIST    # e.g. 32,64; default 1
```

Combinations the pose count doesn't divide are skipped with a note.

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
cargo run --release --no-default-features --features cuda -- --deck ../../data/miniBUDE/bm1 -p 64 -w 32
```

Run the sweep

```bash
bash run.sh cuda
```

## AMD ROCm/HIP

Run one

```bash
cargo run --release --no-default-features --features hip -- --deck ../../data/miniBUDE/bm1 -p 64 -w 64
```

Run the sweep

```bash
bash run.sh hip
```

The sweep runs every ppwi at work-group size 32 (CUDA) or 64 (HIP). Extra arguments are passed through, for example `bash run.sh cuda --float -w 32,64,128 --csv > results.csv`.
