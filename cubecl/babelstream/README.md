# BabelStream - CubeCL

## Setup

<<<<<<< HEAD
Install Rust 1.95 or newer, the vendor toolkit and driver required by the selected backend, and set vendor path.
=======
Install Rust, the vendor toolkit and driver required by the selected backend, and set vendor path.
>>>>>>> main

```bash
export CUDA_PATH=[path-to-cuda]
export HIP_PATH=[path-to-rocm]
```

<<<<<<< HEAD
The build links `libkernelenergy.so` from `../../dependencies`, so build that library first.

If your environment sets `CC=nvc` (NVIDIA HPC SDK), nothing extra is needed: `.cargo/config.toml` builds native dependencies with gcc for this project only.

## Configuration

Precision is selected at run time with a flag:

```bash
--double   # fp64, the default
--float    # fp32
--half     # fp16
```

Other options:

```bash
-s, --arraysize SIZE   # elements per array (default 33554432), a multiple of the block size
-n, --numtimes NUM     # iterations (default 1000), the first is excluded from statistics
--block-size SIZE      # units per cube for every kernel (default 1024)
--device INDEX         # device to run on (default 0); --list shows them
--csv                  # CSV to stdout: one summary row per routine
--csv-raw              # CSV to stdout: one row per routine per measured iteration
--no-energy            # skip energy measurement
--mibibytes            # report MiB instead of MB (CSV stays GB/s)
```

The SM/CU count for the Dot kernel (four workgroups per SM/CU) is read from the device. In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

=======
## Configuration

Precision is selected at compile time in `src/main.rs`:

```rust
const USE_FLOAT: bool = false; // fp64, the default
const USE_FLOAT: bool = true;  // fp32
```

Set `BS_SM_COUNT` to the NVIDIA streaming-multiprocessor count or AMD compute-unit count. The Dot kernel launches four workgroups per SM/CU; an individual run warns and falls back to 64 when the variable is absent, while both sweep scripts require it.

```bash
export BS_SM_COUNT=132
```

>>>>>>> main
## NVIDIA CUDA

Run one

```bash
<<<<<<< HEAD
cargo run --release --no-default-features --features cuda -- -s 33554432 -n 1000
=======
BS_SM_COUNT="$BS_SM_COUNT" cargo run --release --no-default-features --features cuda -- -s 33554432 -n 1000
>>>>>>> main
```

Run the array-size sweep

```bash
<<<<<<< HEAD
bash run.sh cuda
=======
bash run_cuda.sh
>>>>>>> main
```

## AMD ROCm/HIP

Run one

```bash
<<<<<<< HEAD
cargo run --release --no-default-features --features hip -- -s 33554432 -n 1000
=======
BS_SM_COUNT="$BS_SM_COUNT" cargo run --release --no-default-features --features hip -- -s 33554432 -n 1000
>>>>>>> main
```

Run the sweep

```bash
<<<<<<< HEAD
bash run.sh hip
```

Extra arguments are passed to every run of the sweep, for example `bash run.sh cuda --float --csv > results.csv`.
=======
bash run_hip.sh
```
>>>>>>> main
