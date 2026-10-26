# Hartree-Fock - CubeCL

## Setup

Install Rust 1.95 or newer, the vendor toolkit and driver required by the selected backend, and set vendor path.

```bash
export CUDA_PATH=[path-to-cuda]
export HIP_PATH=[path-to-rocm]
```

The build links `libkernelenergy.so` from `../../dependencies`, so build that library first. Inputs are read from `../../data/hartree-fock`.

If your environment sets `CC=nvc` (NVIDIA HPC SDK), nothing extra is needed: `.cargo/config.toml` builds native dependencies with gcc for this project only.

## Configuration

Precision is selected at run time with a flag:

```bash
--double        # fp64, the default
--float         # fp32
```

The fp64 CPU reference check runs automatically up to 128 atoms; it scales as natoms⁴, so it is skipped for larger inputs.
The pinned configuration uses native-precision square roots.

```bash
--check                 # always run the reference check
--no-check              # never run it
--max-check-atoms N     # run it automatically up to N atoms (default 128)
```

Other options:

```bash
-n, --numtimes NUM      # timed iterations (default 10), after one untimed run
--block-size 256        # pinned units per cube
--device INDEX          # device to run on (default 0); --list shows them
--csv                   # CSV to stdout: one summary row
--csv-raw               # CSV to stdout: one row per timed iteration
--no-energy             # skip energy measurement
```

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

```bash
cargo run --release --no-default-features --features cuda -- ../../data/hartree-fock/he64 -n 10
```

Run the input sweep

```bash
bash run.sh cuda
```

## AMD ROCm/HIP

Run one

```bash
cargo run --release --no-default-features --features hip -- ../../data/hartree-fock/he64 -n 10
```

Run the sweep

```bash
bash run.sh hip
```

Extra arguments are passed to every run of the sweep, for example `bash run.sh cuda --float --csv > results.csv`.
