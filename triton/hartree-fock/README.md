# Hartree-Fock - Triton

## Setup

Install [pixi](https://pixi.sh), then the environment for the backend on the current machine (Python 3.12, PyTorch 2.13.0 and Triton 3.7.1; the NVIDIA or AMD driver must already be installed):

```sh
pixi install -e cuda
# or
pixi install -e hip
```

Each environment holds exactly one backend; the plain default environment has no GPU backend and the benchmark refuses to run in it.

Energy is measured through `energy.py` in `../../dependencies`, which loads `libkernelenergy.so` from the same folder, so build that library first. Pass `--no-energy` to run without it.

Input files are read from `../../data/hartree-fock` (he16 ... he256).

## Configuration

Precision is selected at run time with a flag:

```sh
--double   # fp64, the default
--float    # fp32
```

There is no `--half`: the 1e-12 screening thresholds are below the float16 range.

Other options: `-n` timed iterations (10, after 1 untimed run), `--legacy-sqrt`, `--check` / `--no-check` / `--max-check-atoms` (float64 CPU reference, on by default up to 128 atoms), `--block-size` (256), `--device`, `--list`, `--csv`, `--csv-raw`, `--no-energy`. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

```sh
pixi run -e cuda python main.py ../../data/hartree-fock/he64 -n 10
```

Run the sweep

```sh
bash run.sh cuda
```

## AMD ROCm/HIP

Run one

```sh
pixi run -e hip python main.py ../../data/hartree-fock/he64 -n 10
```

Run the sweep

```sh
bash run.sh hip
```

Extra arguments are passed to every run of the sweep, for example `bash run.sh cuda --float --csv > results.csv`.
