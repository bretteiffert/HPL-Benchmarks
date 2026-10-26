# BabelStream - Triton

## Setup

Install [pixi](https://pixi.sh), then the environment for the backend on the current machine (Python 3.12, PyTorch 2.13.0 and Triton 3.7.1; the NVIDIA or AMD driver must already be installed):

```sh
pixi install -e cuda
# or
pixi install -e hip
```

Each environment holds exactly one backend; the plain default environment has no GPU backend and the benchmark refuses to run in it.

Energy is measured through `energy.py` in `../../dependencies`, which loads `libkernelenergy.so` from the same folder, so build that library first. Pass `--no-energy` to run without it.

## Configuration

Precision is selected at run time with a flag:

```sh
--double   # fp64, the default
--float    # fp32
--half     # fp16
```

Other options: `-s` array size (33,554,432), `-n` iterations (1000, the first is warm-up), `--block-size` (1024, a power of two), `--device`, `--list`, `--csv`, `--csv-raw`, `--no-energy`, `--mibibytes`. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver. `FMAD` is reported but has no effect: Triton fuses multiply-add by default.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

```sh
FMAD=1 pixi run -e cuda python main.py -s 33554432 -n 1000
```

Run the sweep

```sh
bash run.sh cuda
```

## AMD ROCm/HIP

Run one

```sh
FMAD=1 pixi run -e hip python main.py -s 33554432 -n 1000
```

Run the sweep

```sh
bash run.sh hip
```

Extra arguments are passed to every run of the sweep, for example `bash run.sh cuda --float --csv > results.csv`.
