# Seven-point Stencil - Triton

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
--half     # fp16, grids of at most 105 per side
```

Other options: `-s L` cubic grid (1024) or `--grid X,Y,Z`, `--block-size 64,4,1` (the pinned 256-point program), `-n` timed iterations (100, after 1 untimed run), `--device`, `--list`, `--csv`, `--csv-raw`, `--no-energy`. Triton uses 8 warps on CUDA and 4 waves on HIP. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

```sh
pixi run -e cuda python main.py -s 1024 --block-size 64,4,1
```

Run the pinned configuration

```sh
bash run.sh cuda
```

## AMD ROCm/HIP

Run one

```sh
pixi run -e hip python main.py -s 1024 --block-size 64,4,1
```

Run the pinned configuration

```sh
bash run.sh hip
```

Extra arguments are passed to the run, for example `bash run.sh cuda --float --csv > results.csv`.
