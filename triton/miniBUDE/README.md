# miniBUDE - Triton

## Setup

Install [pixi](https://pixi.sh), then the environment for the backend on the current machine (Python 3.12, PyTorch 2.13.0 and Triton 3.7.1; the NVIDIA or AMD driver must already be installed):

```sh
pixi install -e cuda
# or
pixi install -e hip
```

Each environment holds exactly one backend; the plain default environment has no GPU backend and the benchmark refuses to run in it.

Energy is measured through `energy.py` in `../../dependencies`, which loads `libkernelenergy.so` from the same folder, so build that library first. Pass `--no-energy` to run without it.

The input deck is read from `../../data/miniBUDE/bm1` (relative to this folder) unless `--deck` names another.

## Configuration

Precision is selected at run time with a flag:

```sh
--double   # fp64, the default
--float    # fp32, checked against one untimed fp64 pass
```

Other options keep the upstream miniBUDE names: `-i` timed iterations (8), `--warmup` (2), `-n` poses (0 = whole deck), `-p` ppwi list or `all` (1), `-w` work-group size list (one warp; must be a power-of-two multiple of the warp size), `--deck`, `-o`, `-r`, `-d` device index or name substring. Every (ppwi, wgsize) combination runs in one invocation, and combinations that don't divide the pose count are skipped with a note. Also `--list`, `--csv`, `--csv-raw`, `--no-energy`. `GPU_NAME` and `GPU_DRIVER` override the reported device name and driver.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

```sh
pixi run -e cuda python main.py --deck ../../data/miniBUDE/bm1 -w 32 -p 64
```

Run the sweep

```sh
bash run.sh cuda
```

## AMD ROCm/HIP

Run one

```sh
pixi run -e hip python main.py --deck ../../data/miniBUDE/bm1 -w 64 -p 64
```

Run the sweep

```sh
bash run.sh hip
```

Extra arguments are passed to the sweep, for example `bash run.sh cuda --float --csv > results.csv`.
