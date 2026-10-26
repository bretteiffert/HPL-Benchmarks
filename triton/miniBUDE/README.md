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
--float    # fp32, the default; checked against one untimed fp64 pass
--double   # optional fp64 mode
```

The launch is pinned to `-p 1 -w 256`, which gives 8 CUDA warps or 4 HIP waves; Triton's pipeline is pinned to `num_stages=3`. Other PPWI, work-group, and stage values are not tuned. Other options include `-i` timed iterations (8), `--warmup` (2), `-n` poses (0 = whole deck), `--deck`, `-o`, `-r`, `-d`, `--list`, `--csv`, `--csv-raw`, and `--no-energy`.

In the CSV modes only the CSV goes to stdout; the report goes to stderr, so `> results.csv` gives a clean file. Run `--help` for the full list.

## NVIDIA CUDA

Run one

```sh
pixi run -e cuda python main.py --deck ../../data/miniBUDE/bm1 --float -w 256 -p 1
```

Run the canonical benchmark

```sh
bash run.sh cuda
```

## AMD ROCm/HIP

Run one

```sh
pixi run -e hip python main.py --deck ../../data/miniBUDE/bm1 --float -w 256 -p 1
```

Run the canonical benchmark

```sh
bash run.sh hip
```

Extra arguments are passed through, for example `bash run.sh cuda --csv > results.csv`.
