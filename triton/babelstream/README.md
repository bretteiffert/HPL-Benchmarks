# BabelStream - Triton

## Setup

Install the environment for the backend on the current machine
```sh
pixi install -e cuda
# or
pixi install -e hip
```

## Configuration

```python
USE_FLOAT = True
```

## CUDA
Run one
```sh
FMAD=1 pixi run -e cuda python babelstream.py -s 33554432 -n 1000
```

Run the sweep
```sh
bash run_cuda.sh
```

## HIP
Run one
```sh
FMAD=1 pixi run -e hip python babelstream.py -s 33554432 -n 1000
```

Run the sweep
```sh
bash run_hip.sh
```
