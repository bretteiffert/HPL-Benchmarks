# miniBUDE - Triton

## Setup

Install the environment for the backend on the current machine:
```sh
pixi install -e cuda
# or
pixi install -e hip
```

## Configuration
```python
PRECISION_BITS = 64
```


## CUDA
Run one
```sh
pixi run -e cuda python miniBUDE.py --deck ../../data/miniBUDE/bm1 -w 32 -p 64
```

Run the sweep
```sh
bash run_cuda.sh
```

## HIP
Run one:
```sh
pixi run -e hip python miniBUDE.py --deck ../../data/miniBUDE/bm1 -w 64 -p 64
```

Run the sweep
```sh
bash run_hip.sh
```
