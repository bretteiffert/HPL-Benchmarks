# Seven-point Stencil - Triton

## Setup

Install the environment for the backend on the current machine
```sh
pixi install -e cuda
# or
pixi install -e hip
```

## Configuration

Set precision
```python
precision = torch.float32
precision_str = "float"
```

## CUDA
Run one
```sh
pixi run -e cuda python laplacian.py 1024 1024 1024 1024 1 1
```

Run the tile sweep
```sh
bash run_cuda.sh
```

## HIP
Run one
```sh
pixi run -e hip python laplacian.py 1024 1024 1024 512 1 1
```

Run the sweep
```sh
bash run_hip.sh
```
