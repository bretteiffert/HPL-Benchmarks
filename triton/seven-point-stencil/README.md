## Seven-point Stencil - Triton
Run a single tile configuration or the provided CUDA and HIP sweeps. PyTorch selects the backend from the active Pixi environment; ROCm also uses the `torch.cuda` API. Run the commands below from `triton/seven-point-stencil`.

#### Setup
The included `pixi.toml` targets `linux-64` with Python 3.12 and provides separate environments for the official PyTorch 2.13.0 CUDA 13.0 and ROCm 7.1 wheels. Both environments use the matching Triton 3.7.1 runtime.

Install the environment for the backend on the current machine:
```sh
pixi install -e cuda
# or
pixi install -e hip
```

##### Set configurations
Set both `precision` and `precision_str` near the top of `laplacian.py`; the defaults are `torch.float32` and `"float"`.

The positional arguments are `nx ny nz blk_x blk_y blk_z`. Each run performs 1,000 timed iterations. The sweep uses a 1024 cubed grid, which allocates two main arrays totaling about 8 GiB in fp32 or 16 GiB in fp64, before runtime overhead; use smaller dimensions when device memory is limited.

#### CUDA
Run one:
```sh
pixi run -e cuda python laplacian.py 1024 1024 1024 1024 1 1
```

Run the tile sweep:
```sh
bash run_cuda.sh
```

#### HIP
Run one:
```sh
pixi run -e hip python laplacian.py 1024 1024 1024 512 1 1
```

Run the tile sweep:
```sh
bash run_hip.sh
```
