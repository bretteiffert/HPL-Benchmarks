## Seven-point Stencil - Triton
Run a single tile configuration or the provided CUDA and HIP sweeps. These commands assume existing `triton-cuda` and `triton-hip` Mamba environments with the corresponding PyTorch build and Triton installed. PyTorch selects the backend from the active environment; ROCm also uses the `torch.cuda` API.

##### Set configurations
Set both `precision` and `precision_str` near the top of `laplacian.py`; the defaults are `torch.float32` and `"float"`.

The positional arguments are `nx ny nz blk_x blk_y blk_z`. Each run performs 1,000 timed iterations. The sweep uses a 1024 cubed grid, which allocates two main arrays totaling about 8 GiB in fp32 or 16 GiB in fp64, before runtime overhead; use smaller dimensions when device memory is limited.

#### CUDA
Run one:
```sh
mamba run -n triton-cuda python laplacian.py 1024 1024 1024 1024 1 1
```

Run the tile sweep:
```sh
bash run_cuda.sh
```

#### HIP
Run one:
```sh
mamba run -n triton-hip python laplacian.py 1024 1024 1024 512 1 1
```

Run the tile sweep:
```sh
bash run_hip.sh
```
