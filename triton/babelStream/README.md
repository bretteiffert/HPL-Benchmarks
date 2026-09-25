## BabelStream - Triton
Run a single configuration or the provided CUDA and HIP sweeps. These commands assume existing `triton-cuda` and `triton-hip` Mamba environments with the corresponding PyTorch build and Triton installed. PyTorch selects the backend from the active environment; ROCm also uses the `torch.cuda` API.

##### Set configurations
Set `USE_FLOAT = True` in `babelstream.py` for fp32 or `USE_FLOAT = False` for fp64 (the default). The `--float` option is currently a no-op. Array sizes must be multiples of 1024.

#### CUDA
Run one:
```sh
FMAD=1 mamba run -n triton-cuda python babelstream.py -s 33554432 -n 1000
```

Run the size sweep:
```sh
bash run_cuda.sh
```

#### HIP
Run one:
```sh
FMAD=1 mamba run -n triton-hip python babelstream.py -s 33554432 -n 1000
```

Run the size sweep:
```sh
bash run_hip.sh
```
