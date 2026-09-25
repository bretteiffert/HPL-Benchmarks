## BabelStream - Triton
Run a single configuration or the provided CUDA and HIP sweeps. PyTorch selects the backend from the active Pixi environment; ROCm also uses the `torch.cuda` API. Run the commands below from `triton/babelStream`.

#### Setup
The included `pixi.toml` targets `linux-64` with Python 3.12 and provides separate environments for the official PyTorch 2.13.0 CUDA 13.0 and ROCm 7.1 wheels. Both environments use the matching Triton 3.7.1 runtime.

Install the environment for the backend on the current machine:
```sh
pixi install -e cuda
# or
pixi install -e hip
```

##### Set configurations
Set `USE_FLOAT = True` in `babelstream.py` for fp32 or `USE_FLOAT = False` for fp64 (the default). The `--float` option is currently a no-op. Array sizes must be multiples of 1024.

#### CUDA
Run one:
```sh
FMAD=1 pixi run -e cuda python babelstream.py -s 33554432 -n 1000
```

Run the size sweep:
```sh
bash run_cuda.sh
```

#### HIP
Run one:
```sh
FMAD=1 pixi run -e hip python babelstream.py -s 33554432 -n 1000
```

Run the size sweep:
```sh
bash run_hip.sh
```
