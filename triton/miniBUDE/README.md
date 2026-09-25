## miniBUDE - Triton
Run a single work configuration or the provided CUDA and HIP sweeps. PyTorch selects the backend from the active Pixi environment; ROCm also uses the `torch.cuda` API. Run the commands below from `triton/miniBUDE`.

#### Setup
The included `pixi.toml` targets `linux-64` with Python 3.12 and provides separate environments for the official PyTorch 2.13.0 CUDA 13.0 and ROCm 7.1 wheels. Both environments use the matching Triton 3.7.1 runtime.

Install the environment for the backend on the current machine:
```sh
pixi install -e cuda
# or
pixi install -e hip
```

##### Set configurations
Set `PRECISION_BITS = 32` or `PRECISION_BITS = 64` near the top of `miniBUDE.py`; there is no precision command-line option. The default deck is `../../data/miniBUDE/bm1`.

Triton derives `num_warps` as work-group size divided by the target warp width, so `--wgsize` must be a multiple of that width (normally 32 for CUDA and 64 for HIP). The pose count must be at least, and divisible by, `ppwi * wgsize`; unsupported or resource-heavy configurations are skipped. fp32 runs also compute an fp64 differential reference pass.

#### CUDA
Run one:
```sh
pixi run -e cuda python miniBUDE.py --deck ../../data/miniBUDE/bm1 -w 32 -p 64
```

Run the PPWI sweep:
```sh
bash run_cuda.sh
```

#### HIP
Run one:
```sh
pixi run -e hip python miniBUDE.py --deck ../../data/miniBUDE/bm1 -w 64 -p 64
```

Run the PPWI sweep:
```sh
bash run_hip.sh
```
