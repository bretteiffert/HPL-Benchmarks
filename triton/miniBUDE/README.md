## miniBUDE - Triton
Run a single work configuration or the provided CUDA and HIP sweeps. These commands assume existing `triton-cuda` and `triton-hip` Mamba environments with the corresponding PyTorch build and Triton installed. PyTorch selects the backend from the active environment; ROCm also uses the `torch.cuda` API.

##### Set configurations
Set `PRECISION_BITS = 32` or `PRECISION_BITS = 64` near the top of `miniBUDE.py`; there is no precision command-line option. The default deck is `../../data/miniBUDE/bm1`.

Triton derives `num_warps` as work-group size divided by the target warp width, so `--wgsize` must be a multiple of that width (normally 32 for CUDA and 64 for HIP). The pose count must be at least, and divisible by, `ppwi * wgsize`; unsupported or resource-heavy configurations are skipped. fp32 runs also compute an fp64 differential reference pass.

#### CUDA
Run one:
```sh
mamba run -n triton-cuda python miniBUDE.py --deck ../../data/miniBUDE/bm1 -w 32 -p 64
```

Run the PPWI sweep:
```sh
bash run_cuda.sh
```

#### HIP
Run one:
```sh
mamba run -n triton-hip python miniBUDE.py --deck ../../data/miniBUDE/bm1 -w 64 -p 64
```

Run the PPWI sweep:
```sh
bash run_hip.sh
```
