## miniBUDE - Mojo
Run a single configuration or the provided PPWI sweep on an NVIDIA CUDA or AMD ROCm/HIP system. The MAX runtime selects the available GPU backend, and the bundled deck is loaded from `../../data/miniBUDE/bm1`.

#### Setup
The Pixi environment targets `linux-64`. Install the pinned Mojo and MAX versions from this directory:
```sh
pixi install
```

##### Set configurations
Set `comptime PRECISION_BITS = 32` or `64` in `miniBUDE.mojo`; precision is not a command-line option.

For NVIDIA, select 32-bit precision. This Mojo release cannot instantiate the fp64 kernel on NVIDIA because fp64 GPU `sin` and `cos` are unavailable. AMD ROCm/HIP can use the default fp64 configuration.

#### CUDA
After selecting fp32, run one with the CUDA sweep's 32-thread workgroup:
```sh
pixi run mojo miniBUDE.mojo --deck ../../data/miniBUDE/bm1 -w 32 -p 64
```

Run the PPWI sweep:
```sh
bash run_cuda.sh
```

#### ROCm/HIP
Run one with the HIP sweep's 64-thread workgroup:
```sh
pixi run mojo miniBUDE.mojo --deck ../../data/miniBUDE/bm1 -w 64 -p 64
```

Run the PPWI sweep:
```sh
bash run_hip.sh
```

The scripts sweep PPWI values from 1 through 128; CUDA uses workgroup size 32 and HIP uses 64. Pass `--golden PATH` only when an external fp64 reference-energy file is available for differential metrics.
