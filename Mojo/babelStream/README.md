## BabelStream - Mojo
Run a single configuration or the provided size sweep on an NVIDIA CUDA or AMD ROCm/HIP system. The MAX runtime selects the available GPU backend.

#### Setup
The Pixi environment targets `linux-64`. Install the pinned Mojo and MAX versions from this directory:
```sh
pixi install
```

##### Set configurations
Set the precision in `babelstream.mojo`: `comptime USE_FLOAT = False` selects fp64 and `True` selects fp32. The `--float` option does not change the compiled precision.

Set `BS_SM_COUNT` to the exact number of SMs on an NVIDIA GPU or CUs on an AMD GPU. It is required because the dot-product grid uses four 1024-thread workgroups per SM/CU; array sizes must also be multiples of 1024.
```sh
export BS_SM_COUNT=132
```

#### CUDA
Run one:
```sh
BS_SM_COUNT="$BS_SM_COUNT" pixi run mojo babelstream.mojo -s 33554432 -n 1000
```

Run the size sweep:
```sh
bash run_cuda.sh
```

#### ROCm/HIP
Run one:
```sh
BS_SM_COUNT="$BS_SM_COUNT" pixi run mojo babelstream.mojo -s 33554432 -n 1000
```

Run the size sweep:
```sh
bash run_hip.sh
```
