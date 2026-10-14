# BabelStream - Mojo

## Setup

```sh
pixi install
```

## Set configurations

Set `BS_SM_COUNT` to the exact number of SMs on an NVIDIA GPU or CUs on an AMD GPU.
```sh
export BS_SM_COUNT=132
```

## CUDA
Run one
```sh
BS_SM_COUNT="$BS_SM_COUNT" pixi run mojo babelstream.mojo -s 33554432 -n 1000
```

Run the sweep
```sh
bash run_cuda.sh
```

## ROCm/HIP
Run one:
```sh
BS_SM_COUNT="$BS_SM_COUNT" pixi run mojo babelstream.mojo -s 33554432 -n 1000
```

Run the sweep
```sh
bash run_hip.sh
```
