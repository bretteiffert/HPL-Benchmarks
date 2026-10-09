# Seven-point Stencil - Mojo

## Setup

```sh
pixi install
```

## Configration

Set precision
```mojo
comptime precision = DType.float64
comptime precision_str = "double"
```

## CUDA
Run one:
```sh
pixi run mojo laplacian.mojo 1024 1024 1024 1024 1 1
```

Run the sweep
```sh
bash run_cuda.sh
```

## ROCm/HIP
Run one
```sh
pixi run mojo laplacian.mojo 1024 1024 1024 512 1 1
```

Run the sweep
```sh
bash run_hip.sh
```
