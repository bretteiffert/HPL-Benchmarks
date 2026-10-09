# miniBUDE - Mojo

## Setup

```sh
pixi install
```

## Configuration

Set precision
```mojo
comptime PRECISION_BITS = 32
```

## CUDA

Run one
```sh
pixi run mojo miniBUDE.mojo --deck ../../data/miniBUDE/bm1 -w 32 -p 64
```

Run the sweep
```sh
bash run_cuda.sh
```

## ROCm/HIP

Run one
```sh
pixi run mojo miniBUDE.mojo --deck ../../data/miniBUDE/bm1 -w 64 -p 64
```

Run the sweep
```sh
bash run_hip.sh
```
