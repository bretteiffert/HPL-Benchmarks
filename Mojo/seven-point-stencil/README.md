## Seven-point Stencil - Mojo
Run a single configuration or the provided workgroup sweep on an NVIDIA CUDA or AMD ROCm/HIP system. The MAX runtime selects the available GPU backend.

#### Setup
The Pixi environment targets `linux-64`. Install the pinned Mojo and MAX versions from this directory:
```sh
pixi install
```

##### Set configurations
Set the precision in `laplacian.mojo`. Use `comptime precision = DType.float64` with `precision_str = "double"`, or `DType.float32` with `precision_str = "single"`.

The final three arguments are the x, y, and z workgroup dimensions. Their product must be supported by the target GPU.

#### CUDA
Run one:
```sh
pixi run mojo laplacian.mojo 1024 1024 1024 1024 1 1
```

Run the workgroup sweep:
```sh
bash run_cuda.sh
```

#### ROCm/HIP
Run one:
```sh
pixi run mojo laplacian.mojo 1024 1024 1024 512 1 1
```

Run the workgroup sweep:
```sh
bash run_hip.sh
```

Both scripts currently test the same workgroups, including configurations with up to 1024 threads, on a `1024^3` grid. Use a supported GPU with sufficient memory.
