## Seven-point Stencil - CUDA
Run a single customized run or configuration sweep of relevant parameters.
##### Set configurations
Set precision desired in `main.cpp`
Set `#define PRECISION_BITS 64` or `#define PRECISION_BITS 64`

#### CUDA
Compile
```
make
```
Run one
```
./bude-cuda -p 64 -w 32
```
Run all
```
source run_cuda.sh
```