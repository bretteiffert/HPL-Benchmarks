## Hartree-Fock - CUDA
Run a single customized run or configuration sweep of relevant parameters.
##### Set configurations
Set precision desired in `hartree-fock.cu`
Set `#define PRECISION 64` or `#define PRECISION 32`

#### CUDA
Compile
```
make
```
Run one
```
./hartree-fock-cuda ../../data/hartree-fock/he64 --csv --iters=10
```
Run all
```
source run_cuda.sh
```