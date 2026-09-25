## Seven-point Stencil - HIP
Run a single customized run or configuration sweep of relevant parameters.
##### Set configurations
Set precision desired in `laplacian.cu`
Set `using precision = double;` or `using precision = single;`
Set `char precision_str[] = "double";` or `char precision_str[] = "single";`


#### CUDA
Compile
```
make
```

Run one
```
./laplacian_kernel 1024 1024 1024 1024 1 1
```
Run all
```
source run_cuda.sh
```