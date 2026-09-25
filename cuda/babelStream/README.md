## Babelstream - CUDA
Run a single customized run or configuration sweep of relevant parameters.

##### Set configurations
Set precision desired in `main.cpp`
Set `bool use_float = false;` for double or `bool use_float = true;` for single.

#### CUDA
Compile
```
make
```
Run one
```
./babelstream_cuda -s 33554432 -n 1000
```
Run all
```
source run_cuda.sh
```

