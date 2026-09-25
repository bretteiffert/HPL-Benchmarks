## Babelstream - CubeCL
Run a single customized run or configuration sweep of relevant parameters on NVIDIA and AMD machines with provided shell scripts.
##### Set configurations
Set precision desired in `src/main.rs`
Set `const USE_FLOAT: bool = false;` for double or `const USE_FLOAT: bool = true;` for single.
Set environment variable `BS_SM_COUNT` to number of SMs in GPU with `export BS_SM_COUNT=[number of SMs]`

#### CUDA
Run one
```
cargo run --release --features cuda -- -s 33554432 -n 1000
```
Run all
```
source run_cuda.sh
```
####  HIP
Run one
```
cargo run --release --features hip -- -s 33554432 -n 1000
```
Run all
```
source run_hip.sh
```

