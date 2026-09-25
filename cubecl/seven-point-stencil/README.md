## Seven-point Stencil - CubeCL
Run a single customized run or configuration sweep of relevant parameters on NVIDIA and AMD machines with provided shell scripts.
##### Set configurations
Set precision desired in `src/main.rs`
Set `type Precision = f64;` or `type Precision = f32;`
Set `const PRECISION_STR: &str = "double";` or `const PRECISION_STR: &str = "single";`


#### CUDA
Run one
```
cargo run --release --features cuda 1024 1024 1024 1024 1 1
```
Run all
```
source run_cuda.sh
```
####  HIP
Run one
```
cargo run --release --features hip 1024 1024 1024 512 1 1
```
Run all
```
source run_hip.sh
```