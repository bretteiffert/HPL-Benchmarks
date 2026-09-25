## Seven-point Stencil - CubeCL
Run a single customized run or configuration sweep of relevant parameters on NVIDIA and AMD machines with provided shell scripts.
##### Set configurations
Set precision desired in `src/main.rs`
Set `const PRECISION_BITS: usize = 64;` or `const PRECISION_BITS: usize = 32;`

#### CUDA
Run one
```
cargo run --release --features cuda -- -p 64 -w 32
```
Run all
```
source run_cuda.sh
```
####  HIP
Run one
```
cargo run --release --features hip -- -p 64 -w 32
```
Run all
```
source run_hip.sh
```