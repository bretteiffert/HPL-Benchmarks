# Hartree-Fock - Triton

## Setup

Install the environment for the backend on the current machine:
```sh
pixi install -e cuda
# or
pixi install -e hip
```

## Configuration
```python
PRECISION = 64
```

## CUDA
Run one
```sh
pixi run -e cuda python hartree-fock.py ../../data/hartree-fock/he64 --csv --iters=10
```

Run the sweep
```sh
bash run_cuda.sh
```

## HIP
Run one
```sh
pixi run -e hip python hartree-fock.py ../../data/hartree-fock/he64 --csv --iters=10
```

Run the sweep
```sh
bash run_hip.sh
```
