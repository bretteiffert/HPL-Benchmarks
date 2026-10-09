# Hartree-Fock - Mojo

## Setup
The Pixi environment targets `linux-64`.
```sh
pixi install
```

## Configuration

Set precision
```mojo
comptime PRECISION = 64
```

## CUDA
Run one:
```sh
pixi run mojo hartree-fock.mojo ../../data/hartree-fock/he64 --csv --iters=10
```

## ROCm/HIP

Run one
```sh
pixi run mojo hartree-fock.mojo ../../data/hartree-fock/he64 --csv --iters=10
```

Run the sweep
```sh
for size in he16 he32 he64 he128 he256; do
  pixi run mojo hartree-fock.mojo "../../data/hartree-fock/$size" --csv --iters=10
done
```
