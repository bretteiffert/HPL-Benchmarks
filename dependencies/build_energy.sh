#!/usr/bin/env bash
# Build libkernelenergy.so — the shared library exposing the C energy-profiler
# API (kernel_energy_capi.{h,cpp} over kernel.energy.h).
#
#   ./build_energy.sh            # auto-detect: HIP if rocm present, else CUDA/NVML
#   ./build_energy.sh --cuda     # force NVIDIA / NVML backend
#   ./build_energy.sh --hip      # force AMD / rocm_smi backend
#
# The CUDA path has no device kernels (host CUDA runtime + NVML only), so it
# builds with plain g++ against the CUDA runtime + libnvidia-ml — no nvcc or
# `module load` required.  The HIP path needs hipcc (it defines __HIPCC__, which
# selects the rocm_smi branch) and links rocm_smi + amdhip.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/kernel_energy_capi.cpp"
OUT="$HERE/libkernelenergy.so"

backend=""
case "${1:-}" in
    --cuda) backend=cuda ;;
    --hip)  backend=hip  ;;
    "")     backend="" ;;
    *) echo "usage: $0 [--cuda|--hip]" >&2; exit 2 ;;
esac

# ---- auto-detect backend ----------------------------------------------------
if [[ -z "$backend" ]]; then
    if command -v hipcc >/dev/null 2>&1 && { [[ -d /opt/rocm ]] || ls -d /opt/rocm-* >/dev/null 2>&1; }; then
        backend=hip
    else
        backend=cuda
    fi
fi
echo ">> backend: $backend"

# ---- locate a CUDA toolkit (include dir with nvml.h + cudart lib) -----------
find_cuda() {
    local c
    for c in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda; do
        [[ -n "$c" && -f "$c/include/nvml.h" ]] && { echo "$c"; return; }
    done
    # nvhpc layout: .../cuda/<ver>/targets/x86_64-linux/{include,lib}
    for c in $(ls -d /opt/nvidia/hpc_sdk/Linux_*/*/cuda/*/targets/*-linux 2>/dev/null | sort -rV); do
        [[ -f "$c/include/nvml.h" ]] && { echo "$c"; return; }
    done
    return 1
}

if [[ "$backend" == cuda ]]; then
    CUDA_DIR="$(find_cuda || true)"
    if [[ -z "$CUDA_DIR" ]]; then
        echo "!! could not find a CUDA toolkit with nvml.h; set CUDA_HOME" >&2
        exit 1
    fi
    echo ">> cuda: $CUDA_DIR"
    # libnvidia-ml.so ships with the driver (usually already on the linker path);
    # libcudart lives in the toolkit.  -rpath so the .so finds cudart at runtime.
    g++ -O2 -fPIC -shared -pthread \
        -I"$CUDA_DIR/include" \
        "$SRC" -o "$OUT" \
        -L"$CUDA_DIR/lib64" -L"$CUDA_DIR/lib" \
        -Wl,-rpath,"$CUDA_DIR/lib64" -Wl,-rpath,"$CUDA_DIR/lib" \
        -lcudart -lnvidia-ml
else
    ROCM="${ROCM_PATH:-$(ls -d /opt/rocm-* /opt/rocm 2>/dev/null | sort -rV | head -1)}"
    [[ -z "$ROCM" ]] && { echo "!! no ROCm install found; set ROCM_PATH" >&2; exit 1; }
    echo ">> rocm: $ROCM"
    hipcc -O2 -fPIC -shared -pthread \
        -I"$ROCM/include" \
        "$SRC" -o "$OUT" \
        -L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" \
        -lrocm_smi64 -lamdhip64
fi

echo ">> built $OUT"
