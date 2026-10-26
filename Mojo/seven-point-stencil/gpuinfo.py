"""Vendor-library device queries, loaded at run time with ctypes.

Fallback used by backend.mojo when the MAX runtime cannot answer. Every
function takes (backend, index), returns "" or -1 on any failure, and never
raises.

CUDA: libcuda   cuDeviceGetName, cuDeviceGetAttribute(16 = SM count),
                cuDriverGetVersion
      NVML      nvmlSystemGetDriverVersion
HIP:  libamdhip64  hipDeviceGetName, hipDeviceGetAttribute(63 = CU count),
                   hipDriverGetVersion
"""

import ctypes

_CUDA_LIBS = ("libcuda.so.1", "libcuda.so")
_NVML_LIBS = ("libnvidia-ml.so.1", "libnvidia-ml.so")
_HIP_LIBS = ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5")

_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
_HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 63


def _load(names):
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    return None


def _cuda_device(index):
    lib = _load(_CUDA_LIBS)
    if lib is None or lib.cuInit(0) != 0:
        return None, None
    dev = ctypes.c_int()
    if lib.cuDeviceGet(ctypes.byref(dev), int(index)) != 0:
        return None, None
    return lib, dev


def device_name(backend, index):
    try:
        buf = ctypes.create_string_buffer(256)
        if backend == "cuda":
            lib, dev = _cuda_device(index)
            if lib is None or lib.cuDeviceGetName(buf, 256, dev) != 0:
                return ""
        else:
            lib = _load(_HIP_LIBS)
            if lib is None:
                return ""
            dev = ctypes.c_int()
            if lib.hipDeviceGet(ctypes.byref(dev), int(index)) != 0:
                return ""
            if lib.hipDeviceGetName(buf, 256, dev) != 0:
                return ""
        return buf.value.decode(errors="replace").strip()
    except Exception:
        return ""


def sm_count(backend, index):
    try:
        value = ctypes.c_int()
        if backend == "cuda":
            lib, dev = _cuda_device(index)
            if lib is None:
                return -1
            rc = lib.cuDeviceGetAttribute(
                ctypes.byref(value), _CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev
            )
        else:
            lib = _load(_HIP_LIBS)
            if lib is None:
                return -1
            rc = lib.hipDeviceGetAttribute(
                ctypes.byref(value),
                _HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                int(index),
            )
        return value.value if rc == 0 and value.value > 0 else -1
    except Exception:
        return -1


def driver_version(backend, index):
    """cuDriverGetVersion / hipDriverGetVersion as an integer."""
    try:
        value = ctypes.c_int()
        lib = _load(_CUDA_LIBS if backend == "cuda" else _HIP_LIBS)
        if lib is None:
            return -1
        fn = lib.cuDriverGetVersion if backend == "cuda" else lib.hipDriverGetVersion
        if fn(ctypes.byref(value)) != 0 or value.value <= 0:
            return -1
        return value.value
    except Exception:
        return -1


def nvml_driver(backend, index):
    """NVIDIA kernel-driver version string, e.g. 550.54.15 (CUDA only)."""
    if backend != "cuda":
        return ""
    try:
        lib = _load(_NVML_LIBS)
        if lib is None:
            return ""
        init = getattr(lib, "nvmlInit_v2", None) or lib.nvmlInit
        if init() != 0:
            return ""
        try:
            buf = ctypes.create_string_buffer(96)
            if lib.nvmlSystemGetDriverVersion(buf, 96) != 0:
                return ""
            return buf.value.decode(errors="replace").strip()
        finally:
            lib.nvmlShutdown()
    except Exception:
        return ""
