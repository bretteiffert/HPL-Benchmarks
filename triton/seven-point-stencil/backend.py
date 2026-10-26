"""Backend selection, device list, and device name / driver / compute-unit lookup.

Shared by every Triton benchmark (keep this file identical across them).

The backend is fixed by the installed PyTorch build (pixi environment ``cuda`` or
``hip``); there is no run-time switch. Device facts are resolved in order:
override env var -> PyTorch runtime query -> vendor library loaded at run time
-> placeholder plus a note.
"""

import ctypes
import os

CUDA = "CUDA"
HIP = "HIP"

# Vendor attribute ids (cuda.h CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
# hip_runtime_api.h hipDeviceAttributeMultiprocessorCount in ROCm 7.1).
_CU_ATTR_SM_COUNT = 16
_HIP_ATTR_CU_COUNT = 63


class BackendError(Exception):
    pass


class DeviceInfo:
    def __init__(self, index, name, driver, compute_units, unit_label, notes):
        self.index = index
        self.name = name
        self.driver = driver
        self.compute_units = compute_units  # int, or None when unknown
        self.unit_label = unit_label        # "SMs" or "CUs"
        self.notes = notes


def load_torch():
    try:
        import torch
    except ImportError:
        raise BackendError("PyTorch is not installed in this environment; "
                           "run through 'pixi run -e cuda' or 'pixi run -e hip'")
    return torch


def select(torch):
    """Return CUDA or HIP; exactly one must be present in the installed build."""
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    if hip and cuda:
        raise BackendError("PyTorch reports both CUDA %s and HIP %s; install exactly "
                           "one backend (pixi environment 'cuda' or 'hip')" % (cuda, hip))
    if hip:
        return HIP
    if cuda:
        return CUDA
    raise BackendError("PyTorch has no GPU backend (CPU-only build); install exactly "
                       "one backend (pixi environment 'cuda' or 'hip')")


def device_count(torch):
    return torch.cuda.device_count()


def list_devices(torch, backend):
    return [(i, _name(torch, backend, i)[0]) for i in range(device_count(torch))]


def describe(torch, backend, index):
    notes = []
    name, note = _name(torch, backend, index)
    if note:
        notes.append(note)
    driver = _driver(backend)
    if driver is None:
        driver = "unknown"
        notes.append("driver version unavailable; set GPU_DRIVER to override")
    units = _compute_units(torch, backend, index)
    return DeviceInfo(index, name, driver, units,
                      "CUs" if backend == HIP else "SMs", notes)


# -- name ---------------------------------------------------------------------

def _name(torch, backend, index):
    env = os.environ.get("GPU_NAME")
    if env:
        return env, None
    try:
        name = torch.cuda.get_device_properties(index).name
        if name:
            return name, None
    except Exception:
        pass
    name = _vendor_name(backend, index)
    if name:
        return name, None
    return ("%s device %d" % (backend, index),
            "device name unavailable; set GPU_NAME to override")


def _vendor_name(backend, index):
    buf = ctypes.create_string_buffer(256)
    if backend == CUDA:
        lib = _cuda()
        dev = ctypes.c_int()
        if (lib and lib.cuDeviceGet(ctypes.byref(dev), index) == 0
                and lib.cuDeviceGetName(buf, len(buf), dev) == 0):
            return buf.value.decode(errors="replace")
    else:
        lib = _hip()
        if lib and lib.hipDeviceGetName(buf, len(buf), index) == 0:
            return buf.value.decode(errors="replace")
    return None


# -- driver -------------------------------------------------------------------

def _driver(backend):
    env = os.environ.get("GPU_DRIVER")
    if env:
        return env
    # PyTorch exposes only the toolkit it was built with, not the driver, so
    # the driver comes from the vendor libraries.
    version = ctypes.c_int()
    if backend == CUDA:
        cuda_ver = None
        lib = _cuda()
        if lib and lib.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value > 0:
            v = version.value
            cuda_ver = "CUDA %d.%d" % (v // 1000, (v % 1000) // 10)
        nvml = _nvml_driver()
        if nvml and cuda_ver:
            return "%s (%s)" % (nvml, cuda_ver)
        return nvml or cuda_ver
    lib = _hip()
    if lib and lib.hipDriverGetVersion(ctypes.byref(version)) == 0 and version.value > 0:
        v = version.value
        return "HIP %d.%d.%d" % (v // 10000000, (v // 100000) % 100, v % 100000)
    return None


def _nvml_driver():
    lib = _open(["libnvidia-ml.so.1", "libnvidia-ml.so"])
    if lib is None:
        return None
    try:
        if lib.nvmlInit_v2() != 0:
            return None
        buf = ctypes.create_string_buffer(96)
        ok = lib.nvmlSystemGetDriverVersion(buf, ctypes.c_uint(len(buf))) == 0
        lib.nvmlShutdown()
        return buf.value.decode(errors="replace") if ok and buf.value else None
    except AttributeError:
        return None


# -- compute units ------------------------------------------------------------

def _compute_units(torch, backend, index):
    try:
        n = int(torch.cuda.get_device_properties(index).multi_processor_count)
        if n > 0:
            return n
    except Exception:
        pass
    value = ctypes.c_int()
    if backend == CUDA:
        lib = _cuda()
        dev = ctypes.c_int()
        if (lib and lib.cuDeviceGet(ctypes.byref(dev), index) == 0
                and lib.cuDeviceGetAttribute(ctypes.byref(value), _CU_ATTR_SM_COUNT, dev) == 0
                and value.value > 0):
            return value.value
    else:
        lib = _hip()
        if (lib and lib.hipDeviceGetAttribute(ctypes.byref(value), _HIP_ATTR_CU_COUNT, index) == 0
                and value.value > 0):
            return value.value
    return None


# -- vendor libraries (loaded lazily, once) -----------------------------------

_LIBS = {}


def _open(names):
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    return None


def _cuda():
    if "cuda" not in _LIBS:
        lib = _open(["libcuda.so.1", "libcuda.so"])
        try:
            if lib is not None and lib.cuInit(0) != 0:
                lib = None
        except AttributeError:
            lib = None
        _LIBS["cuda"] = lib
    return _LIBS["cuda"]


def _hip():
    if "hip" not in _LIBS:
        names = ["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"]
        try:
            import torch
            names.append(os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so"))
        except ImportError:
            pass
        lib = _open(names)
        try:
            if lib is not None and lib.hipInit(0) != 0:
                lib = None
        except AttributeError:
            lib = None
        _LIBS["hip"] = lib
    return _LIBS["hip"]
