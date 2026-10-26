# Backend selection, device list, and device name / driver / SM-count lookup.
#
# One backend per compilation, chosen with a define:
#     mojo run -D BACKEND=cuda ...   or   mojo run -D BACKEND=hip ...
# Building without BACKEND (or with any other value) fails to compile.
#
# Device facts are resolved in this order:
#   override env var (GPU_NAME, GPU_DRIVER)
#   -> MAX runtime query (DeviceContext.name / get_attribute / get_api_version)
#   -> vendor library loaded at run time (gpuinfo.py: libcuda / NVML / libamdhip64)
#   -> placeholder plus a note.

from max.gpu.host import DeviceContext, DeviceAttribute
from std.collections import List
from std.os import getenv
from std.os.path import dirname, exists
from std.python import Python, PythonObject
from std.reflection import source_location
from std.sys import argv
from std.sys.defines import get_defined_string
from std.sys import default_accelerator

comptime BACKEND = get_defined_string["BACKEND", ""]()


def check_backend():
    comptime assert BACKEND == "cuda" or BACKEND == "hip", (
        "select exactly one backend when compiling: -D BACKEND=cuda or"
        " -D BACKEND=hip"
    )
    comptime assert not (BACKEND == "cuda" and default_accelerator().is_amd_gpu()), (
        "BACKEND=cuda but the target accelerator is AMD; compile with"
        " -D BACKEND=hip"
    )
    comptime assert not (BACKEND == "hip" and default_accelerator().is_nvidia_gpu()), (
        "BACKEND=hip but the target accelerator is NVIDIA; compile with"
        " -D BACKEND=cuda"
    )


def backend_api() -> String:
    return String(BACKEND)


def backend_label() -> String:
    comptime if BACKEND == "cuda":
        return "CUDA"
    else:
        return "HIP"


def compute_unit_label() -> String:
    comptime if BACKEND == "cuda":
        return "SMs"
    else:
        return "CUs"


# ===----------------------------------------------------------------------=== #
# Python helpers (energy.py, gpuinfo.py)
# ===----------------------------------------------------------------------=== #


def project_dir() -> String:
    """Directory holding the sources (and energy.py / gpuinfo.py)."""
    return dirname(String(source_location().file_name()))


def add_python_paths() raises:
    """Make energy.py and gpuinfo.py importable: the source directory, the
    binary's directory and its parent."""
    var src = project_dir()
    if src.byte_length() > 0:
        Python.add_to_path(src)
    var exe_dir = dirname(String(argv()[0]))
    if exe_dir.byte_length() > 0:
        Python.add_to_path(exe_dir)
        Python.add_to_path(dirname(exe_dir))


def _gpuinfo() raises -> PythonObject:
    add_python_paths()
    return Python.import_module("gpuinfo")


def _vendor_int(func: String, index: Int) -> Int:
    try:
        var m = _gpuinfo()
        return Int(Int64(py=m.__getattr__(func)(
            PythonObject(backend_api()), PythonObject(Int64(index))
        )))
    except:
        return -1


def _vendor_str(func: String, index: Int) -> String:
    try:
        var m = _gpuinfo()
        return String(py=m.__getattr__(func)(
            PythonObject(backend_api()), PythonObject(Int64(index))
        ))
    except:
        return ""


# ===----------------------------------------------------------------------=== #
# Device list
# ===----------------------------------------------------------------------=== #


def device_count() -> Int:
    return DeviceContext.number_of_devices(api=backend_api())


def open_device(index: Int) raises -> DeviceContext:
    var count = device_count()
    if count <= 0:
        raise Error("no " + backend_label() + " devices found")
    if index >= count:
        raise Error(
            "device index "
            + String(index)
            + " out of range ("
            + String(count)
            + " "
            + backend_label()
            + " device(s); see --list)"
        )
    try:
        return DeviceContext(device_id=index, api=backend_api())
    except e:
        raise Error(
            "cannot open "
            + backend_label()
            + " device "
            + String(index)
            + ": "
            + String(e)
        )


def _runtime_name(index: Int) -> String:
    try:
        return DeviceContext(device_id=index, api=backend_api()).name()
    except:
        return ""


def list_devices() raises:
    var count = device_count()
    if count <= 0:
        raise Error("no " + backend_label() + " devices found")
    for i in range(count):
        var name = _runtime_name(i)
        if name.byte_length() == 0:
            name = _vendor_str("device_name", i)
        if name.byte_length() == 0:
            name = "(unavailable)"
        print(String(i) + ": " + name, flush=True)


# ===----------------------------------------------------------------------=== #
# Device facts
# ===----------------------------------------------------------------------=== #


struct DeviceInfo(Movable):
    var name: String
    var driver: String
    var sm_count: Int  # SMs (CUDA) or CUs (HIP); fallback value when unknown
    var sm_known: Bool
    var max_threads_per_block: Int  # -1 if unknown
    var warp_size: Int  # -1 if unknown
    var max_grid_x: Int  # -1 if unknown
    var notes: List[String]

    def __init__(out self):
        self.name = ""
        self.driver = ""
        self.sm_count = 0
        self.sm_known = False
        self.max_threads_per_block = -1
        self.warp_size = -1
        self.max_grid_x = -1
        self.notes = List[String]()


def _attr(ctx: DeviceContext, attr: DeviceAttribute) -> Int:
    try:
        return ctx.get_attribute(attr)
    except:
        return -1


def _version_string(v: Int) -> String:
    comptime if BACKEND == "cuda":
        # cuDriverGetVersion: 1000 * major + 10 * minor (12040 -> 12.4)
        return "CUDA " + String(v // 1000) + "." + String((v % 1000) // 10)
    else:
        # HIP_VERSION: major * 10^7 + minor * 10^5 + patch (60342134 -> 6.3.42134)
        return (
            "HIP "
            + String(v // 10000000)
            + "."
            + String((v // 100000) % 100)
            + "."
            + String(v % 100000)
        )


def _api_version(ctx: DeviceContext) -> Int:
    try:
        return ctx.get_api_version()
    except:
        return -1


def _driver(ctx: DeviceContext, index: Int) -> String:
    var v = _api_version(ctx)
    if v <= 0:
        v = _vendor_int("driver_version", index)
    var api = _version_string(v) if v > 0 else String("")

    comptime if BACKEND == "cuda":
        var nvml = _vendor_str("nvml_driver", index)
        if nvml.byte_length() > 0 and api.byte_length() > 0:
            return nvml + " (" + api + ")"
        if nvml.byte_length() > 0:
            return nvml
    if api.byte_length() > 0:
        return api
    return "unknown"


def query_device(ctx: DeviceContext, index: Int) -> DeviceInfo:
    var info = DeviceInfo()

    # Name
    var name_env = getenv("GPU_NAME")
    if name_env.byte_length() > 0:
        info.name = name_env
    else:
        info.name = ctx.name()
        if info.name.byte_length() == 0:
            info.name = _vendor_str("device_name", index)
        if info.name.byte_length() == 0:
            info.name = backend_label() + " device " + String(index)
            info.notes.append(
                "device name unavailable; set GPU_NAME to report it"
            )

    # Driver
    var driver_env = getenv("GPU_DRIVER")
    if driver_env.byte_length() > 0:
        info.driver = driver_env
    else:
        info.driver = _driver(ctx, index)

    # SM / CU count
    var sms = _attr(ctx, DeviceAttribute.MULTIPROCESSOR_COUNT)
    if sms <= 0:
        sms = _vendor_int("sm_count", index)
    if sms > 0:
        info.sm_count = sms
        info.sm_known = True
    else:
        info.sm_count = 64
        info.sm_known = False

    info.max_threads_per_block = _attr(ctx, DeviceAttribute.MAX_THREADS_PER_BLOCK)
    info.warp_size = _attr(ctx, DeviceAttribute.WARP_SIZE)
    info.max_grid_x = _attr(ctx, DeviceAttribute.MAX_GRID_DIM_X)
    return info^
