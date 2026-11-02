# CUDA backend (CUDA.jl): device list, device facts and the device timer.
# CUDA.jl calls libcuda and NVML directly, so the "runtime" and "vendor
# library" steps of the lookup are the same calls here: cuDeviceGetName,
# cuDeviceGetAttribute(MULTIPROCESSOR_COUNT = 16), nvmlSystemGetDriverVersion
# and cuDriverGetVersion.

import CUDA
const CC = CUDA.CUDACore

const BACKEND = CUDA.CUDABackend()

backend_name() = "CUDA"

compute_unit_label() = "SMs"

# Number of visible devices. Throws on runtime failure.
function device_count()::Int
    try
        return KI.ndevices(BACKEND)
    catch e
        error("CUDA cuDeviceGetCount: " * error_text(e))
    end
end

function cuda_name(index::Int)::String
    try
        return String(CC.name(CC.CuDevice(index)))
    catch
        return ""
    end
end

# Index and name of every visible device.
function list_devices()::Vector{Tuple{Int, String}}
    out = Tuple{Int, String}[]
    for i in 0:device_count()-1
        name = cuda_name(i)
        isempty(name) && (name = "CUDA device " * string(i))
        push!(out, (i, name))
    end
    return out
end

function nvml_driver_string()::String
    try
        CUDA.has_nvml() || return ""
        buf = Vector{Cchar}(undef, CUDA.NVML.NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
        CUDA.NVML.nvmlSystemGetDriverVersion(pointer(buf), length(buf))
        return GC.@preserve buf unsafe_string(pointer(buf))
    catch
        return ""
    end
end

function query_device!(info::DeviceInfo, index::Int)
    dev = CC.CuDevice(index)
    cap = try CC.capability(dev) catch; v"0.0" end
    info.half_supported = cap >= v"5.3"

    # Name
    info.name = env_or_empty("GPU_NAME")
    isempty(info.name) && (info.name = cuda_name(index))
    if isempty(info.name)
        info.name = "CUDA device " * string(index)
        push!(info.notes, "device name unavailable from the CUDA driver; set GPU_NAME to override")
    end

    # SM count (KernelInterface asks cuDeviceGetAttribute 16; ask again if it gave nothing)
    if info.compute_units <= 0
        info.compute_units = try
            Int(CC.attribute(dev, CC.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
        catch
            0
        end
    end

    # Driver: "550.54.15 (CUDA 12.4)"
    info.driver = env_or_empty("GPU_DRIVER")
    if isempty(info.driver)
        cuda = try
            v = CC.driver_version()
            "CUDA " * string(v.major) * "." * string(v.minor)
        catch
            ""
        end
        nv = nvml_driver_string()
        info.driver = !isempty(nv) && !isempty(cuda) ? nv * " (" * cuda * ")" :
                      !isempty(nv) ? nv : !isempty(cuda) ? cuda : "unknown"
    end
    return info
end

# Whether the device has native atomic add on Float64 (compute capability
# 6.0+, as CUDA C's atomicAdd(double*)). Needed for --double.
function fp64_atomic_add_supported(index::Int)
    cap = try CC.capability(CC.CuDevice(index)) catch; v"0.0" end
    return cap >= v"6.0"
end

# ---- device timer: CUDA events on the task's stream (the one KA launches on) --

struct DeviceTimer
    start::CC.CuEvent
    stop::CC.CuEvent
end

make_timer() = DeviceTimer(CC.CuEvent(), CC.CuEvent())    # EVENT_DEFAULT: timing enabled

# Records start, runs f (the launch), records stop, waits for it and returns
# the device-measured seconds: (double)ms * 1e-3, as the CUDA C version.
function time_launch(t::DeviceTimer, f)::Float64
    CC.record(t.start, CC.stream())
    f()
    CC.record(t.stop, CC.stream())
    CC.synchronize(t.stop)
    ms = Ref{Cfloat}(0.0f0)
    if CC.driver_version() >= v"12.8"
        CC.cuEventElapsedTime_v2(ms, t.start, t.stop)
    else
        CC.cuEventElapsedTime(ms, t.start, t.stop)
    end
    return Float64(ms[]) * 1.0e-3
end
