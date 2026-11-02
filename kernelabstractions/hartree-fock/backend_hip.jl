# HIP backend (AMDGPU.jl): device list, device facts and the device timer.
# AMDGPU.jl calls libamdhip64 directly, so the "runtime" and "vendor library"
# steps of the lookup are the same calls here: hipDeviceGetName,
# hipDeviceGetAttribute(hipDeviceAttributeMultiprocessorCount = 63),
# hipDriverGetVersion.

import AMDGPU
import AMDGPU: HIP

const BACKEND = AMDGPU.ROCBackend()

backend_name() = "HIP"

compute_unit_label() = "CUs"

function hip_available()
    AMDGPU.functional(:hip) ||
        error("HIP hipGetDeviceCount: the HIP runtime (libamdhip64) was not found; check the ROCm installation (ROCM_PATH)")
    return true
end

# Number of visible devices. Throws on runtime failure.
function device_count()::Int
    hip_available()
    try
        return KI.ndevices(BACKEND)
    catch e
        error("HIP hipGetDeviceCount: " * error_text(e))
    end
end

hip_device(index::Int) = AMDGPU.devices()[index + 1]

function hip_name(index::Int)::String
    try
        return String(HIP.name(hip_device(index)))
    catch
        return ""
    end
end

# Index and name of every visible device.
function list_devices()::Vector{Tuple{Int, String}}
    out = Tuple{Int, String}[]
    for i in 0:device_count()-1
        name = hip_name(i)
        isempty(name) && (name = "HIP device " * string(i))
        push!(out, (i, name))
    end
    return out
end

# 60342134 -> "6.3.42134" (major * 10^7 + minor * 10^5 + patch)
hip_version_string(v::Integer) =
    string(v ÷ 10000000) * "." * string((v ÷ 100000) % 100) * "." * string(v % 100000)

function query_device!(info::DeviceInfo, index::Int)
    info.half_supported = true    # fp16 arithmetic is native on every ROCm-supported GPU

    # Name
    info.name = env_or_empty("GPU_NAME")
    isempty(info.name) && (info.name = hip_name(index))
    if isempty(info.name)
        info.name = "HIP device " * string(index)
        push!(info.notes, "device name unavailable from the HIP runtime (libamdhip64); set GPU_NAME to override")
    end

    # CU count (KernelInterface asks hipDeviceGetAttribute 63; ask again if it gave nothing)
    if info.compute_units <= 0
        info.compute_units = try
            Int(HIP.attribute(hip_device(index), HIP.hipDeviceAttributeMultiprocessorCount))
        catch
            0
        end
    end

    # Driver
    info.driver = env_or_empty("GPU_DRIVER")
    if isempty(info.driver)
        v = Ref{Cint}(0)
        ok = try
            HIP.hipDriverGetVersion(v)
            true
        catch
            false
        end
        info.driver = (ok && v[] > 0) ? "HIP " * hip_version_string(v[]) : "unknown"
    end
    return info
end

# Whether the device can do atomic add on Float64 (needed for --double).
# atomicAdd on double is available on every ROCm-supported GPU.
fp64_atomic_add_supported(index::Int) = true

# ---- device timer: HIP events on the task's stream (the one KA launches on) --

struct DeviceTimer
    start::HIP.HIPEvent
    stop::HIP.HIPEvent
end

function make_timer()
    s = AMDGPU.stream()
    return DeviceTimer(HIP.HIPEvent(s; do_record = false, timing = true),
                       HIP.HIPEvent(s; do_record = false, timing = true))
end

# Records start, runs f (the launch), records stop, waits for it and returns
# the device-measured seconds: (double)ms * 1e-3, as the HIP version.
function time_launch(t::DeviceTimer, f)::Float64
    HIP.record(t.start)
    f()
    HIP.record(t.stop)
    HIP.wait(t.stop)                              # hipEventSynchronize
    ms = Ref{Cfloat}(0.0f0)
    HIP.hipEventElapsedTime(ms, t.start, t.stop)
    return Float64(ms[]) * 1.0e-3
end
