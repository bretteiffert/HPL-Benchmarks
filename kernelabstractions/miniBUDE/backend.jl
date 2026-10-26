# backend.jl — backend selection, device list and device facts (keep byte-identical
# across benchmarks).
#
# The backend is fixed by the Julia environment, not by a flag:
#   julia --project=env/cuda ...   -> CUDA.jl   (backend "CUDA")
#   julia --project=env/hip  ...   -> AMDGPU.jl (backend "HIP")
# An environment with both or neither package is rejected.
#
# Device facts resolve in order: GPU_NAME / GPU_DRIVER override -> runtime query
# -> placeholder plus a note.

module Backend

using TOML

export BACKEND, backend_error, backend_label, ka_backend, device_count, device_names,
       select_device!, device_name, device_driver, compute_unit_count, compute_unit_label,
       max_workgroup_size, max_groups, warp_size, supports_half, current_stream,
       elapsed_seconds, device_sync, timing_method

function _detect()
    proj = Base.active_project()
    deps = Dict{String,Any}()
    if proj !== nothing && isfile(proj)
        deps = get(TOML.parsefile(proj), "deps", deps)
    end
    has_cuda = haskey(deps, "CUDA")
    has_hip = haskey(deps, "AMDGPU")
    where_ = proj === nothing ? "no active project" : proj
    if has_cuda && has_hip
        return :none, "the active environment ($where_) has both CUDA and AMDGPU; " *
                      "use exactly one: --project=env/cuda or --project=env/hip"
    elseif !has_cuda && !has_hip
        return :none, "the active environment ($where_) has neither CUDA nor AMDGPU; " *
                      "use --project=env/cuda or --project=env/hip"
    end
    return (has_cuda ? :cuda : :hip), ""
end

const BACKEND, _BACKEND_ERROR = _detect()

"Empty when a backend was selected, otherwise why not."
backend_error() = _BACKEND_ERROR

if BACKEND === :cuda
    @eval using CUDA
elseif BACKEND === :hip
    @eval using AMDGPU
end

# Vendor code is defined through @eval so the macros (CUDA.@elapsed,
# AMDGPU.@elapsed) are only expanded for the package that was loaded.
if BACKEND === :cuda
    @eval begin
        backend_label() = "CUDA"
        compute_unit_label() = "SMs"
        ka_backend() = CUDA.CUDABackend()
        _functional() = CUDA.functional()
        device_count() = _functional() ? length(CUDA.devices()) : 0
        device_names() = [CUDA.name(d) for d in CUDA.devices()]
        select_device!(i::Integer) = (CUDA.device!(i); nothing)

        _raw_name() = CUDA.name(CUDA.device())
        function _raw_driver()
            cuda = CUDA.driver_version()
            cuda_s = "CUDA $(cuda.major).$(cuda.minor)"
            CUDA.has_nvml() || return cuda_s
            # Raw string (VersionNumber would drop leading zeros, e.g. 580.65.06).
            buf = zeros(Cchar, 96)
            nvml = GC.@preserve buf begin
                CUDA.NVML.nvmlSystemGetDriverVersion(pointer(buf), length(buf))
                unsafe_string(pointer(buf))
            end
            return isempty(nvml) ? cuda_s : "$nvml ($cuda_s)"
        end
        _raw_cu_count() = Int(CUDA.attribute(CUDA.device(),
                                             CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))

        max_workgroup_size() = Int(CUDA.attribute(CUDA.device(),
                                                  CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
        max_groups() = Int(CUDA.attribute(CUDA.device(),
                                          CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X))
        warp_size() = Int(CUDA.warpsize(CUDA.device()))
        # Float16 arithmetic needs compute capability 5.3 or newer.
        supports_half() = CUDA.capability(CUDA.device()) >= v"5.3"

        current_stream() = CUDA.stream()
        elapsed_seconds(f) = Float64(CUDA.@elapsed f())
        device_sync() = CUDA.synchronize()
    end
elseif BACKEND === :hip
    @eval begin
        backend_label() = "HIP"
        compute_unit_label() = "CUs"
        ka_backend() = AMDGPU.ROCBackend()
        _functional() = AMDGPU.functional(:hip)
        device_count() = _functional() ? length(AMDGPU.devices()) : 0
        device_names() = [AMDGPU.HIP.name(d) for d in AMDGPU.devices()]
        select_device!(i::Integer) = (AMDGPU.device!(AMDGPU.devices()[i+1]); nothing)

        _raw_name() = AMDGPU.HIP.name(AMDGPU.device())
        function _raw_driver()
            v = Ref{Cint}()
            AMDGPU.HIP.hipDriverGetVersion(v)
            major = v[] ÷ 10_000_000
            minor = (v[] ÷ 100_000) % 100
            patch = v[] % 100_000
            return "HIP $major.$minor.$patch"
        end
        _raw_cu_count() = Int(AMDGPU.HIP.properties(AMDGPU.device()).multiProcessorCount)

        max_workgroup_size() = Int(AMDGPU.HIP.properties(AMDGPU.device()).maxThreadsPerBlock)
        max_groups() = Int(AMDGPU.HIP.properties(AMDGPU.device()).maxGridSize[1])
        warp_size() = Int(AMDGPU.HIP.wavefrontsize(AMDGPU.device()))
        supports_half() = true

        current_stream() = AMDGPU.stream()
        elapsed_seconds(f) = Float64(AMDGPU.@elapsed f())
        device_sync() = AMDGPU.synchronize()
    end
end

timing_method() = "device timestamps"

_override(var) = (v = strip(get(ENV, var, "")); isempty(v) ? nothing : String(v))

"(name, note) — note is empty unless a placeholder was used."
function device_name(index::Integer)
    o = _override("GPU_NAME")
    o === nothing || return o, ""
    try
        return _raw_name(), ""
    catch
        return "$(backend_label()) device $index",
               "could not query the device name; set GPU_NAME to override"
    end
end

"(driver, note)"
function device_driver()
    o = _override("GPU_DRIVER")
    o === nothing || return o, ""
    try
        return _raw_driver(), ""
    catch
        return "unknown", "could not query the driver version; set GPU_DRIVER to override"
    end
end

"SM/CU count, or `nothing` if the runtime cannot report it."
function compute_unit_count()
    try
        n = _raw_cu_count()
        return n > 0 ? n : nothing
    catch
        return nothing
    end
end

end # module Backend
