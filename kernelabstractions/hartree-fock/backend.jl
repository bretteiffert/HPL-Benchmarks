# Backend selection (fixed by the Julia environment the program runs in:
# --project=env/cuda or --project=env/hip, each listing exactly one of CUDA.jl
# and AMDGPU.jl), the
# device list and device facts. Shared unchanged by every KernelAbstractions
# benchmark in this suite.
#
# Lookup order for every fact: override env var (GPU_NAME, GPU_DRIVER) ->
# KernelAbstractions/KernelInterface query (limits, SM/CU count) or the
# vendor runtime the Julia backend package wraps (name: cuDeviceGetName /
# hipDeviceGetName; driver: NVML + cuDriverGetVersion / hipDriverGetVersion)
# -> placeholder plus a note.
#
# Each backend file defines: BACKEND (the KernelAbstractions backend object),
# backend_name(), compute_unit_label(), device_count(), list_devices(),
# query_device!(info, index), fp64_atomic_add_supported(index), and the
# device timer (make_timer(), time_launch(timer, f)), which uses the vendor's
# timestamp events because KernelAbstractions has no timing API.

using KernelAbstractions
const KI = KernelAbstractions.KernelInterface

import TOML

# The backend is the one GPU package (CUDA or AMDGPU) listed in the active
# project. None or both is an error naming the two environments.
function selected_backend()::String
    proj = Base.active_project()
    deps = (proj === nothing || !isfile(proj)) ? Dict{String, Any}() :
           get(TOML.parsefile(proj), "deps", Dict{String, Any}())
    found = String[]
    haskey(deps, "CUDA") && push!(found, "cuda")
    haskey(deps, "AMDGPU") && push!(found, "hip")
    length(found) == 1 && return found[1]
    what = isempty(found) ? "no GPU backend in the active Julia environment" :
                            "both CUDA and AMDGPU in the active Julia environment"
    print(stderr, "error: ", what, "; run with julia --project=", joinpath(@__DIR__, "env", "cuda"),
          " or --project=", joinpath(@__DIR__, "env", "hip"), " (or use run.sh <cuda|hip>)\n")
    exit(1)
end

const BUILD_BACKEND = selected_backend()

Base.@kwdef mutable struct DeviceInfo
    index::Int = 0
    name::String = ""             # GPU_NAME > runtime > placeholder
    driver::String = ""           # GPU_DRIVER > runtime/vendor library > "unknown"
    compute_units::Int = 0        # SMs / CUs; 0 when unknown
    max_threads_per_block::Int = 0
    max_grid_x::Int = 0
    warp_size::Int = 0
    half_supported::Bool = false  # native fp16 arithmetic on this device
    notes::Vector{String} = String[]
end

env_or_empty(name::AbstractString) = get(ENV, name, "")

# First line of an exception's message, for "error: ..." lines.
function error_text(e)
    msg = e isa ErrorException ? e.msg : sprint(showerror, e)
    return String(strip(first(split(msg, '\n'))))
end

@static if BUILD_BACKEND == "cuda"
    include("backend_cuda.jl")
elseif BUILD_BACKEND == "hip"
    include("backend_hip.jl")
else
    error("unknown backend '" * BUILD_BACKEND * "'")
end

# Make the device current and collect its facts. Throws on runtime failure.
function query_device(index::Int)::DeviceInfo
    info = DeviceInfo(index = index)
    KernelAbstractions.device!(BACKEND, index + 1)     # KernelInterface ids are 1-based

    info.max_threads_per_block = KI.max_work_group_size(BACKEND)
    info.max_grid_x = KI.max_num_groups(BACKEND)[1]
    info.warp_size = KI.sub_group_size(BACKEND)
    info.compute_units = KI.multiprocessor_count(BACKEND)

    query_device!(info, index)                         # name, driver, fp16, fallbacks
    return info
end
