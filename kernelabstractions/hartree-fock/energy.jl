# Board-power measurement through libkernelenergy (../../dependencies).
# Shared unchanged by every benchmark in this suite. Sampling covers only the
# span between start! and stop!, which callers place around the timed
# iterations. The library is loaded at run time from
# ../../dependencies/libkernelenergy.so next to this folder, and only when
# energy is enabled.

using Libdl

Base.@kwdef mutable struct EnergyResult
    enabled::Bool = false
    joules::Float64 = 0.0
    joules_per_iter::Float64 = 0.0
    total_wh::Float64 = 0.0
    avg_power_w::Float64 = 0.0
    seconds::Float64 = 0.0         # wall time between start! and stop!
    iterations::UInt = 0
    samples::Int = 0
end

mutable struct EnergyMeter
    lib::Ptr{Cvoid}
    handle::Ptr{Cvoid}
    f_start::Ptr{Cvoid}
    f_stop::Ptr{Cvoid}
    f_destroy::Ptr{Cvoid}
    f_joules::Ptr{Cvoid}
    f_samples::Ptr{Cvoid}
    running::Bool
    t0::UInt64
    seconds::Float64
end

energy_library_path() = normpath(joinpath(@__DIR__, "..", "..", "dependencies", "libkernelenergy.so"))

# enabled == false gives a meter that does nothing (--no-energy).
# Throws if the profiler cannot be created.
function EnergyMeter(device::Int, enabled::Bool)
    m = EnergyMeter(C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, false, 0, 0.0)
    enabled || return m

    path = energy_library_path()
    lib = Libdl.dlopen(path, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL; throw_error = false)
    if lib === nothing
        error("could not load the board-power profiler library " * path * " (run with --no-energy to skip it)")
    end
    sym(name) = Libdl.dlsym(lib, name)
    f_create = sym(:energy_create)
    m.lib = lib
    m.f_start = sym(:energy_start)
    m.f_stop = sym(:energy_stop)
    m.f_destroy = sym(:energy_destroy)
    m.f_joules = sym(:energy_joules)
    m.f_samples = sym(:energy_sample_count)

    m.handle = ccall(f_create, Ptr{Cvoid}, (Cint, Ptr{Cvoid}), device, C_NULL)
    if m.handle == C_NULL
        error("could not start the board-power profiler on device " * string(device) *
              " (run with --no-energy to skip it)")
    end
    finalizer(close_meter!, m)
    return m
end

function close_meter!(m::EnergyMeter)
    if m.handle != C_NULL
        m.running && ccall(m.f_stop, Cvoid, (Ptr{Cvoid},), m.handle)
        ccall(m.f_destroy, Cvoid, (Ptr{Cvoid},), m.handle)
        m.handle = C_NULL
        m.running = false
    end
    return nothing
end

enabled(m::EnergyMeter) = m.handle != C_NULL

function start!(m::EnergyMeter)
    m.t0 = time_ns()                      # monotonic, like std::chrono::steady_clock
    m.handle != C_NULL && ccall(m.f_start, Cvoid, (Ptr{Cvoid},), m.handle)
    m.running = true
    return nothing
end

function stop!(m::EnergyMeter)
    m.running || return nothing
    m.handle != C_NULL && ccall(m.f_stop, Cvoid, (Ptr{Cvoid},), m.handle)
    m.seconds = (time_ns() - m.t0) * 1.0e-9
    m.running = false
    return nothing
end

# Totals for the sampled span, divided over `iterations`.
function result(m::EnergyMeter, iterations::Integer)::EnergyResult
    r = EnergyResult()
    r.enabled = enabled(m)
    r.seconds = m.seconds
    r.iterations = UInt(iterations)
    r.enabled || return r

    r.joules = ccall(m.f_joules, Cdouble, (Ptr{Cvoid},), m.handle)
    r.samples = Int(ccall(m.f_samples, Cint, (Ptr{Cvoid},), m.handle))
    r.joules_per_iter = iterations > 0 ? r.joules / Float64(iterations) : 0.0
    r.total_wh = r.joules / 3600.0
    r.avg_power_w = m.seconds > 0.0 ? r.joules / m.seconds : 0.0
    return r
end
