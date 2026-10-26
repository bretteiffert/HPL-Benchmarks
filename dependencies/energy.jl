# energy.jl — Julia wrapper around the GPU energy-profiler C API (libkernelenergy.so).
#
# Julia `ccall` binding to the same `energy_*` functions declared in
# `kernel_energy_capi.h` that `energy.py` wraps with ctypes — the identical flat
# C surface over the C++ EnergyProfiler.  The C profiler samples instantaneous
# board power on a background thread between `start!` / `stop!` and integrates it
# against wall-clock time to report joules; elapsed time is measured with GPU
# events on the given stream.
#
# Build the shared library first:
#
#     dslbench/build_energy.sh            # --cuda (NVML) auto on NVIDIA, --hip on ROCm
#
# Manual use mirroring the C sequence:
#
#     include("energy.jl"); using .Energy
#     prof = EnergyProfiler(gpu_id=0)
#     try
#         start!(prof)
#         for _ in 1:iters; launch_kernel(...); end
#         stop!(prof)
#         @show joules(prof) elapsed_ms(prof) avg_power_w(prof)
#     finally
#         close!(prof)
#     end
#
# Or the do-block form (auto-closes, like Python's `with`):
#
#     EnergyProfiler(gpu_id=0) do prof
#         start!(prof); ...; stop!(prof)
#     end
#
# Or let `profile` run the warmup + timed loop and return a metrics NamedTuple:
#
#     m = profile(() -> launch_kernel(...), 100; ops = 2*M*N*K,
#                 sync = () -> CUDA.synchronize())
#     @show m.joules_per_iter m.avg_power_w m.tops_per_watt

module Energy

using Libdl

export EnergyProfiler, load_lib,
       start!, stop!, close!,
       joules, elapsed_ms, avg_power_w, sample_count,
       display_report, metrics, profile

const _DEFAULT_LIB = joinpath(@__DIR__, "libkernelenergy.so")

# One cached dlopen handle per resolved library path (ccall's library argument
# cannot be a runtime variable, so we dlopen + dlsym and call function pointers).
const _LIB_CACHE = Dict{String,Ptr{Cvoid}}()

"""
    load_lib(lib_path=nothing) -> Ptr{Cvoid}

Resolve the path to `libkernelenergy.so`, `dlopen` it (cached), and return the
library handle for `dlsym`.  Override with the `lib_path` argument or the
`DSLBENCH_ENERGY_LIB` environment variable.  Throws a helpful error (pointing at
build_energy.sh) if the library is missing — mirroring `energy.py.load_lib`.
"""
function load_lib(lib_path=nothing)
    path = something(lib_path, get(ENV, "DSLBENCH_ENERGY_LIB", _DEFAULT_LIB))
    path = abspath(path)
    haskey(_LIB_CACHE, path) && return _LIB_CACHE[path]
    isfile(path) || error(
        "energy library not found: $path\n" *
        "Build it first:  dslbench/build_energy.sh   (--cuda or --hip)")
    handle = Libdl.dlopen(path)
    _LIB_CACHE[path] = handle
    return handle
end

"""
    EnergyProfiler(; gpu_id=0, stream=C_NULL, lib_path=nothing)
    EnergyProfiler(f::Function; kwargs...)

Handle to a C `EnergyProfiler` for one GPU.

- `gpu_id`  : GPU / SMI device index to sample (see the `SQUID_*_DEV` env-var
  overrides in `kernel.energy.h` for visibility-mask corner cases).
- `stream`  : GPU stream to time on.  `C_NULL`/`0` is the default stream.  Accepts
  a `Ptr`, an integer pointer value, or any object with a `.handle` field (e.g.
  a CUDA.jl `CuStream` — pass `CUDA.stream()`), so this file needs no CUDA
  dependency of its own.
- `lib_path`: path to libkernelenergy.so (default: alongside this file, or
  `\$DSLBENCH_ENERGY_LIB`).

The function form runs `f(prof)` and closes the handle afterward (like Python's
`with EnergyProfiler(...) as prof:`).
"""
mutable struct EnergyProfiler
    lib::Ptr{Cvoid}     # dlopen handle (for dlsym)
    h::Ptr{Cvoid}       # opaque C EnergyProfiler*

    function EnergyProfiler(; gpu_id::Integer=0, stream=C_NULL, lib_path=nothing)
        lib = load_lib(lib_path)
        h = ccall(Libdl.dlsym(lib, :energy_create), Ptr{Cvoid},
                  (Cint, Ptr{Cvoid}), Cint(gpu_id), _as_stream_ptr(stream))
        h == C_NULL && error("energy_create returned NULL (allocation failed)")
        prof = new(lib, h)
        finalizer(close!, prof)
        return prof
    end
end

# Coerce a stream argument to a raw pointer without hard-depending on CUDA.jl or
# AMDGPU.jl.  The C API takes an opaque void* that is a CUstream on NVIDIA and a
# hipStream_t on AMD — both pointers — so we duck-type the handle out of whatever
# stream object is passed: CUDA.jl's `CuStream.handle` or AMDGPU.jl's
# `HIPStream.stream` (and plain Ptr / integer pointer values directly).
_as_stream_ptr(s::Ptr) = Ptr{Cvoid}(s)
_as_stream_ptr(s::Integer) = Ptr{Cvoid}(UInt(s))
function _as_stream_ptr(s)
    for f in (:handle, :stream)          # CuStream.handle / HIPStream.stream
        if hasproperty(s, f)
            return Ptr{Cvoid}(UInt(getproperty(s, f)))
        end
    end
    error("don't know how to get a stream pointer from a $(typeof(s)); " *
          "pass a Ptr, an integer, or a CUDA.jl/AMDGPU.jl stream")
end

function EnergyProfiler(f::Function; kwargs...)
    prof = EnergyProfiler(; kwargs...)
    try
        return f(prof)
    finally
        close!(prof)
    end
end

# -- lifecycle ----------------------------------------------------------------

"Sync the device and begin timing + power sampling."
function start!(p::EnergyProfiler)
    _check(p)
    ccall(Libdl.dlsym(p.lib, :energy_start), Cvoid, (Ptr{Cvoid},), p.h)
    return p
end

"Record the end event, drain the stream, and integrate energy."
function stop!(p::EnergyProfiler)
    _check(p)
    ccall(Libdl.dlsym(p.lib, :energy_stop), Cvoid, (Ptr{Cvoid},), p.h)
    return p
end

"Release the underlying C handle (idempotent)."
function close!(p::EnergyProfiler)
    if p.h != C_NULL
        ccall(Libdl.dlsym(p.lib, :energy_destroy), Cvoid, (Ptr{Cvoid},), p.h)
        p.h = C_NULL
    end
    return nothing
end

# -- results (valid after stop!(), over the whole timed window) ---------------

"Integrated energy over the timed window, in joules."
function joules(p::EnergyProfiler)
    _check(p)
    return ccall(Libdl.dlsym(p.lib, :energy_joules), Cdouble, (Ptr{Cvoid},), p.h)
end

"GPU elapsed time over the timed window, in milliseconds."
function elapsed_ms(p::EnergyProfiler)
    _check(p)
    return ccall(Libdl.dlsym(p.lib, :energy_elapsed_ms), Cdouble, (Ptr{Cvoid},), p.h)
end

"Average board power over the timed window (joules / seconds), in watts."
function avg_power_w(p::EnergyProfiler)
    _check(p)
    return ccall(Libdl.dlsym(p.lib, :energy_avg_power_w), Cdouble, (Ptr{Cvoid},), p.h)
end

"Number of power samples taken during the timed window."
function sample_count(p::EnergyProfiler)
    _check(p)
    return Int(ccall(Libdl.dlsym(p.lib, :energy_sample_count), Cint, (Ptr{Cvoid},), p.h))
end

"""
    display_report(p; ops=0, iterations=1)

Print the C profiler's human-readable report (energy/time/power/TOPS/
TOPS-per-watt), dividing totals by `iterations`.  (Named `display_report` rather
than `display` to avoid shadowing `Base.display`.)
"""
function display_report(p::EnergyProfiler; ops::Real=0, iterations::Integer=1)
    _check(p)
    ccall(Libdl.dlsym(p.lib, :energy_display), Cvoid,
          (Ptr{Cvoid}, Cdouble, Cint), p.h, Cdouble(ops), Cint(iterations))
    return nothing
end

"""
    metrics(p; ops=0, iterations=1) -> NamedTuple

Per-iteration metrics (no printing).  `ops` is the total op count over the timed
window; per-iteration and TOPS / TOPS-per-watt figures divide by `iterations` to
match `display_report`.  Mirrors `energy.py.metrics`.
"""
function metrics(p::EnergyProfiler; ops::Real=0, iterations::Integer=1)
    _check(p)
    iters = max(Int(iterations), 1)
    j = joules(p)
    ms = elapsed_ms(p)
    power = avg_power_w(p)
    per_it_s = (ms / iters) * 1e-3
    tops = per_it_s > 0 ? ops / (per_it_s * 1e12) : 0.0
    return (
        joules = j,
        joules_per_iter = j / iters,
        elapsed_ms = ms,
        ms_per_iter = ms / iters,
        avg_power_w = power,
        tops = tops,
        tops_per_watt = power > 0 ? tops / power : 0.0,
        sample_count = sample_count(p),
        iterations = iters,
        ops = Float64(ops),
    )
end

# -- convenience --------------------------------------------------------------

"""
    profile(fn, iters; ops=0, warmup=3, sync=nothing) -> NamedTuple

Run `fn` `warmup`+`iters` times under the profiler and return `metrics`.

Mirrors the `ENERGY_PROFILE` C macro: `warmup` untimed calls, then a timed loop
of `iters` calls between `start!`/`stop!`.  `sync` (e.g. `() -> CUDA.synchronize()`)
is called once before starting so the warmup is fully drained; `start!`/`stop!`
also sync the stream.
"""
function profile(fn, iters::Integer; ops::Real=0, warmup::Integer=3, sync=nothing)
    EnergyProfiler() do p
        for _ in 1:max(warmup, 0); fn(); end
        sync === nothing || sync()
        start!(p)
        for _ in 1:iters; fn(); end
        stop!(p)
        return metrics(p; ops=ops, iterations=iters)
    end
end

# -- internal -----------------------------------------------------------------

function _check(p::EnergyProfiler)
    p.h == C_NULL && error("EnergyProfiler handle is closed")
    return nothing
end

end # module Energy

# Smoke test: profile an empty window (no kernels) — should return ~0 J and a
# nonzero idle power reading from the SMI sampler.  Run: julia energy.jl
if abspath(PROGRAM_FILE) == @__FILE__
    using .Energy
    gpu = parse(Int, get(ENV, "DSLBENCH_GPU", "0"))
    EnergyProfiler(gpu_id=gpu) do p
        start!(p)
        sleep(0.2)
        stop!(p)
        println("idle window: $(round(elapsed_ms(p); digits=2)) ms (GPU-event), ",
                "$(sample_count(p)) power samples, ",
                "$(round(avg_power_w(p); digits=1)) W avg, ",
                "$(round(joules(p); digits=4)) J")
    end
end
