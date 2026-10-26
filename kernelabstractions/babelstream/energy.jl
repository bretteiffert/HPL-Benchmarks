# energy.jl — board-power measurement shared by every benchmark (keep byte-identical).
#
# Loads ../../dependencies/energy.jl (module `Energy`, a ccall wrapper around
# libkernelenergy.so) relative to the project root, i.e. <project>/../../dependencies.
# The dependency file is only included if it exists, and libkernelenergy.so is only
# opened when a measurement is requested, so `--no-energy` runs need neither.
#
# Usage from the run loop:
#
#     res = measure_energy(device_index, stream, iterations) do
#         for it in 1:iterations ... timed launches ... end
#     end
#
# `stream` must be the stream the kernels are launched on (CUDA.stream() /
# AMDGPU.stream()); CUDA.jl task streams are non-blocking, so timing on the
# NULL stream would not wait for the kernels.

module BoardEnergy

export EnergyResult, EnergyError, measure_energy, energy_available,
       joules_per_iter, total_wh

const DEPS_DIR = normpath(joinpath(@__DIR__, "..", "..", "dependencies"))
const ENERGY_JL = joinpath(DEPS_DIR, "energy.jl")

if isfile(ENERGY_JL)
    include(ENERGY_JL)          # defines submodule `Energy`
end

energy_available() = isdefined(@__MODULE__, :Energy)

struct EnergyError <: Exception
    msg::String
end
Base.showerror(io::IO, e::EnergyError) = print(io, e.msg)

"Board energy over the timed iterations only."
struct EnergyResult
    joules::Float64       # integrated over the window
    time_s::Float64       # measured window length (GPU events), seconds
    avg_power_w::Float64
    iterations::Int
    samples::Int
end

joules_per_iter(r::EnergyResult) = r.joules / r.iterations
total_wh(r::EnergyResult) = r.joules / 3600.0

_hint() = " (try '--no-energy')"

"""
    measure_energy(f, device, stream, iterations) -> EnergyResult

Start the profiler on `device` / `stream`, run `f()` (the timed iterations
only — no warm-up), stop it and return the totals. Any failure to load or
start the profiler, or a missing measurement, throws `EnergyError`.
"""
function measure_energy(f, device::Integer, stream, iterations::Integer)
    iterations >= 1 || throw(ArgumentError("iterations must be >= 1"))
    energy_available() || throw(EnergyError(
        "energy profiler not found: $(ENERGY_JL)" * _hint()))

    prof = try
        Energy.EnergyProfiler(gpu_id = device, stream = stream)
    catch e
        throw(EnergyError("could not start the energy profiler on device $device: " *
                          sprint(showerror, e) * _hint()))
    end

    try
        Energy.start!(prof)
        f()
        Energy.stop!(prof)

        j  = Energy.joules(prof)
        ms = Energy.elapsed_ms(prof)
        w  = Energy.avg_power_w(prof)
        n  = Energy.sample_count(prof)

        (isfinite(ms) && ms > 0) || throw(EnergyError(
            "energy profiler returned no measured time ($(ms) ms)" * _hint()))
        (isfinite(j) && j >= 0) || throw(EnergyError(
            "energy profiler returned an invalid energy ($(j) J)" * _hint()))

        return EnergyResult(j, ms * 1e-3, isfinite(w) ? w : j / (ms * 1e-3),
                            Int(iterations), n)
    finally
        Energy.close!(prof)
    end
end

end # module BoardEnergy
