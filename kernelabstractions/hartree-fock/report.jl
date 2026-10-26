# report.jl — the Hartree-Fock report and its three formatters (text, --csv, --csv-raw).
# Formatters read only from the Report; nothing else prints results.

module Reports

using Printf
using ..BoardEnergy: EnergyResult, joules_per_iter, total_wh

export Report, Metric, print_header, print_text, print_csv, print_csv_raw,
       print_stderr_sections, IMPLEMENTATION, ROUTINE

const IMPLEMENTATION = "KernelAbstractions"
const ROUTINE = "Fock build"

"One correctness line: label and value (scientific notation)."
struct Metric
    label::String
    value::Float64
    digits::Int
end

Base.@kwdef mutable struct Report
    backend::String
    device::String
    driver::String
    precision::String             # float32 / float64
    sqrt_mode::String             # native / legacy-sqrtf
    input::String                 # basename of the input file
    natoms::Int
    ngauss::Int
    nnnn::Int                     # work items (unique integral quartets)
    block_size::Int
    num_times::Int
    timing::String
    energy_enabled::Bool
    header_notes::Vector{String} = String[]
    times::Vector{Float64} = Float64[]       # timed iterations only
    energy::Union{Nothing,EnergyResult} = nothing
    energy_notes::Vector{String} = String[]
    e2_energy::Float64 = NaN                 # 2e- energy, this precision end to end
    rel_err_ref::Float64 = NaN               # vs fp64 CPU reference (NaN = not checked)
    metrics::Vector{Metric} = Metric[]
    check_notes::Vector{String} = String[]
end

# ---------------------------------------------------------------- helpers

function thousands(n::Integer)
    s = string(abs(n))
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    pushfirst!(parts, s)
    return (n < 0 ? "-" : "") * join(parts, ",")
end

"Scientific notation without exponent padding: 1.230000e-1, 6.83e5."
function sci(x::Real, digits::Int)
    isfinite(x) || return string(x)
    s = Printf.format(Printf.Format("%.$(digits)e"), Float64(x))
    m, e = split(s, 'e')
    return "$(m)e$(parse(Int, e))"
end

fixed(x::Real, d::Int) = Printf.format(Printf.Format("%.$(d)f"), Float64(x))

csv_field(s::AbstractString) =
    any(c -> c in (',', '"', '\n', '\r'), s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : String(s)

stats(t::Vector{Float64}) = (minimum(t), maximum(t), sum(t) / length(t))

kv(io, k, v) = println(io, "  ", rpad(k, 13), v)

# ---------------------------------------------------------------- text

"Section 1: printed before the run starts."
function print_header(io::IO, r::Report)
    println(io, "Hartree-Fock (", IMPLEMENTATION, ")")
    kv(io, "Backend", r.backend)
    kv(io, "Device", r.device)
    kv(io, "Driver", r.driver)
    kv(io, "Precision", r.precision)
    kv(io, "Sqrt mode", r.sqrt_mode)
    kv(io, "Input", "$(r.input) ($(thousands(r.natoms)) atoms, $(thousands(r.ngauss)) Gaussians)")
    kv(io, "Quartets", "$(thousands(r.nnnn)) ($(thousands(cld(r.nnnn + 1, r.block_size))) workgroups)")
    kv(io, "Block size", string(r.block_size))
    kv(io, "Iterations", "$(r.num_times) (after 1 untimed run)")
    kv(io, "Timing", r.timing)
    kv(io, "Energy", r.energy_enabled ? "libkernelenergy (board power)" : "disabled (--no-energy)")
    for n in r.header_notes
        println(io, "  note: ", n)
    end
    flush(io)
end

function print_results(io::IO, r::Report)
    println(io, rpad("Routine", 12), lpad("Min (s)", 14), lpad("Max (s)", 14), lpad("Avg (s)", 14))
    tmin, tmax, tavg = stats(r.times)
    println(io, rpad(ROUTINE, 12), lpad(fixed(tmin, 8), 14), lpad(fixed(tmax, 8), 14),
            lpad(fixed(tavg, 8), 14))
end

function print_energy(io::IO, r::Report)
    if !r.energy_enabled
        println(io, "Energy  disabled (--no-energy)")
    elseif r.energy !== nothing
        e = r.energy
        println(io, rpad("Energy", 8), lpad("J/iter", 14), lpad("Total (Wh)", 14),
                lpad("Avg power (W)", 16), lpad("Time (s)", 12), lpad("Iterations", 13),
                lpad("Samples", 10))
        println(io, rpad("board", 8), lpad(fixed(joules_per_iter(e), 6), 14),
                lpad(fixed(total_wh(e), 6), 14), lpad(fixed(e.avg_power_w, 2), 16),
                lpad(fixed(e.time_s, 6), 12), lpad(thousands(e.iterations), 13),
                lpad(thousands(e.samples), 10))
    end
    for n in r.energy_notes
        println(io, "  note: ", n)
    end
end

function print_correctness(io::IO, r::Report)
    println(io, "Correctness (", r.precision, ", untimed run)")
    for m in r.metrics
        println(io, rpad(m.label, 46), lpad(sci(m.value, m.digits), 24))
    end
    for n in r.check_notes
        println(io, "  note: ", n)
    end
end

"Text mode: sections 2-4 (header already printed) on stdout."
function print_text(io::IO, r::Report)
    println(io)
    print_results(io, r)
    println(io)
    print_energy(io, r)
    println(io)
    print_correctness(io, r)
end

"CSV modes: energy and correctness go to stderr."
function print_stderr_sections(io::IO, r::Report)
    println(io)
    print_energy(io, r)
    println(io)
    print_correctness(io, r)
end

# ---------------------------------------------------------------- CSV

prefix(r::Report) = join((IMPLEMENTATION, csv_field(r.backend), csv_field(r.device),
                          r.precision), ",")

const CSV_HEADER = "implementation,backend,device,precision,sqrt_mode,input,natoms,ngauss," *
                   "block_size,quartets,groups,num_times,routine,min_s,max_s,avg_s,e2_energy,rel_err_ref," *
                   "energy_j_per_iter,avg_power_w"

const CSV_RAW_HEADER = "implementation,backend,device,precision,sqrt_mode,input,natoms," *
                       "ngauss,block_size,quartets,groups,routine,iteration,time_s"

f9(x) = fixed(x, 9)

function print_csv(io::IO, r::Report)
    println(io, CSV_HEADER)
    tmin, tmax, tavg = stats(r.times)
    e = r.energy
    ej, ew = e === nothing ? ("", "") : (fixed(joules_per_iter(e), 6), fixed(e.avg_power_w, 2))
    rel = isnan(r.rel_err_ref) ? "" : sci(r.rel_err_ref, 6)
    println(io, join((prefix(r), r.sqrt_mode, csv_field(r.input), r.natoms, r.ngauss,
                      r.block_size, r.nnnn, cld(r.nnnn + 1, r.block_size), r.num_times,
                      ROUTINE, f9(tmin), f9(tmax), f9(tavg),
                      sci(r.e2_energy, 15), rel, ej, ew), ","))
end

function print_csv_raw(io::IO, r::Report)
    println(io, CSV_RAW_HEADER)
    for (k, t) in enumerate(r.times)
        println(io, join((prefix(r), r.sqrt_mode, csv_field(r.input), r.natoms, r.ngauss,
                          r.block_size, r.nnnn, cld(r.nnnn + 1, r.block_size),
                          ROUTINE, k, f9(t)), ","))
    end
end

end # module Reports
