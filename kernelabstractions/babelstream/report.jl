# report.jl — the BabelStream report and its three formatters (text, --csv, --csv-raw).
# Formatters read only from the Report; nothing else prints results.

module Reports

using Printf
using ..BoardEnergy: EnergyResult, joules_per_iter, total_wh

export Report, RoutineResult, ArrayCheck, print_header, print_text, print_csv,
       print_csv_raw, print_stderr_sections, IMPLEMENTATION

const IMPLEMENTATION = "KernelAbstractions"

struct RoutineResult
    name::String
    bytes::Int
    times::Vector{Float64}     # timed iterations only (warm-up excluded); Init has one
    one_shot::Bool
end

struct ArrayCheck
    name::String
    expected::Float64
    checksum_rel_err::Float64
    max_elem_rel_err::Float64
end

Base.@kwdef mutable struct Report
    backend::String
    device::String
    driver::String
    precision::String             # float16 / float32 / float64
    elem_bytes::Int
    array_size::Int
    block_size::Int
    dot_block::Int
    dot_groups::Int
    dot_groups_desc::String       # e.g. "432 (4 × 108 SMs)"
    num_times::Int
    timing::String
    energy_enabled::Bool
    mibibytes::Bool = false
    header_notes::Vector{String} = String[]
    routines::Vector{RoutineResult} = RoutineResult[]
    energy::Union{Nothing,EnergyResult} = nothing
    energy_notes::Vector{String} = String[]
    checks::Vector{ArrayCheck} = ArrayCheck[]
    dot_expected::Float64 = NaN
    dot_rel_err::Float64 = NaN
    dot_actual::Float64 = NaN
    check_notes::Vector{String} = String[]
end

# ---------------------------------------------------------------- helpers

"1234567 -> \"1,234,567\""
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
    ev = parse(Int, e)
    return "$(m)e$(ev)"
end

fixed(x::Real, d::Int) = Printf.format(Printf.Format("%.$(d)f"), Float64(x))

const DASH = "—"

csv_field(s::AbstractString) =
    any(c -> c in (',', '"', '\n', '\r'), s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : String(s)

bw_gbs(bytes, t) = 1.0e-9 * bytes / t
bw_mibs(bytes, t) = bytes / 2.0^20 / t

stats(t::Vector{Float64}) = (minimum(t), maximum(t), sum(t) / length(t))

function size_str(r::Report, bytes::Real)
    r.mibibytes ? @sprintf("%.1f MiB", bytes / 2.0^20) : @sprintf("%.1f MB", bytes * 1.0e-6)
end

kv(io, k, v) = println(io, "  ", rpad(k, 13), v)

# ---------------------------------------------------------------- text

"Section 1: printed before the run starts."
function print_header(io::IO, r::Report)
    println(io, "BabelStream (", IMPLEMENTATION, ")")
    kv(io, "Backend", r.backend)
    kv(io, "Device", r.device)
    kv(io, "Driver", r.driver)
    kv(io, "Precision", r.precision)
    per = r.elem_bytes * r.array_size
    kv(io, "Array size", "$(thousands(r.array_size)) elements ($(size_str(r, per)) per array, " *
                         "$(size_str(r, 3per)) total)")
    bs = r.block_size == r.dot_block ? string(r.block_size) :
         "$(r.block_size) (Dot uses $(r.dot_block))"
    kv(io, "Block size", bs)
    kv(io, "Dot groups", r.dot_groups_desc)
    kv(io, "Iterations", "$(r.num_times) (first excluded from statistics)")
    kv(io, "Timing", r.timing)
    kv(io, "Energy", r.energy_enabled ? "libkernelenergy (board power)" : "disabled (--no-energy)")
    for n in r.header_notes
        println(io, "  note: ", n)
    end
    flush(io)
end

function print_results(io::IO, r::Report)
    unit = r.mibibytes ? "Bandwidth (MiB/s)" : "Bandwidth (GB/s)"
    bww = r.mibibytes ? 18 : 16
    println(io, rpad("Function", 12), lpad(unit, bww),
            lpad("Min (s)", 14), lpad("Max (s)", 14), lpad("Avg (s)", 14))
    bwf = r.mibibytes ? bw_mibs : bw_gbs
    for rt in r.routines
        tmin, tmax, tavg = stats(rt.times)
        bw = fixed(bwf(rt.bytes, tmin), 3)
        mx = rt.one_shot ? DASH : fixed(tmax, 8)
        av = rt.one_shot ? DASH : fixed(tavg, 8)
        println(io, rpad(rt.name, 12), lpad(bw, bww),
                lpad(fixed(tmin, 8), 14), lpad(mx, 14), lpad(av, 14))
    end
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
    println(io, "Correctness (", r.precision, ", ", r.num_times, " iterations)")
    println(io, rpad("Array", 6), lpad("Expected", 20), lpad("Checksum rel err", 20),
            lpad("Max elem rel err", 20))
    for c in r.checks
        println(io, rpad(c.name, 6), lpad(sci(c.expected, 6), 20),
                lpad(sci(c.checksum_rel_err, 3), 20), lpad(sci(c.max_elem_rel_err, 3), 20))
    end
    println(io, rpad("dot", 6), lpad(sci(r.dot_expected, 6), 20),
            lpad(sci(r.dot_rel_err, 3), 20), lpad(DASH, 20))
    println(io, "  note: actual dot = ", sci(r.dot_actual, 9))
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

const CSV_HEADER = "implementation,backend,device,precision,array_size,block_size,dot_groups," *
                   "num_times,routine,bw_gbs,min_s,max_s,avg_s,energy_j_per_iter,avg_power_w"

const CSV_RAW_HEADER = "implementation,backend,device,precision,array_size,block_size," *
                       "routine,iteration,time_s,bw_gbs"

f9(x) = fixed(x, 9)
f4(x) = fixed(x, 4)

function print_csv(io::IO, r::Report)
    println(io, CSV_HEADER)
    e = r.energy
    for rt in r.routines
        tmin, tmax, tavg = stats(rt.times)
        # Energy covers the timed iterations; the one-shot Init is outside it.
        ej, ew = (e === nothing || rt.one_shot) ? ("", "") :
                 (fixed(joules_per_iter(e), 6), fixed(e.avg_power_w, 2))
        println(io, join((prefix(r), r.array_size, r.block_size, r.dot_groups, r.num_times,
                          rt.name, f4(bw_gbs(rt.bytes, tmin)), f9(tmin),
                          rt.one_shot ? "" : f9(tmax), rt.one_shot ? "" : f9(tavg),
                          ej, ew), ","))
    end
end

function print_csv_raw(io::IO, r::Report)
    println(io, CSV_RAW_HEADER)
    for rt in r.routines
        rt.one_shot && continue
        for (k, t) in enumerate(rt.times)
            println(io, join((prefix(r), r.array_size, r.block_size, rt.name, k,
                              f9(t), f4(bw_gbs(rt.bytes, t))), ","))
        end
    end
end

end # module Reports
