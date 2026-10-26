# report.jl — the stencil report and its three formatters (text, --csv, --csv-raw).
# Formatters read only from the Report; nothing else prints results.

module Reports

using Printf
using ..BoardEnergy: EnergyResult, joules_per_iter, total_wh

export Report, print_header, print_text, print_csv, print_csv_raw,
       print_stderr_sections, IMPLEMENTATION, ROUTINE

const IMPLEMENTATION = "KernelAbstractions"
const ROUTINE = "Laplacian"

Base.@kwdef mutable struct Report
    backend::String
    device::String
    driver::String
    precision::String
    elem_bytes::Int
    nx::Int
    ny::Int
    nz::Int
    block::NTuple{3,Int}
    workgroups::Int
    fetch_bytes::Int
    write_bytes::Int
    num_times::Int
    timing::String
    energy_enabled::Bool
    header_notes::Vector{String} = String[]
    times::Vector{Float64} = Float64[]
    energy::Union{Nothing,EnergyResult} = nothing
    energy_notes::Vector{String} = String[]
    max_abs_err::Float64 = NaN
    mean_abs_err::Float64 = NaN
    l2_rel_err::Float64 = NaN
    points_checked::Int = 0
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

bytes_moved(r::Report) = r.fetch_bytes + r.write_bytes
bw_gbs(r::Report, t) = 1.0e-9 * bytes_moved(r) / t
mb(x) = @sprintf("%.1f MB", x * 1.0e-6)

kv(io, k, v) = println(io, "  ", rpad(k, 13), v)

# ---------------------------------------------------------------- text

function print_header(io::IO, r::Report)
    println(io, "Seven-point stencil (", IMPLEMENTATION, ")")
    kv(io, "Backend", r.backend)
    kv(io, "Device", r.device)
    kv(io, "Driver", r.driver)
    kv(io, "Precision", r.precision)
    kv(io, "Grid", "$(thousands(r.nx)) × $(thousands(r.ny)) × $(thousands(r.nz)) " *
                   "($(mb(r.elem_bytes * r.nx * r.ny * r.nz)) per array)")
    kv(io, "Block size", "$(r.block[1]) × $(r.block[2]) × $(r.block[3]) " *
                         "($(thousands(r.workgroups)) workgroups)")
    kv(io, "Traffic", "$(mb(r.fetch_bytes)) read, $(mb(r.write_bytes)) written per " *
                      "iteration (theoretical)")
    kv(io, "Iterations", "$(r.num_times) (after 1 untimed run)")
    kv(io, "Timing", r.timing)
    kv(io, "Energy", r.energy_enabled ? "libkernelenergy (board power)" : "disabled (--no-energy)")
    for n in r.header_notes
        println(io, "  note: ", n)
    end
    flush(io)
end

function print_results(io::IO, r::Report)
    println(io, rpad("Routine", 12), lpad("Bandwidth (GB/s)", 16), lpad("Min (s)", 14),
            lpad("Max (s)", 14), lpad("Avg (s)", 14))
    tmin, tmax, tavg = stats(r.times)
    println(io, rpad(ROUTINE, 12), lpad(fixed(bw_gbs(r, tmin), 3), 16), lpad(fixed(tmin, 8), 14),
            lpad(fixed(tmax, 8), 14), lpad(fixed(tavg, 8), 14))
    println(io, "  note: bandwidth uses the minimum time; from the average time it is ",
            fixed(bw_gbs(r, tavg), 3), " GB/s")
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
    println(io, "Correctness (", r.precision, ", untimed run, ∇²u = 3 at interior points)")
    println(io, rpad("Points checked", 20), lpad(thousands(r.points_checked), 20))
    println(io, rpad("Max abs err", 20), lpad(sci(r.max_abs_err, 3), 20))
    println(io, rpad("Mean abs err", 20), lpad(sci(r.mean_abs_err, 3), 20))
    println(io, rpad("L2 rel err", 20), lpad(sci(r.l2_rel_err, 3), 20))
    for n in r.check_notes
        println(io, "  note: ", n)
    end
end

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

const CSV_HEADER = "implementation,backend,device,precision,nx,ny,nz,block_x,block_y,block_z," *
                   "num_times,routine,bw_gbs,min_s,max_s,avg_s,energy_j_per_iter,avg_power_w"

const CSV_RAW_HEADER = "implementation,backend,device,precision,nx,ny,nz,block_x,block_y," *
                       "block_z,routine,iteration,time_s,bw_gbs"

f9(x) = fixed(x, 9)
f4(x) = fixed(x, 4)

function print_csv(io::IO, r::Report)
    println(io, CSV_HEADER)
    tmin, tmax, tavg = stats(r.times)
    e = r.energy
    ej, ew = e === nothing ? ("", "") : (fixed(joules_per_iter(e), 6), fixed(e.avg_power_w, 2))
    println(io, join((prefix(r), r.nx, r.ny, r.nz, r.block..., r.num_times, ROUTINE,
                      f4(bw_gbs(r, tmin)), f9(tmin), f9(tmax), f9(tavg), ej, ew), ","))
end

function print_csv_raw(io::IO, r::Report)
    println(io, CSV_RAW_HEADER)
    for (k, t) in enumerate(r.times)
        println(io, join((prefix(r), r.nx, r.ny, r.nz, r.block..., ROUTINE, k, f9(t),
                          f4(bw_gbs(r, t))), ","))
    end
end

end # module Reports
