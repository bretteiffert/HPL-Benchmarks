# report.jl — the miniBUDE report and its three formatters (text, --csv, --csv-raw).
# Formatters read only from the Report; nothing else prints results.

module Reports

using Printf
using ..BoardEnergy: EnergyResult, joules_per_iter, total_wh
using ..Bude: Metrics, ULP_NBUCKETS, ulp_bucket_label, REPORT_TOLERANCE_PCT

export Report, ConfigResult, print_header, print_text, print_csv, print_csv_raw,
       print_stderr_sections, IMPLEMENTATION, best_config

const IMPLEMENTATION = "KernelAbstractions"

struct ConfigResult
    ppwi::Int
    wgsize::Int
    times::Vector{Float64}                 # timed iterations only, seconds
    gflops::Float64
    gfinsts::Float64
    ginteractions::Float64
    launch_overhead_s::Float64             # avg (wall - kernel) per timed iteration
    context_s::Float64                     # allocation + host-to-device copies
    energy::Union{Nothing,EnergyResult}
    ref::Union{Nothing,Metrics}            # vs float64 pass (float32 runs only)
    deck::Metrics                          # vs ref_energies.out
    energies_head::Vector{Float64}
end

Base.@kwdef mutable struct Report
    backend::String
    device::String
    driver::String
    precision::String
    deck::String                 # basename of the deck directory
    natlig::Int
    natpro::Int
    ntypes::Int
    nposes::Int
    max_poses::Int
    ppwi::Vector{Int}
    wgsize::Vector{Int}
    iterations::Int
    warmup::Int
    timing::String
    energy_enabled::Bool
    rows::Int = 8
    header_notes::Vector{String} = String[]
    configs::Vector{ConfigResult} = ConfigResult[]
    result_notes::Vector{String} = String[]
    energy_notes::Vector{String} = String[]
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

const DASH = "—"

csv_field(s::AbstractString) =
    any(c -> c in (',', '"', '\n', '\r'), s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : String(s)

avg(t) = sum(t) / length(t)
"Population standard deviation (as upstream miniBUDE)."
stddev(t) = (m = avg(t); sqrt(sum((x - m)^2 for x in t) / length(t)))

kv(io, k, v) = println(io, "  ", rpad(k, 13), v)

"Configuration with the lowest average time."
best_config(r::Report) = r.configs[argmin([avg(c.times) for c in r.configs])]

cfg_label(c::ConfigResult) = "p$(c.ppwi) w$(c.wgsize)"

# ---------------------------------------------------------------- text

function print_header(io::IO, r::Report)
    println(io, "miniBUDE (", IMPLEMENTATION, ")")
    kv(io, "Backend", r.backend)
    kv(io, "Device", r.device)
    kv(io, "Driver", r.driver)
    kv(io, "Precision", r.precision)
    kv(io, "Deck", "$(r.deck) ($(thousands(r.natlig)) ligand atoms, " *
                   "$(thousands(r.natpro)) protein atoms, $(r.ntypes) types)")
    kv(io, "Poses", "$(thousands(r.nposes)) (of $(thousands(r.max_poses)))")
    kv(io, "PPWI", join(r.ppwi, ","))
    kv(io, "Workgroup", join(r.wgsize, ","))
    kv(io, "Iterations", "$(r.iterations) (after $(r.warmup) warm-up)")
    kv(io, "Timing", r.timing)
    kv(io, "Energy", r.energy_enabled ? "libkernelenergy (board power)" : "disabled (--no-energy)")
    for n in r.header_notes
        println(io, "  note: ", n)
    end
    flush(io)
end

function print_results(io::IO, r::Report)
    println(io, lpad("PPWI", 6), lpad("WG", 6), lpad("Min (s)", 14), lpad("Max (s)", 14),
            lpad("Avg (s)", 14), lpad("Stddev (s)", 14), lpad("GFLOP/s", 12),
            lpad("GFInst/s", 12), lpad("GInter/s", 12), lpad("Overhead (s)", 14),
            lpad("Context (s)", 14))
    for c in r.configs
        t = c.times
        println(io, lpad(string(c.ppwi), 6), lpad(string(c.wgsize), 6),
                lpad(fixed(minimum(t), 8), 14), lpad(fixed(maximum(t), 8), 14),
                lpad(fixed(avg(t), 8), 14), lpad(fixed(stddev(t), 8), 14),
                lpad(fixed(c.gflops, 3), 12), lpad(fixed(c.gfinsts, 3), 12),
                lpad(fixed(c.ginteractions, 3), 12), lpad(fixed(c.launch_overhead_s, 8), 14),
                lpad(fixed(c.context_s, 8), 14))
    end
    if length(r.configs) > 1
        b = best_config(r)
        println(io, "  note: best: ppwi $(b.ppwi), wgsize $(b.wgsize) (lowest avg, " *
                    "$(fixed(avg(b.times), 8)) s)")
    end
    for n in r.result_notes
        println(io, "  note: ", n)
    end
end

function print_energy(io::IO, r::Report)
    if !r.energy_enabled
        println(io, "Energy  disabled (--no-energy)")
    else
        println(io, rpad("Energy", 12), lpad("J/iter", 14), lpad("Total (Wh)", 14),
                lpad("Avg power (W)", 16), lpad("Time (s)", 12), lpad("Iterations", 13),
                lpad("Samples", 10))
        for c in r.configs
            e = c.energy
            e === nothing && continue
            println(io, rpad(cfg_label(c), 12), lpad(fixed(joules_per_iter(e), 6), 14),
                    lpad(fixed(total_wh(e), 6), 14), lpad(fixed(e.avg_power_w, 2), 16),
                    lpad(fixed(e.time_s, 6), 12), lpad(thousands(e.iterations), 13),
                    lpad(thousands(e.samples), 10))
        end
    end
    for n in r.energy_notes
        println(io, "  note: ", n)
    end
end

function print_correctness(io::IO, r::Report)
    println(io, "Correctness (", r.precision, ", ", thousands(r.nposes), " poses)")
    println(io, lpad("PPWI", 6), lpad("WG", 6), lpad("Ref L2 rel", 14), lpad("Ref max rel", 14),
            lpad("Ref max ULP", 14), lpad("Deck L2 rel", 14), lpad("Deck max rel", 14),
            lpad("Deck beyond tol", 18))
    for c in r.configs
        ref = c.ref
        println(io, lpad(string(c.ppwi), 6), lpad(string(c.wgsize), 6),
                lpad(ref === nothing ? DASH : sci(ref.l2RelNorm, 3), 14),
                lpad(ref === nothing ? DASH : sci(ref.maxRel, 3), 14),
                lpad(ref === nothing ? DASH : string(ref.maxUlp), 14),
                lpad(sci(c.deck.l2RelNorm, 3), 14), lpad(sci(c.deck.maxRel, 3), 14),
                lpad("$(c.deck.beyondTolerance)/$(c.deck.compared)", 18))
    end

    # ULP histograms vs the float64 pass, collapsed when identical.
    withref = [c for c in r.configs if c.ref !== nothing]
    if !isempty(withref)
        groups = Vector{Tuple{Vector{Int},Vector{String}}}()
        for c in withref
            h = c.ref.ulpHistogram
            k = findfirst(g -> g[1] == h, groups)
            k === nothing ? push!(groups, (copy(h), [cfg_label(c)])) : push!(groups[k][2], cfg_label(c))
        end
        for (h, labels) in groups
            who = length(labels) == length(r.configs) ? "all configurations" : join(labels, ", ")
            println(io, "ULP histogram vs float64 (", who, ")")
            for i in 1:ULP_NBUCKETS
                h[i] > 0 && println(io, lpad(ulp_bucket_label(i), 12), lpad(thousands(h[i]), 14))
            end
        end
    end

    if r.rows > 0 && !isempty(r.configs)
        c = first(r.configs)
        k = min(r.rows, length(c.energies_head))
        println(io, "Energies (first $k, ppwi $(c.ppwi), wgsize $(c.wgsize))")
        for e in c.energies_head[1:k]
            println(io, lpad(fixed(e, 6), 20))
        end
    end
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

"CSV modes: notes, energy and correctness go to stderr."
function print_stderr_sections(io::IO, r::Report)
    if length(r.configs) > 1 || !isempty(r.result_notes)
        println(io)
        if length(r.configs) > 1
            b = best_config(r)
            println(io, "  note: best: ppwi $(b.ppwi), wgsize $(b.wgsize) (lowest avg, " *
                        "$(fixed(avg(b.times), 8)) s)")
        end
        for n in r.result_notes
            println(io, "  note: ", n)
        end
    end
    println(io)
    print_energy(io, r)
    println(io)
    print_correctness(io, r)
end

# ---------------------------------------------------------------- CSV

prefix(r::Report) = join((IMPLEMENTATION, csv_field(r.backend), csv_field(r.device),
                          r.precision), ",")

const CSV_HEADER = "implementation,backend,device,precision,deck,natlig,natpro,nposes,ppwi," *
                   "wgsize,iterations,min_s,max_s,avg_s,stddev_s,gflops,gfinsts,ginteractions," *
                   "launch_overhead_s,context_s,energy_j_per_iter,avg_power_w,ref_l2_rel," *
                   "ref_max_rel,ref_max_ulp,deck_l2_rel,deck_beyond_tol"

const CSV_RAW_HEADER = "implementation,backend,device,precision,deck,natlig,natpro,nposes," *
                       "ppwi,wgsize,iteration,time_s"

f9(x) = fixed(x, 9)
f4(x) = fixed(x, 4)

function print_csv(io::IO, r::Report)
    println(io, CSV_HEADER)
    for c in r.configs
        t = c.times
        e = c.energy
        ej, ew = e === nothing ? ("", "") : (fixed(joules_per_iter(e), 6), fixed(e.avg_power_w, 2))
        ref = c.ref
        rl2, rmax, rulp = ref === nothing ? ("", "", "") :
            (sci(ref.l2RelNorm, 6), sci(ref.maxRel, 6), string(ref.maxUlp))
        println(io, join((prefix(r), csv_field(r.deck), r.natlig, r.natpro, r.nposes,
                          c.ppwi, c.wgsize, r.iterations, f9(minimum(t)), f9(maximum(t)),
                          f9(avg(t)), f9(stddev(t)), f4(c.gflops), f4(c.gfinsts),
                          f4(c.ginteractions), f9(c.launch_overhead_s), f9(c.context_s),
                          ej, ew, rl2, rmax, rulp, sci(c.deck.l2RelNorm, 6),
                          c.deck.beyondTolerance), ","))
    end
end

function print_csv_raw(io::IO, r::Report)
    println(io, CSV_RAW_HEADER)
    for c in r.configs, (k, t) in enumerate(c.times)
        println(io, join((prefix(r), csv_field(r.deck), r.natlig, r.natpro, r.nposes,
                          c.ppwi, c.wgsize, k, f9(t)), ","))
    end
end

end # module Reports
