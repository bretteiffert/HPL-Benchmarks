# Hartree-Fock report: one struct filled by the run, three formatters
# (text, --csv, --csv-raw) that read only from it.
#
# Stream rules (applied by main): text mode writes everything to stdout; the
# CSV modes write only CSV to stdout and the text header, energy and
# correctness sections to stderr.

using Printf

Base.@kwdef mutable struct RoutineResult
    name::String = ""                    # "Fock build"
    times::Vector{Float64} = Float64[]   # seconds, timed builds only (untimed first build excluded)
end

min_s(r::RoutineResult) = isempty(r.times) ? 0.0 : minimum(r.times)
max_s(r::RoutineResult) = isempty(r.times) ? 0.0 : maximum(r.times)

Base.@kwdef mutable struct Report
    benchmark::String = "Hartree-Fock"
    implementation::String = ""          # port name, "KernelAbstractions"
    backend::String = ""
    device::String = ""
    driver::String = ""
    precision::Precision = PrecisionDouble
    sqrt_mode::SqrtMode = SqrtNative

    input::String = ""                   # file name without directories
    natoms::UInt32 = 0
    ngauss::UInt32 = 0
    pairs::UInt64 = 0                    # unique atom pairs
    quartets::UInt64 = 0                 # unique pair quartets = kernel threads doing work

    block_size::UInt32 = 0
    grid::UInt64 = 0
    num_times::Int = 0                   # timed builds
    timing::String = ""

    header_notes::Vector{String} = String[]

    fock::RoutineResult = RoutineResult()
    checksum::Float64 = 0.0              # 2e- energy of the first build, run-precision sum
    energy::EnergyResult = EnergyResult()

    checked::Bool = false
    check_skipped::String = ""           # reason, when not checked
    check::FockCheck = FockCheck()
end

# Left-to-right sum, as std::accumulate (Base.sum would sum pairwise).
function avg_s(r::RoutineResult)
    isempty(r.times) && return 0.0
    acc = 0.0
    for t in r.times
        acc += t
    end
    return acc / length(r.times)
end

# ---- formatting helpers ----------------------------------------------------

function format_thousands(v::Integer)::String
    neg = v < 0
    digits = string(neg ? -Int128(v) : Int128(v))
    out = Char[]
    n = 0
    for ch in reverse(digits)
        (n > 0 && n % 3 == 0) && push!(out, ',')
        push!(out, ch)
        n += 1
    end
    neg && push!(out, '-')
    return String(reverse(out))
end

const _fmt_cache = Dict{String, Printf.Format}()
cfmt(spec::String, v) = Printf.format(get!(() -> Printf.Format(spec), _fmt_cache, spec), v)

function format_fixed(v::Float64, decimals::Int)::String
    isnan(v) && return "nan"
    isinf(v) && return v > 0 ? "inf" : "-inf"
    return cfmt("%." * string(decimals) * "f", v)
end

# 1.230000e-1: C %.*e with the exponent padding removed.
function format_sci(v::Float64, decimals::Int)::String
    isnan(v) && return "nan"
    isinf(v) && return v > 0 ? "inf" : "-inf"
    s = cfmt("%." * string(decimals) * "e", v)
    e = findfirst('e', s)
    e === nothing && return s
    exponent = parse(Int, s[e+1:end])
    return s[1:e-1] * "e" * string(exponent)
end

function csv_field(s::AbstractString)::String
    any(c -> c in (',', '"', '\n', '\r'), s) || return String(s)
    return "\"" * replace(s, "\"" => "\"\"") * "\""
end

pad_left(s::AbstractString, width::Int) = ncodeunits(s) >= width ? String(s) : " "^(width - ncodeunits(s)) * s
pad_right(s::AbstractString, width::Int) = ncodeunits(s) >= width ? String(s) : s * " "^(width - ncodeunits(s))

key(out::IO, k::AbstractString, v::AbstractString) = print(out, "  ", pad_right(k, 13), v, "\n")

csv_prefix(r::Report) = csv_field(r.implementation) * "," * csv_field(r.backend) * "," *
                        csv_field(r.device) * "," * precision_name(r.precision) * ","

# CSV numbers: full precision for energies, plain C exponent form (printf %.16e).
function csv_sci(v::Float64)::String
    isnan(v) && return signbit(v) ? "-nan" : "nan"
    isinf(v) && return v > 0 ? "inf" : "-inf"
    return cfmt("%.16e", v)
end

# ---- text ------------------------------------------------------------------

function print_text_header(r::Report, out::IO)
    print(out, r.benchmark, " (", r.implementation, ")\n")
    key(out, "Backend", r.backend)
    key(out, "Device", r.device)
    key(out, "Driver", r.driver)
    key(out, "Precision", precision_name(r.precision))
    key(out, "Sqrt mode", sqrt_mode_name(r.sqrt_mode))
    key(out, "Input", r.input * " (" * format_thousands(r.natoms) * " atoms, " *
                      format_thousands(r.ngauss) * " Gaussians)")
    key(out, "Quartets", format_thousands(r.quartets) * " (from " *
                         format_thousands(r.pairs) * " atom pairs)")
    key(out, "Block size", string(r.block_size))
    key(out, "Grid", format_thousands(r.grid) * " blocks")
    key(out, "Iterations", format_thousands(r.num_times) * " (after 1 untimed build)")
    key(out, "Timing", r.timing)
    key(out, "Energy", r.energy.enabled ? "libkernelenergy (board power)" : "disabled (--no-energy)")

    for n in r.header_notes
        print(out, "  note: ", n, "\n")
    end
    flush(out)
end

function print_text_results(r::Report, out::IO)
    t_w = 14
    print(out, "\n",
          pad_right("Routine", 12), pad_left("Min (s)", t_w), pad_left("Max (s)", t_w),
          pad_left("Avg (s)", t_w), "\n")
    print(out, pad_right(r.fock.name, 12),
          pad_left(format_fixed(min_s(r.fock), 8), t_w),
          pad_left(format_fixed(max_s(r.fock), 8), t_w),
          pad_left(format_fixed(avg_s(r.fock), 8), t_w), "\n")
    print(out, "  note: each timed build adds into the same Fock matrix; the checksum and the check use the untimed first build\n")
    flush(out)
end

function print_text_energy(r::Report, out::IO)
    print(out, "\n")
    if !r.energy.enabled
        print(out, "Energy  disabled (--no-energy)\n")
        flush(out)
        return
    end

    e = r.energy
    print(out, pad_right("Energy", 8), pad_left("J/iter", 14), pad_left("Total (Wh)", 14),
          pad_left("Avg power (W)", 16), pad_left("Time (s)", 12),
          pad_left("Iterations", 13), pad_left("Samples", 10), "\n")
    print(out, pad_right("board", 8),
          pad_left(format_fixed(e.joules_per_iter, 6), 14),
          pad_left(format_fixed(e.total_wh, 6), 14),
          pad_left(format_fixed(e.avg_power_w, 2), 16),
          pad_left(format_fixed(e.seconds, 6), 12),
          pad_left(format_thousands(e.iterations), 13),
          pad_left(format_thousands(e.samples), 10), "\n")
    e.samples < 2 && print(out, "  note: fewer than two power samples; the energy figures may be unreliable\n")
    flush(out)
end

function print_text_correctness(r::Report, out::IO)
    prec = precision_name(r.precision)
    label_w = 42
    value_w = 24
    row(label, value) = print(out, pad_right(label, label_w), pad_left(value, value_w), "\n")

    print(out, "\n",
          "Correctness (", prec, ", sqrt ", sqrt_mode_name(r.sqrt_mode),
          ", untimed first build vs fp64 CPU reference)\n")
    print(out, pad_right("Quantity", label_w), pad_left("Value", value_w), "\n")

    row("2e- energy (" * prec * " Fock, " * prec * " sum)", format_sci(r.checksum, 16))

    if r.checked
        c = r.check
        row("2e- energy (" * prec * " Fock, fp64 sum)", format_sci(c.e_gpu, 16))
        row("2e- energy (fp64 CPU reference)", format_sci(c.e_ref, 16))
        row("rel err, Fock build only", format_sci(c.rel_loss, 6))
        row("rel err, incl. " * prec * " sum", format_sci(c.rel_inpipe, 6))
        row("max |F - F_ref|", format_sci(c.max_abs, 6))
        row("max |F - F_ref| / max |F_ref|", format_sci(c.max_rel, 6))
        row("rms |F - F_ref|", format_sci(c.rms_abs, 6))
        row("max |F(i,j) - F(j,i)|", format_sci(c.asym, 6))
        row("2e- energy, symmetrized F (diagnostic)", format_sci(c.e_symmetrized, 16))
    else
        print(out, "  note: reference check skipped: ", r.check_skipped, "\n")
    end
    print(out, "  note: the 2e- energy is an unsymmetrized benchmark checksum, not a physical two-electron energy\n")
    flush(out)
end

# ---- CSV -------------------------------------------------------------------

function print_csv(r::Report, out::IO; with_header::Bool = true)
    if with_header
        print(out, "implementation,backend,device,precision,sqrt_mode,input,natoms,ngauss,block_size,quartets,groups,num_times,",
                   "routine,min_s,max_s,avg_s,e2_energy,rel_err_ref,energy_j_per_iter,avg_power_w\n")
    end

    print(out, csv_prefix(r), sqrt_mode_name(r.sqrt_mode), ",", csv_field(r.input), ",",
          r.natoms, ",", r.ngauss, ",", r.block_size, ",", r.quartets, ",",
          r.grid, ",", r.num_times, ",",
          csv_field(r.fock.name), ",",
          format_fixed(min_s(r.fock), 9), ",", format_fixed(max_s(r.fock), 9), ",",
          format_fixed(avg_s(r.fock), 9), ",",
          csv_sci(r.checksum), ",")
    r.checked && print(out, csv_sci(r.check.rel_inpipe))
    print(out, ",")
    if r.energy.enabled
        print(out, format_fixed(r.energy.joules_per_iter, 6), ",", format_fixed(r.energy.avg_power_w, 3))
    else
        print(out, ",")
    end
    print(out, "\n")
    flush(out)
end

function print_csv_raw(r::Report, out::IO; with_header::Bool = true)
    if with_header
        print(out, "implementation,backend,device,precision,sqrt_mode,input,natoms,ngauss,block_size,quartets,groups,",
                   "routine,iteration,time_s\n")
    end

    prefix = csv_prefix(r) * sqrt_mode_name(r.sqrt_mode) * "," * csv_field(r.input) * "," *
             string(r.natoms) * "," * string(r.ngauss) * "," *
             string(r.block_size) * "," * string(r.quartets) * "," *
             string(r.grid) * "," * csv_field(r.fock.name) * ","
    for (j, t) in enumerate(r.fock.times)
        print(out, prefix, j, ",", format_fixed(t, 9), "\n")
    end
    flush(out)
end
