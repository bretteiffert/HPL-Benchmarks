# Command-line parsing and --help for Hartree-Fock.

@enum OutputMode OutputText OutputCsv OutputCsvRaw

Base.@kwdef mutable struct Options
    input::String = ""                     # positional input file
    precision::Precision = PrecisionDouble
    sqrt_mode::SqrtMode = SqrtNative
    num_times::Int = 10                    # timed builds, after 1 untimed build
    device::Int = 0
    block_size::Int = 256
    output::OutputMode = OutputText
    energy::Bool = true
    check::Bool = true                     # --no-check turns it off
    force_check::Bool = false              # --check: ignore --max-check-atoms
    max_check_atoms::Int = 128
    list::Bool = false
    help::Bool = false
end

# A user error. main prints "error: <what> (try '--help')" and exits 1.
struct CliError <: Exception
    msg::String
end

const INT_MAX = Int64(typemax(Int32))

# strtoll(s, &end, 10) with errno == 0 and *end == '\0': optional leading C
# whitespace, optional sign, decimal digits, nothing after them, no overflow.
function parse_ll(s::AbstractString)::Union{Int64, Nothing}
    b = codeunits(s)
    n = length(b)
    n == 0 && return nothing
    p = 1
    while p <= n && b[p] in (0x20, 0x09, 0x0a, 0x0b, 0x0c, 0x0d)
        p += 1
    end
    neg = false
    if p <= n && (b[p] == UInt8('+') || b[p] == UInt8('-'))
        neg = b[p] == UInt8('-')
        p += 1
    end
    start = p
    v = Int128(0)
    overflow = false
    while p <= n && UInt8('0') <= b[p] <= UInt8('9')
        v = v * 10 + (b[p] - UInt8('0'))
        if v > Int128(typemax(Int64)) + 1
            overflow = true
            v = Int128(typemax(Int64)) + 1
        end
        p += 1
    end
    p == start && return nothing          # no digits: end == s
    p <= n && return nothing              # *end != '\0'
    v = neg ? -v : v
    (overflow || v > typemax(Int64) || v < typemin(Int64)) && return nothing  # ERANGE
    return Int64(v)
end

function value_for(args::Vector{String}, i::Int)
    i + 1 > length(args) && throw(CliError("missing value for " * args[i]))
    return args[i + 1]
end

function int_value(args::Vector{String}, i::Int, what::AbstractString, min::Int64)
    v = value_for(args, i)
    n = parse_ll(v)
    if n === nothing || n < min || n > INT_MAX
        throw(CliError("invalid " * what * " '" * v * "' (must be an integer >= " * string(min) * ")"))
    end
    return Int(n)
end

# Throws CliError. --help and --list are reported through Options.
function parse_cli(args::Vector{String})::Options
    o = Options()
    csv = false
    csv_raw = false

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "-h" || arg == "--help"
            o.help = true
        elseif arg == "--list"
            o.list = true
        elseif arg == "--device"
            o.device = int_value(args, i, "device index", 0); i += 1
        elseif arg == "-n" || arg == "--iters"
            o.num_times = int_value(args, i, "number of iterations", 1); i += 1
        elseif arg == "--block-size"
            o.block_size = int_value(args, i, "block size", 1); i += 1
        elseif arg == "--max-check-atoms"
            o.max_check_atoms = int_value(args, i, "--max-check-atoms value", 0); i += 1
        elseif arg == "--double"
            o.precision = PrecisionDouble
        elseif arg == "--float"
            o.precision = PrecisionFloat
        elseif arg == "--legacy-sqrt"
            throw(CliError("the pinned configuration requires native square roots; --legacy-sqrt is not supported"))
        elseif arg == "--check"
            o.check = true; o.force_check = true
        elseif arg == "--no-check"
            o.check = false; o.force_check = false
        elseif arg == "--csv"
            csv = true
        elseif arg == "--csv-raw"
            csv_raw = true
        elseif arg == "--no-energy"
            o.energy = false
        elseif arg == "--half"
            throw(CliError("--half is not available for Hartree-Fock (the 1e-12 screening thresholds are below " *
                           "the float16 range); use --float or --double"))
        elseif ncodeunits(arg) > 1 && arg[1] == '-'
            eq = findfirst('=', arg)
            if startswith(arg, "--") && eq !== nothing
                flag = arg[1:prevind(arg, eq)]
                flag == "--iters" && (flag = "-n")
                throw(CliError("unrecognized argument '" * arg * "' (use '" * flag * " " *
                               arg[nextind(arg, eq):end] * "')"))
            end
            throw(CliError("unrecognized argument '" * arg * "'"))
        else
            if !isempty(o.input)
                throw(CliError("unexpected argument '" * arg * "' (only one input file is allowed; already have '" *
                               o.input * "')"))
            end
            o.input = arg
        end
        i += 1
    end

    csv && csv_raw && throw(CliError("--csv and --csv-raw cannot be used together"))
    o.block_size != 256 && throw(CliError("the pinned configuration requires --block-size 256"))
    csv && (o.output = OutputCsv)
    csv_raw && (o.output = OutputCsvRaw)

    if !o.help && !o.list && isempty(o.input)
        throw(CliError("missing input file"))
    end

    return o
end

function help_text(program::AbstractString)::String
    return "Usage: " * program * " INPUT [OPTIONS]\n" *
           "\n" *
           "Hartree-Fock Fock-matrix build (two-electron integrals over s-type Gaussians).\n" *
           "INPUT is a geometry file, e.g. ../../data/hartree-fock/he64.\n" *
           "\n" *
           "Options:\n" *
           "  -n, --iters NUM        timed Fock builds after one untimed build (default 10)\n" *
           "      --double           float64 (default)\n" *
           "      --float            float32\n" *
           "      --block-size N     pinned threads per block; N must be 256 (default 256)\n" *
           "      --device INDEX     device to run on; the energy profiler uses the same\n" *
           "                         index (default 0)\n" *
           "      --list             list devices (index: name) and exit\n" *
           "      --check            always run the fp64 CPU reference check\n" *
           "      --no-check         skip the fp64 CPU reference check\n" *
           "      --max-check-atoms N  run the check automatically only up to N atoms\n" *
           "                         (default 128; the CPU reference grows as atoms^4)\n" *
           "      --csv              summary CSV on stdout, report on stderr\n" *
           "      --csv-raw          one CSV row per timed build on stdout, report on stderr\n" *
           "      --no-energy        skip board-power measurement\n" *
           "  -h, --help             show this help and exit\n" *
           "\n" *
           "The pinned configuration uses native-precision square roots. The last\n" *
           "precision flag wins. Options take their value as a separate\n" *
           "argument (-n 10, not --iters=10). Timed builds add into the same Fock\n" *
           "matrix; the checksum and the check use the untimed first build.\n" *
           "\n" *
           "Environment (all optional):\n" *
           "  GPU_NAME               device name to report instead of the queried one\n" *
           "  GPU_DRIVER             driver string to report instead of the queried one\n" *
           "\n" *
           "Backend (this run: " * lowercase(backend_name()) * "):\n" *
           "  --project=env/cuda     Julia environment for CUDA (CUDA.jl)\n" *
           "  --project=env/hip      Julia environment for HIP (AMDGPU.jl)\n"
end
