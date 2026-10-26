# cli.jl — BabelStream command line.

module CLI

export Args, CliError, parse_args, print_help

struct CliError <: Exception
    msg::String
end

Base.@kwdef mutable struct Args
    array_size::Int = 33_554_432
    num_times::Int = 1000
    precision::Symbol = :double        # :double | :float | :half
    block_size::Int = 256
    device::Int = 0
    csv::Bool = false
    csv_raw::Bool = false
    energy::Bool = true
    mibibytes::Bool = false
    list::Bool = false
    help::Bool = false
end

function _value(argv, i, flag)
    i + 1 <= length(argv) || throw(CliError("$flag needs a value"))
    return argv[i+1]
end

function _int(s, flag; min = 1)
    v = tryparse(Int, s)
    (v === nothing || v < min) &&
        throw(CliError("invalid value '$s' for $flag (expected an integer >= $min)"))
    return v
end

"Parse argv into Args. Throws CliError for user errors."
function parse_args(argv::Vector{String})
    a = Args()
    i = 1
    while i <= length(argv)
        f = argv[i]
        if f == "-h" || f == "--help"
            a.help = true
        elseif f == "--double"
            a.precision = :double
        elseif f == "--float"
            a.precision = :float
        elseif f == "--half"
            a.precision = :half
        elseif f == "--csv"
            a.csv = true
        elseif f == "--csv-raw"
            a.csv_raw = true
        elseif f == "--no-energy"
            a.energy = false
        elseif f == "--mibibytes"
            a.mibibytes = true
        elseif f == "--list"
            a.list = true
        elseif f == "-s" || f == "--arraysize"
            a.array_size = _int(_value(argv, i, f), f); i += 1
        elseif f == "-n" || f == "--numtimes"
            a.num_times = _int(_value(argv, i, f), f; min = 2); i += 1
        elseif f == "--block-size"
            a.block_size = _int(_value(argv, i, f), f); i += 1
        elseif f == "--device"
            a.device = _int(_value(argv, i, f), f; min = 0); i += 1
        elseif startswith(f, "--") && occursin('=', f)
            throw(CliError("use '$(split(f, '='; limit = 2)[1]) VALUE', not '$f'"))
        else
            throw(CliError("unknown argument '$f'"))
        end
        i += 1
    end
    a.csv && a.csv_raw && throw(CliError("--csv and --csv-raw are mutually exclusive"))
    a.block_size == 256 || throw(CliError("the pinned configuration requires --block-size 256"))
    return a
end

function print_help(io::IO = stdout)
    print(io, """
    Usage: julia --project=env/<cuda|hip> main.jl [OPTIONS]

    BabelStream (KernelAbstractions). The backend is the Julia environment:
    env/cuda (CUDA.jl) or env/hip (AMDGPU.jl).

    Options:
      -s, --arraysize SIZE   elements per array (default 33554432)
      -n, --numtimes NUM     iterations, NUM >= 2; the first is a warm-up and is
                             excluded from statistics, CSV and energy (default 1000)
          --double           float64 (default)
          --float            float32
          --half             float16
          --block-size N     pinned work-items per workgroup for every kernel;
                             N must be 256 (default 256).
          --device INDEX     device index; the energy profiler uses the same index (default 0)
          --list             list devices (index: name) and exit
          --csv              summary CSV on stdout (report on stderr)
          --csv-raw          one CSV row per timed iteration on stdout (report on stderr)
          --no-energy        skip board-power measurement
          --mibibytes        text report in MiB/s and MiB (CSV stays GB/s)
      -h, --help             show this help

    Environment (optional):
      GPU_NAME     override the reported device name
      GPU_DRIVER   override the reported driver string
    """)
end

end # module CLI
