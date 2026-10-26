# cli.jl — seven-point stencil command line.

module CLI

export Args, CliError, parse_args, print_help

struct CliError <: Exception
    msg::String
end

Base.@kwdef mutable struct Args
    grid::NTuple{3,Int} = (1024, 1024, 1024)
    num_times::Int = 100
    precision::Symbol = :double        # :double | :float | :half
    block::NTuple{3,Int} = (64, 4, 1)
    device::Int = 0
    csv::Bool = false
    csv_raw::Bool = false
    energy::Bool = true
    list::Bool = false
    help::Bool = false
end

function _value(argv, i, flag)
    i + 1 <= length(argv) || throw(CliError("$flag needs a value"))
    return argv[i+1]
end

function _int(s, flag; min = 1)
    v = tryparse(Int, strip(s))
    (v === nothing || v < min) &&
        throw(CliError("invalid value '$s' for $flag (expected an integer >= $min)"))
    return v
end

"Parse 'X', 'X,Y' or 'X,Y,Z'; missing trailing values are `fill`."
function _triple(s, flag; min, fill, need3 = false)
    parts = split(s, ',')
    (1 <= length(parts) <= 3) || throw(CliError("invalid value '$s' for $flag (expected X[,Y[,Z]])"))
    need3 && length(parts) != 3 && throw(CliError("invalid value '$s' for $flag (expected X,Y,Z)"))
    v = [_int(p, flag; min) for p in parts]
    while length(v) < 3
        push!(v, fill)
    end
    return (v[1], v[2], v[3])
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
        elseif f == "--list"
            a.list = true
        elseif f == "-s" || f == "--size"
            L = _int(_value(argv, i, f), f; min = 3); i += 1
            a.grid = (L, L, L)
        elseif f == "--grid"
            a.grid = _triple(_value(argv, i, f), f; min = 3, fill = 0, need3 = true); i += 1
        elseif f == "-n"
            a.num_times = _int(_value(argv, i, f), f); i += 1
        elseif f == "--block-size"
            a.block = _triple(_value(argv, i, f), f; min = 1, fill = 1); i += 1
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
    a.block == (64, 4, 1) || throw(CliError("the pinned configuration requires --block-size 64,4,1"))
    return a
end

function print_help(io::IO = stdout)
    print(io, """
    Usage: julia --project=env/<cuda|hip> main.jl [OPTIONS]

    Seven-point Laplacian stencil (KernelAbstractions). The backend is the Julia
    environment: env/cuda (CUDA.jl) or env/hip (AMDGPU.jl).

    Options:
      -s, --size L           cubic grid L × L × L, L >= 3 (default 1024)
          --grid X,Y,Z       non-cubic grid, each >= 3 (last of -s/--grid wins)
      -n NUM                 timed iterations after 1 untimed run (default 100)
          --double           float64 (default)
          --float            float32
          --half             float16 (only while 2·Σ(n-1)² <= 65,504, i.e. L <= 105)
          --block-size X,Y,Z      pinned workgroup shape; must be 64,4,1
          --device INDEX     device index; the energy profiler uses the same index (default 0)
          --list             list devices (index: name) and exit
          --csv              summary CSV on stdout (report on stderr)
          --csv-raw          one CSV row per timed iteration on stdout (report on stderr)
          --no-energy        skip board-power measurement
      -h, --help             show this help

    Environment (optional):
      GPU_NAME     override the reported device name
      GPU_DRIVER   override the reported driver string
    """)
end

end # module CLI
