# cli.jl — miniBUDE command line (upstream miniBUDE flag names).

module CLI

export Args, CliError, parse_args, print_help, DEFAULT_DECK

struct CliError <: Exception
    msg::String
end

const DEFAULT_DECK = normpath(joinpath(@__DIR__, "..", "..", "data", "miniBUDE", "bm1"))
const PPWIS = (1, 2, 4, 8, 16, 32, 64, 128)

Base.@kwdef mutable struct Args
    iterations::Int = 8
    warmup::Int = 2
    poses::Int = 0                     # 0 = all poses in the deck
    ppwi::Vector{Int} = [1]
    wgsize::Vector{Int} = [1]
    deck::String = DEFAULT_DECK
    out::String = ""
    rows::Int = 8
    precision::Symbol = :double        # :double | :float
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
    v = tryparse(Int, s)
    (v === nothing || v < min) &&
        throw(CliError("invalid value '$s' for $flag (expected an integer >= $min)"))
    return v
end

_list(s, flag) = [_int(strip(t), flag) for t in split(s, ',')]

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
            throw(CliError("--half is not supported by miniBUDE (use --double or --float)"))
        elseif f == "--csv"
            a.csv = true
        elseif f == "--csv-raw"
            a.csv_raw = true
        elseif f == "--no-energy"
            a.energy = false
        elseif f == "--list"
            a.list = true
        elseif f == "-i" || f == "--iter"
            a.iterations = _int(_value(argv, i, f), f); i += 1
        elseif f == "--warmup"
            a.warmup = _int(_value(argv, i, f), f; min = 0); i += 1
        elseif f == "-n" || f == "--poses"
            a.poses = _int(_value(argv, i, f), f); i += 1
        elseif f == "-p" || f == "--ppwi"
            v = _value(argv, i, f); i += 1
            if v == "all"
                a.ppwi = collect(PPWIS)
            else
                a.ppwi = _list(v, f)
                for p in a.ppwi
                    p in PPWIS || throw(CliError("invalid PPWI $p for $f (valid: " *
                                                 "$(join(PPWIS, ", ")) or all)"))
                end
            end
        elseif f == "-w" || f == "--wgsize"
            a.wgsize = _list(_value(argv, i, f), f); i += 1
        elseif f == "--deck"
            a.deck = _value(argv, i, f); i += 1
        elseif f == "-o" || f == "--out"
            a.out = _value(argv, i, f); i += 1
        elseif f == "-r" || f == "--rows"
            a.rows = _int(_value(argv, i, f), f; min = 0); i += 1
        elseif f == "-d" || f == "--device"
            a.device = _int(_value(argv, i, f), f; min = 0); i += 1
        elseif startswith(f, "--") && occursin('=', f)
            throw(CliError("use '$(split(f, '='; limit = 2)[1]) VALUE', not '$f'"))
        else
            throw(CliError("unknown argument '$f'"))
        end
        i += 1
    end
    a.csv && a.csv_raw && throw(CliError("--csv and --csv-raw are mutually exclusive"))
    unique!(a.ppwi)
    unique!(a.wgsize)
    return a
end

function print_help(io::IO = stdout)
    print(io, """
    Usage: julia --project=env/<cuda|hip> main.jl [OPTIONS]

    miniBUDE (KernelAbstractions). The backend is the Julia environment:
    env/cuda (CUDA.jl) or env/hip (AMDGPU.jl). One run sweeps every
    (PPWI, wgsize) combination; combinations that do not divide the pose
    count are skipped with a note.

    Options:
      -i, --iter N           timed iterations (default 8)
          --warmup N         warm-up iterations, excluded from everything (default 2)
      -n, --poses N          poses to use (default: all poses in the deck)
      -p, --ppwi LIST|all    poses per work-item, from 1,2,4,8,16,32,64,128 (default 1)
      -w, --wgsize LIST      work-group sizes (default 1)
          --deck DIR         input deck (default ../../data/miniBUDE/bm1)
      -o, --out PATH         save the first configuration's energies
      -r, --rows N           energies to print in the text report (default 8)
          --double           float64 (default)
          --float            float32; also runs one untimed float64 pass as the reference
      -d, --device INDEX     device index; the energy profiler uses the same index (default 0)
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
