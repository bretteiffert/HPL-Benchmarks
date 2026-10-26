# cli.jl — Hartree-Fock command line.

module CLI

export Args, CliError, parse_args, print_help

struct CliError <: Exception
    msg::String
end

Base.@kwdef mutable struct Args
    input::String = ""
    num_times::Int = 10
    precision::Symbol = :double        # :double | :float
    legacy_sqrt::Bool = false
    block_size::Int = 256
    device::Int = 0
    check::Symbol = :auto              # :auto | :force | :off
    max_check_atoms::Int = 128
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

"Parse argv into Args. Throws CliError for user errors."
function parse_args(argv::Vector{String})
    a = Args()
    positional = String[]
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
            throw(CliError("--half is not supported: the 1e-12 screening thresholds are below " *
                           "the float16 range"))
        elseif f == "--legacy-sqrt"
            a.legacy_sqrt = true
        elseif f == "--check"
            a.check = :force
        elseif f == "--no-check"
            a.check = :off
        elseif f == "--csv"
            a.csv = true
        elseif f == "--csv-raw"
            a.csv_raw = true
        elseif f == "--no-energy"
            a.energy = false
        elseif f == "--list"
            a.list = true
        elseif f == "-n"
            a.num_times = _int(_value(argv, i, f), f); i += 1
        elseif f == "--block-size"
            a.block_size = _int(_value(argv, i, f), f); i += 1
        elseif f == "--device"
            a.device = _int(_value(argv, i, f), f; min = 0); i += 1
        elseif f == "--max-check-atoms"
            a.max_check_atoms = _int(_value(argv, i, f), f; min = 0); i += 1
        elseif startswith(f, "--") && occursin('=', f)
            throw(CliError("use '$(split(f, '='; limit = 2)[1]) VALUE', not '$f'"))
        elseif startswith(f, "-") && length(f) > 1
            throw(CliError("unknown argument '$f'"))
        else
            push!(positional, f)
        end
        i += 1
    end
    a.csv && a.csv_raw && throw(CliError("--csv and --csv-raw are mutually exclusive"))
    length(positional) > 1 &&
        throw(CliError("expected one input file, got $(length(positional)): $(join(positional, ' '))"))
    isempty(positional) || (a.input = positional[1])
    if isempty(a.input) && !a.help && !a.list
        throw(CliError("missing input file"))
    end
    return a
end

function print_help(io::IO = stdout)
    print(io, """
    Usage: julia --project=env/<cuda|hip> main.jl INPUT [OPTIONS]

    Hartree-Fock Fock build (KernelAbstractions). The backend is the Julia
    environment: env/cuda (CUDA.jl) or env/hip (AMDGPU.jl).

    Arguments:
      INPUT                  input file (e.g. ../../data/hartree-fock/he128)

    Options:
      -n NUM                 timed iterations after 1 untimed run (default 10)
          --double           float64 (default)
          --float            float32
          --legacy-sqrt      Boys-function square roots in float32 (sqrt mode legacy-sqrtf)
          --block-size N     work-items per workgroup (default 256)
          --check            always run the fp64 CPU reference check
          --no-check         skip the fp64 CPU reference check
          --max-check-atoms N  run the check automatically up to N atoms (default 128)
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
