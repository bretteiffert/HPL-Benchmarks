# Hartree-Fock driver (Julia / KernelAbstractions): parses the command line,
# reads the input file, validates the configuration against the device
# limits before allocating anything, runs one untimed Fock build (checksum
# and correctness source) and then num_times event-timed builds inside one
# energy measurement, runs the fp64 CPU reference check outside the measured
# window, and hands one report struct to the formatters.
#
# Run as  julia --project=env/<cuda|hip> main.jl INPUT [OPTIONS]; the
# environment picks the backend.

module HartreeFockKA

include("backend.jl")
include("deck.jl")
include("kernels.jl")
include("hf.jl")
include("cli.jl")
include("energy.jl")
include("report.jl")

# Throws CliError naming the limit that a setting breaks.
function validate(o::Options, dev::DeviceInfo, deck::Deck)
    device = "device " * string(dev.index) * " (" * dev.name * ")"

    if o.precision == PrecisionDouble && !fp64_atomic_add_supported(dev.index)
        throw(CliError("float64 needs atomicAdd on double, which " * device *
                       " does not support; use --float"))
    end

    if dev.max_threads_per_block > 0 && o.block_size > dev.max_threads_per_block
        throw(CliError("block size " * string(o.block_size) * " exceeds the " *
                       string(dev.max_threads_per_block) * " threads per block allowed on " * device))
    end

    # quartet_count uses 64-bit (nn * nn + nn) / 2: safe up to 65,535 atoms.
    if deck.natoms > 65535
        throw(CliError("input '" * deck.path * "' has " * string(deck.natoms) *
                       " atoms; the quartet index fits 64 bits only up to 65,535 atoms"))
    end

    grid = grid_blocks(deck.natoms, UInt32(o.block_size))
    max_grid = dev.max_grid_x > 0 ? UInt64(dev.max_grid_x) : UInt64(2147483647)
    if grid > max_grid
        throw(CliError("grid of " * string(grid) * " blocks for " * string(deck.natoms) *
                       " atoms exceeds the device maximum of " * string(max_grid) *
                       "; the pinned block size is 256, so use a smaller input"))
    end
end

function run_benchmark(o::Options)::Int
    csv_mode = o.output != OutputText
    text_out = csv_mode ? stderr : stdout

    count = device_count()
    count == 0 && error("no " * backend_name() * " devices found")
    if o.device >= count
        throw(CliError("device index " * string(o.device) * " is out of range (" *
                       string(count) * (count == 1 ? " device" : " devices") * " found; see --list)"))
    end

    dev = query_device(o.device)
    deck = read_deck(o.input)

    validate(o, dev, deck)

    r = Report()
    r.implementation = "KernelAbstractions"
    r.backend = backend_name()
    r.device = dev.name
    r.driver = dev.driver
    r.precision = o.precision
    r.sqrt_mode = o.sqrt_mode
    r.input = deck.name
    r.natoms = deck.natoms
    r.ngauss = deck.ngauss
    r.pairs = pair_count(deck.natoms)
    r.quartets = quartet_count(deck.natoms)
    r.block_size = UInt32(o.block_size)
    r.grid = grid_blocks(deck.natoms, r.block_size)
    r.num_times = o.num_times
    r.timing = "device timestamps"
    r.header_notes = dev.notes
    r.fock.name = "Fock build"

    meter = EnergyMeter(o.device, o.energy)
    r.energy.enabled = enabled(meter)

    print_text_header(r, text_out)

    hf = make_hartree_fock(o.precision, o.sqrt_mode, deck, r.block_size)

    # Untimed first build: fills the Fock matrix used by the checksum and check.
    # (It also compiles the kernel, so compilation is never timed.)
    first_build!(hf)
    r.checksum = checksum(hf)

    start!(meter)
    for it in 1:o.num_times
        push!(r.fock.times, timed_build!(hf))
    end
    stop!(meter)
    r.energy = result(meter, o.num_times)

    # fp64 CPU reference, outside the measured window.
    if !o.check
        r.check_skipped = "--no-check"
    elseif !o.force_check && Int(deck.natoms) > o.max_check_atoms
        r.check_skipped = string(deck.natoms) * " atoms > --max-check-atoms " *
                          string(o.max_check_atoms) * "; use --check to force it"
    else
        r.check = check(hf)
        r.checked = true
    end

    if o.output == OutputText
        print_text_results(r, stdout)
    elseif o.output == OutputCsv
        print_csv(r, stdout)
    else
        print_csv_raw(r, stdout)
    end
    print_text_energy(r, text_out)
    print_text_correctness(r, text_out)

    close_meter!(meter)
    return 0
end

function report_error(msg::AbstractString, hint::Bool)
    print(stderr, "error: ", msg, hint ? " (try '--help')" : "", "\n")
    flush(stderr)
    return 1
end

function main(args::Vector{String})::Int
    program = "julia --project=env/" * lowercase(backend_name()) * " " * PROGRAM_FILE
    try
        o = parse_cli(args)

        if o.help
            print(stdout, help_text(program))
            flush(stdout)
            return 0
        end

        if o.list
            devices = list_devices()
            if isempty(devices)
                return report_error("no " * backend_name() * " devices found", false)
            end
            for (i, name) in devices
                print(stdout, i, ": ", name, "\n")
            end
            flush(stdout)
            return 0
        end

        return run_benchmark(o)
    catch e
        e isa CliError && return report_error(e.msg, true)
        e isa DeckError && return report_error(e.msg, true)
        e isa InterruptException && rethrow()
        return report_error(error_text(e), false)
    end
end

end # module

exit(HartreeFockKA.main(ARGS))
