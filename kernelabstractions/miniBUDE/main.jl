# main.jl — miniBUDE (KernelAbstractions): wiring, validation, run loop.
#
#   julia --project=env/cuda -O3 --check-bounds=no main.jl [OPTIONS]
#   julia --project=env/hip  -O3 --check-bounds=no main.jl [OPTIONS]

include(joinpath(@__DIR__, "backend.jl"))
include(joinpath(@__DIR__, "energy.jl"))
include(joinpath(@__DIR__, "cli.jl"))
include(joinpath(@__DIR__, "kernels.jl"))
include(joinpath(@__DIR__, "deck.jl"))
include(joinpath(@__DIR__, "bude.jl"))
include(joinpath(@__DIR__, "report.jl"))

using Printf
using .Backend, .BoardEnergy, .CLI, .Kernels, .Deck, .Bude, .Reports

struct UserError <: Exception
    msg::String
end
Base.showerror(io::IO, e::UserError) = print(io, e.msg)

fail(msg) = throw(UserError(msg * " (try '--help')"))

precision_type(p::Symbol) = p === :double ? Float64 : Float32
precision_name(::Type{T}) where {T} = "float$(8 * sizeof(T))"

function list_devices()
    n = device_count()
    n == 0 && throw(UserError("no $(backend_label()) devices found"))
    for (i, name) in enumerate(device_names())
        println(i - 1, ": ", name)
    end
end

"Validate against device limits; returns (runnable configs, skip notes)."
function plan_configs(a::Args, n::Int)
    n <= typemax(Int32) ||
        fail("$n poses exceed the kernel's Int32 index range")
    max_wg = max_workgroup_size()
    for w in a.wgsize
        w <= max_wg ||
            fail("wgsize $w exceeds the device maximum of $max_wg work-items per workgroup")
    end
    max_g = max_groups()
    configs = Tuple{Int,Int}[]
    skipped = String[]
    for p in a.ppwi, w in a.wgsize
        if n % (p * w) != 0
            push!(skipped, "ppwi $p, wgsize $w")
            continue
        end
        groups = n ÷ (p * w)
        groups <= max_g ||
            fail("ppwi $p, wgsize $w needs $groups workgroups; the device maximum is $max_g")
        push!(configs, (p, w))
    end
    isempty(configs) &&
        fail("no (ppwi, wgsize) combination divides the $n poses; use -n, -p or -w")
    notes = isempty(skipped) ? String[] :
        ["skipped (pose count $n not divisible by ppwi × wgsize): $(join(skipped, "; "))"]
    return configs, notes
end

function run_config(::Type{T}, backend, d, a::Args, ppwi, wgsize, golden) where {T}
    dev = BudeDevice(T, backend, d, wgsize, ppwi)
    # At least one untimed launch: Julia compiles on first launch, and the
    # device events would otherwise include the compile time.
    for _ in 1:max(a.warmup, 1)
        launch!(dev)
    end
    device_sync()

    kernel_s = Vector{Float64}(undef, a.iterations)
    wall_s = Vector{Float64}(undef, a.iterations)
    timed_loop() = for it in 1:a.iterations
        kernel_s[it], wall_s[it] = launch!(dev)
    end
    energy = nothing
    if a.energy
        energy = measure_energy(timed_loop, a.device, current_stream(), a.iterations)
    else
        timed_loop()
    end
    device_sync()

    for t in kernel_s
        (isfinite(t) && t > 0) ||
            throw(UserError("a kernel timing measurement is missing (got $t s)"))
    end

    energies = read_energies(dev)
    avg_s = sum(kernel_s) / length(kernel_s)
    gflops, gfinsts, ginter = throughput(d, ppwi, avg_s)
    overhead = sum(wall_s .- kernel_s) / length(kernel_s)
    ref = golden === nothing ? nothing : compute_metrics(energies, golden, 8 * sizeof(T))
    deckm = compute_metrics(energies, Float64.(d.ref_energies[1:nposes(d)]), 64)
    return ConfigResult(ppwi, wgsize, kernel_s, gflops, gfinsts, ginter, overhead,
                        dev.context_s, energy, ref, deckm, energies), energies
end

function save_energies(path, energies)
    try
        open(path, "w") do io
            for e in energies
                @printf(io, "%.17g\n", e)
            end
        end
    catch e
        throw(UserError("cannot write energies to '$path': $(sprint(showerror, e))"))
    end
end

function run_benchmark(::Type{T}, a::Args) where {T}
    d = read_deck(a.deck, a.poses)
    configs, skip_notes = plan_configs(a, nposes(d))

    notes = String[]
    name, name_note = device_name(a.device)
    driver, driver_note = device_driver()
    isempty(name_note) || push!(notes, name_note)
    isempty(driver_note) || push!(notes, driver_note)
    a.warmup == 0 && push!(notes, "--warmup 0: one untimed launch per configuration still " *
                                  "compiles the kernel before timing")
    T === Float32 && push!(notes, "float32 results are compared with one untimed float64 pass " *
                                  "(ppwi $(configs[1][1]), wgsize $(configs[1][2]))")

    r = Report(backend = backend_label(), device = name, driver = driver,
               precision = precision_name(T), deck = basename(normpath(a.deck)),
               natlig = natlig(d), natpro = natpro(d), ntypes = ntypes(d),
               nposes = nposes(d), max_poses = d.max_poses, ppwi = a.ppwi, wgsize = a.wgsize,
               iterations = a.iterations, warmup = a.warmup, timing = timing_method(),
               energy_enabled = a.energy, rows = a.rows, header_notes = notes,
               result_notes = skip_notes)

    csv_mode = a.csv || a.csv_raw
    print_header(csv_mode ? stderr : stdout, r)

    backend = ka_backend()
    golden = nothing
    if T === Float32
        g = BudeDevice(Float64, backend, d, configs[1][2], configs[1][1])
        launch!(g)                       # one untimed float64 pass
        device_sync()
        golden = read_energies(g)
    end

    for (k, (ppwi, wgsize)) in enumerate(configs)
        res, energies = run_config(T, backend, d, a, ppwi, wgsize, golden)
        push!(r.configs, res)
        k == 1 && !isempty(a.out) && save_energies(a.out, energies)
    end

    if a.energy
        few = [c for c in r.configs if c.energy !== nothing && c.energy.samples < 2]
        isempty(few) || push!(r.energy_notes,
            "fewer than 2 power samples for $(join(map(c -> "p$(c.ppwi) w$(c.wgsize)", few), ", ")); " *
            "energy figures are unreliable (use a larger -i)")
    end
    if T === Float64
        push!(r.check_notes, "float64 is the reference precision; the Ref columns are empty")
    end
    push!(r.check_notes, "ref_energies.out is low precision; the Deck columns are informational " *
                         "(tolerance $(REPORT_TOLERANCE_PCT)%)")
    isempty(a.out) || push!(r.check_notes, "energies of ppwi $(configs[1][1]), wgsize " *
                                          "$(configs[1][2]) saved to $(a.out)")

    if a.csv
        print_csv(stdout, r)
        print_stderr_sections(stderr, r)
    elseif a.csv_raw
        print_csv_raw(stdout, r)
        print_stderr_sections(stderr, r)
    else
        print_text(stdout, r)
    end
    return nothing
end

function main(argv)
    args = parse_args(argv)
    if args.help
        print_help()
        return 0
    end
    isempty(backend_error()) || throw(UserError(backend_error()))
    if args.list
        list_devices()
        return 0
    end
    n = device_count()
    n == 0 && throw(UserError("no $(backend_label()) devices found"))
    args.device < n ||
        fail("device $(args.device) does not exist ($n device$(n == 1 ? "" : "s"); see --list)")
    select_device!(args.device)
    run_benchmark(precision_type(args.precision), args)
    return 0
end

const EXIT_CODE = try
    main(ARGS)
catch e
    if e isa CliError
        println(stderr, "error: ", e.msg, " (try '--help')")
        1
    elseif e isa UserError || e isa EnergyError || e isa DeckError
        println(stderr, "error: ", sprint(showerror, e))
        1
    else
        rethrow()
    end
end
flush(stdout)
exit(EXIT_CODE)
