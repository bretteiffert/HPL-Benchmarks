# main.jl — BabelStream (KernelAbstractions): wiring, validation, run loop.
#
#   julia --project=env/cuda -O3 --check-bounds=no main.jl [OPTIONS]
#   julia --project=env/hip  -O3 --check-bounds=no main.jl [OPTIONS]

include(joinpath(@__DIR__, "backend.jl"))
include(joinpath(@__DIR__, "energy.jl"))
include(joinpath(@__DIR__, "cli.jl"))
include(joinpath(@__DIR__, "kernels.jl"))
include(joinpath(@__DIR__, "stream.jl"))
include(joinpath(@__DIR__, "report.jl"))

using .Backend, .BoardEnergy, .CLI, .Kernels, .Stream, .Reports

struct UserError <: Exception
    msg::String
end
Base.showerror(io::IO, e::UserError) = print(io, e.msg)

fail(msg) = throw(UserError(msg * " (try '--help')"))

precision_type(p::Symbol) = p === :double ? Float64 : p === :float ? Float32 : Float16
precision_name(::Type{T}) where {T} = "float$(8 * sizeof(T))"

function list_devices()
    n = device_count()
    n == 0 && throw(UserError("no $(backend_label()) devices found"))
    for (i, name) in enumerate(device_names())
        println(i - 1, ": ", name)
    end
end

"Validate against device limits before allocating. Returns (dot_groups, desc, notes)."
function validate(::Type{T}, a::Args) where {T}
    notes = String[]
    if T === Float16 && !supports_half()
        fail("float16 is not supported on this device (needs compute capability 5.3 or newer)")
    end
    max_wg = max_workgroup_size()
    a.block_size <= max_wg ||
        fail("block size $(a.block_size) exceeds the device maximum of $max_wg work-items per workgroup")
    DOT_BLOCK <= max_wg ||
        fail("the Dot kernel needs $DOT_BLOCK work-items per workgroup; the device maximum is $max_wg")
    a.array_size % a.block_size == 0 ||
        fail("array size $(a.array_size) must be a multiple of the block size $(a.block_size) " *
             "(the kernels have no bounds check)")

    cu = compute_unit_count()
    unit = compute_unit_label()
    if cu === nothing
        dot_groups = 64
        desc = "64 (fallback)"
        push!(notes, "could not query the $unit count; Dot uses 64 groups")
    else
        dot_groups = 4 * cu
        desc = "$dot_groups (4 × $cu $unit)"
    end

    max_g = max_groups()
    a.array_size ÷ a.block_size <= max_g ||
        fail("array size $(a.array_size) needs $(a.array_size ÷ a.block_size) workgroups; " *
             "the device maximum is $max_g")
    dot_groups <= max_g ||
        fail("Dot needs $dot_groups workgroups; the device maximum is $max_g")
    return dot_groups, desc, notes
end

function correctness!(r, ::Type{T}, ntimes, a, b, c, dot_sum) where {T}
    goldA, goldB, goldC = gold_values(T, ntimes)
    for (name, v, g) in (("a", a, goldA), ("b", b, goldB), ("c", c, goldC))
        chk, mx = array_metrics(v, g)
        push!(r.checks, ArrayCheck(name, Float64(g), chk, mx))
    end

    expected_dot = Float64(goldA) * Float64(goldB) * Float64(r.array_size)
    r.dot_expected = expected_dot
    r.dot_actual = dot_sum
    r.dot_rel_err = expected_dot != 0.0 ?
        abs(dot_sum - expected_dot) / abs(expected_dot) : abs(dot_sum)

    pname = precision_name(T)
    small = [n for (n, g) in (("a", goldA), ("b", goldB), ("c", goldC))
             if g != 0 && abs(Float64(g)) < Float64(floatmin(T))]
    if !isempty(small)
        push!(r.check_notes, "expected value of $(join(small, ", ")) is below the smallest normal " *
              "$pname after $ntimes iterations; relative errors are not meaningful")
    end
    product = abs(Float64(goldA) * Float64(goldB))
    if product < Float64(floatmin(T))
        push!(r.check_notes, "expected element product $(Reports.sci(product, 3)) underflows " *
              "$pname; the dot relative error is not meaningful")
    end
    if abs(expected_dot) > Float64(floatmax(T))
        fmax = T === Float16 ? "65,504" : Reports.sci(Float64(floatmax(T)), 3)
        push!(r.check_notes, "expected dot exceeds the $pname maximum ($fmax); " *
              "the dot relative error is not meaningful")
    end
end

function run_benchmark(::Type{T}, a::Args) where {T}
    dot_groups, groups_desc, notes = validate(T, a)

    name, name_note = device_name(a.device)
    driver, driver_note = device_driver()
    isempty(name_note) || push!(notes, name_note)
    isempty(driver_note) || push!(notes, driver_note)
    push!(notes, "Dot time includes copying the partial sums to the host")

    r = Report(backend = backend_label(), device = name, driver = driver,
               precision = precision_name(T), elem_bytes = sizeof(T),
               array_size = a.array_size, block_size = a.block_size, dot_block = DOT_BLOCK,
               dot_groups = dot_groups, dot_groups_desc = groups_desc,
               num_times = a.num_times, timing = timing_method(),
               energy_enabled = a.energy, mibibytes = a.mibibytes, header_notes = notes)

    csv_mode = a.csv || a.csv_raw
    report_io = csv_mode ? stderr : stdout
    print_header(report_io, r)

    s = KAStream{T}(ka_backend(), a.array_size, a.block_size, dot_groups)
    warmup!(s)

    t_init = init_arrays!(s)

    # Iteration 1: warm-up, excluded from statistics, CSV and energy.
    ka_copy!(s); ka_mul!(s); ka_add!(s); ka_triad!(s)
    _, first_dot = ka_dot!(s)
    dot_ref = Ref(first_dot)   # Ref, not a reassigned capture, keeps the closure type-stable

    timed = a.num_times - 1
    times = [Vector{Float64}(undef, timed) for _ in 1:5]
    function timed_loop()
        for it in 1:timed
            times[1][it] = ka_copy!(s)
            times[2][it] = ka_mul!(s)
            times[3][it] = ka_add!(s)
            times[4][it] = ka_triad!(s)
            times[5][it], dot_ref[] = ka_dot!(s)
        end
    end

    if a.energy
        r.energy = measure_energy(timed_loop, a.device, current_stream(), timed)
        r.energy.samples < 2 && push!(r.energy_notes,
            "only $(r.energy.samples) power sample(s) in $(Reports.fixed(r.energy.time_s, 6)) s; " *
            "energy figures are unreliable (use a larger -n or -s)")
    else
        timed_loop()
    end
    device_sync()

    for ts in times, t in ts
        (isfinite(t) && t > 0) ||
            throw(UserError("a kernel timing measurement is missing (got $t s)"))
    end
    (isfinite(t_init) && t_init > 0) ||
        throw(UserError("the Init timing measurement is missing (got $t_init s)"))

    bytes = routine_bytes(T, a.array_size)
    push!(r.routines, RoutineResult("Init", bytes.Init, [t_init], true))
    for (k, nm) in enumerate(("Copy", "Mul", "Add", "Triad", "Dot"))
        push!(r.routines, RoutineResult(nm, getfield(bytes, Symbol(nm)), times[k], false))
    end

    ha, hb, hc = read_arrays(s)
    correctness!(r, T, a.num_times, ha, hb, hc, dot_ref[])

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
    elseif e isa UserError || e isa EnergyError
        println(stderr, "error: ", sprint(showerror, e))
        1
    else
        rethrow()
    end
end
flush(stdout)
exit(EXIT_CODE)
