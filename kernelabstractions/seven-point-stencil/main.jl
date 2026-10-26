# main.jl — seven-point stencil (KernelAbstractions): wiring, validation, run loop.
#
#   julia --project=env/cuda -O3 --check-bounds=no main.jl [OPTIONS]
#   julia --project=env/hip  -O3 --check-bounds=no main.jl [OPTIONS]

include(joinpath(@__DIR__, "backend.jl"))
include(joinpath(@__DIR__, "energy.jl"))
include(joinpath(@__DIR__, "cli.jl"))
include(joinpath(@__DIR__, "kernels.jl"))
include(joinpath(@__DIR__, "stencil.jl"))
include(joinpath(@__DIR__, "report.jl"))

using .Backend, .BoardEnergy, .CLI, .Kernels, .Stencil, .Reports

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

"Validate before allocating; returns the workgroup count."
function validate(::Type{T}, a::Args) where {T}
    nx, ny, nz = a.grid
    if T === Float16
        supports_half() ||
            fail("float16 is not supported on this device (needs compute capability 5.3 or newer)")
        # invhxyz2 = -2·Σ(n-1)² must be representable in float16.
        c = 2 * ((nx - 1)^2 + (ny - 1)^2 + (nz - 1)^2)
        c <= 65504 ||
            fail("float16 overflows for this grid: the centre coefficient 2·Σ(n-1)² = " *
                 "$c exceeds 65,504 (use L <= 105 per side, or --float)")
    end
    bx, by, bz = a.block
    threads = bx * by * bz
    max_wg = max_workgroup_size()
    threads <= max_wg ||
        fail("block size $bx × $by × $bz = $threads exceeds the device maximum of $max_wg " *
             "work-items per workgroup")
    groups = cld(nx, bx) * cld(ny, by) * cld(nz, bz)
    max_g = max_groups()
    groups <= max_g ||
        fail("the grid needs $groups workgroups; the device maximum is $max_g")
    return groups
end

function run_benchmark(::Type{T}, a::Args) where {T}
    groups = validate(T, a)
    nx, ny, nz = a.grid
    fetch_b, write_b = traffic_bytes(T, nx, ny, nz)

    notes = String[]
    name, name_note = device_name(a.device)
    driver, driver_note = device_driver()
    isempty(name_note) || push!(notes, name_note)
    isempty(driver_note) || push!(notes, driver_note)

    r = Report(backend = backend_label(), device = name, driver = driver,
               precision = precision_name(T), elem_bytes = sizeof(T), nx = nx, ny = ny, nz = nz,
               block = a.block, workgroups = groups, fetch_bytes = fetch_b, write_bytes = write_b,
               num_times = a.num_times, timing = timing_method(), energy_enabled = a.energy,
               header_notes = notes)

    csv_mode = a.csv || a.csv_raw
    print_header(csv_mode ? stderr : stdout, r)

    s = StencilDevice{T}(ka_backend(), nx, ny, nz, a.block)
    init_field!(s)
    laplacian!(s)                 # untimed run: compiles the kernel, gives the checked field
    device_sync()
    chk = check_interior(s)
    r.max_abs_err, r.mean_abs_err = chk.max_abs_err, chk.mean_abs_err
    r.l2_rel_err, r.points_checked = chk.l2_rel_err, chk.num_checked
    T === Float16 && push!(r.check_notes,
        "float16 loses the result to cancellation: terms of size up to 2·Σ(n-1)² " *
        "(here $(2 * ((nx - 1)^2 + (ny - 1)^2 + (nz - 1)^2))) cancel to 3, so errors of order 1-10 are expected")

    times = Vector{Float64}(undef, a.num_times)
    timed_loop() = for it in 1:a.num_times
        times[it] = timed_laplacian!(s)
    end
    if a.energy
        r.energy = measure_energy(timed_loop, a.device, current_stream(), a.num_times)
        r.energy.samples < 2 && push!(r.energy_notes,
            "only $(r.energy.samples) power sample(s) in $(Reports.fixed(r.energy.time_s, 6)) s; " *
            "energy figures are unreliable (use a larger -n or grid)")
    else
        timed_loop()
    end
    device_sync()

    for t in times
        (isfinite(t) && t > 0) ||
            throw(UserError("a kernel timing measurement is missing (got $t s)"))
    end
    r.times = times

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
