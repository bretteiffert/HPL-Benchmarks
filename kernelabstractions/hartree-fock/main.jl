# main.jl — Hartree-Fock (KernelAbstractions): wiring, validation, run loop.
#
#   julia --project=env/cuda -O3 --check-bounds=no main.jl INPUT [OPTIONS]
#   julia --project=env/hip  -O3 --check-bounds=no main.jl INPUT [OPTIONS]

include(joinpath(@__DIR__, "backend.jl"))
include(joinpath(@__DIR__, "energy.jl"))
include(joinpath(@__DIR__, "cli.jl"))
include(joinpath(@__DIR__, "kernels.jl"))
include(joinpath(@__DIR__, "hf.jl"))
include(joinpath(@__DIR__, "report.jl"))

using KernelAbstractions
using .Backend, .BoardEnergy, .CLI, .Kernels, .HF, .Reports

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

"Validate against device limits before allocating."
function validate(a::Args, nnnn::Int)
    max_wg = max_workgroup_size()
    a.block_size <= max_wg ||
        fail("block size $(a.block_size) exceeds the device maximum of $max_wg work-items per workgroup")
    groups = cld(nnnn + 1, a.block_size)
    max_g = max_groups()
    groups <= max_g ||
        fail("$(nnnn + 1) work items need $groups workgroups of $(a.block_size); the device maximum " *
             "is $max_g (the pinned block size is 256; use a smaller input)")
    # Float atomic add (Atomix) is native or a CAS loop on both backends for
    # float32/float64; float16 is rejected by the CLI.
    return nothing
end

"Check the device erf against SpecialFunctions on the host before the run."
function check_device_erf(::Type{T}, backend) where {T}
    xs = T[0.1, 0.5, 1.0, 2.0, 3.0]
    d_x = KernelAbstractions.allocate(backend, T, length(xs))
    copyto!(d_x, xs)
    d_out = KernelAbstractions.zeros(backend, T, length(xs))
    got = try
        erf_selftest_kernel!(backend, length(xs))(d_out, d_x; ndrange = length(xs))
        device_sync()
        Array(d_out)
    catch e
        throw(UserError("device erf failed on $(backend_label()): $(sprint(showerror, e))"))
    end
    tol = 64 * eps(T)
    for (x, g) in zip(xs, got)
        ref = Kernels.SpecialFunctions.erf(Float64(x))
        abs(Float64(g) - ref) <= tol * abs(ref) ||
            throw(UserError("device erf($x) = $g differs from the host value $ref " *
                            "beyond $(precision_name(T)) tolerance"))
    end
end

function correctness!(r, ::Type{T}, S, h_fock::Vector{T}, a::Args) where {T}
    erep = contract_erep(h_fock, S.h_dens, S.natoms)
    e_inpipe = Float64(erep) * 0.5
    r.e2_energy = e_inpipe
    pname = precision_name(T)

    run = a.check === :force || (a.check === :auto && S.natoms <= a.max_check_atoms)
    if !run
        push!(r.metrics, Metric("2e- energy ($pname Fock, $pname reduction)", e_inpipe, 15))
        push!(r.check_notes, a.check === :off ?
            "fp64 CPU reference check skipped (--no-check)" :
            "fp64 CPU reference check skipped: $(S.natoms) atoms > --max-check-atoms " *
            "$(a.max_check_atoms) (use --check to force)")
        return
    end

    fock_gpu = Float64.(h_fock)
    fock_ref = fock_build_cpu(S.ngauss, S.natoms, S.geom_d, S.xpnt_d, S.coef_d, S.dens_d)

    e_gpu = energy_from_fock(fock_gpu, S.dens_d, S.natoms)
    e_ref = energy_from_fock(fock_ref, S.dens_d, S.natoms)
    denom = abs(e_ref) > 0.0 ? abs(e_ref) : 1.0
    rel_loss = abs(e_gpu - e_ref) / denom
    rel_inpipe = abs(e_inpipe - e_ref) / denom
    r.rel_err_ref = rel_inpipe

    maxd = 0.0; sumsq = 0.0; fscale = 0.0
    for t in 1:S.nsq
        d = abs(fock_gpu[t] - fock_ref[t])
        d > maxd && (maxd = d)
        sumsq += d * d
        abs(fock_ref[t]) > fscale && (fscale = abs(fock_ref[t]))
    end
    rmsd = sqrt(sumsq / S.nsq)
    fscale == 0.0 && (fscale = 1.0)

    asym = 0.0
    n = S.natoms
    for i in 0:(n - 1), j in 0:(n - 1)
        d = abs(fock_gpu[i * n + j + 1] - fock_gpu[j * n + i + 1])
        d > asym && (asym = d)
    end

    append!(r.metrics, [
        Metric("2e- energy ($pname Fock, $pname reduction)", e_inpipe, 15),
        Metric("2e- energy ($pname Fock, fp64 reduction)", e_gpu, 15),
        Metric("2e- energy (fp64 CPU reference)", e_ref, 15),
        Metric("rel err vs reference, Fock build only", rel_loss, 6),
        Metric("rel err vs reference, incl. reduction", rel_inpipe, 6),
        Metric("max |F - F_ref|", maxd, 6),
        Metric("max |F - F_ref| / max |F_ref|", maxd / fscale, 6),
        Metric("rms |F - F_ref|", rmsd, 6),
        Metric("max |F(i,j) - F(j,i)|", asym, 6),
        Metric("2e- energy from symmetrized F (diagnostic)", energy_symmetrized(fock_gpu, S.dens_d, n), 15),
    ])
end

function run_benchmark(::Type{T}, a::Args) where {T}
    legacy = Val(a.legacy_sqrt)
    S = host_setup(T, a.input, legacy)
    validate(a, S.nnnn)

    notes = String[]
    name, name_note = device_name(a.device)
    driver, driver_note = device_driver()
    isempty(name_note) || push!(notes, name_note)
    isempty(driver_note) || push!(notes, driver_note)

    r = Report(backend = backend_label(), device = name, driver = driver,
               precision = precision_name(T),
               sqrt_mode = a.legacy_sqrt ? "legacy-sqrtf" : "native",
               input = basename(a.input), natoms = S.natoms, ngauss = S.ngauss,
               nnnn = S.nnnn, block_size = a.block_size, num_times = a.num_times,
               timing = timing_method(), energy_enabled = a.energy, header_notes = notes)

    csv_mode = a.csv || a.csv_raw
    print_header(csv_mode ? stderr : stdout, r)

    backend = ka_backend()
    check_device_erf(T, backend)
    d = HFDevice(backend, S, a.block_size, legacy)

    # Untimed run: compiles the kernel and gives the Fock matrix that is checked
    # (timed runs keep accumulating into it, matching the native benchmark).
    launch_fock!(d)
    h_fock = read_fock(d)

    times = Vector{Float64}(undef, a.num_times)
    timed_loop() = for it in 1:a.num_times
        times[it] = timed_fock!(d)
    end
    if a.energy
        r.energy = measure_energy(timed_loop, a.device, current_stream(), a.num_times)
        r.energy.samples < 2 && push!(r.energy_notes,
            "only $(r.energy.samples) power sample(s) in $(Reports.fixed(r.energy.time_s, 6)) s; " *
            "energy figures are unreliable (use a larger -n or input)")
    else
        timed_loop()
    end
    device_sync()

    for t in times
        (isfinite(t) && t > 0) ||
            throw(UserError("a kernel timing measurement is missing (got $t s)"))
    end
    r.times = times

    correctness!(r, T, S, h_fock, a)

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
    elseif e isa UserError || e isa EnergyError || e isa InputError
        println(stderr, "error: ", sprint(showerror, e))
        1
    else
        rethrow()
    end
end
flush(stdout)
exit(EXIT_CODE)
