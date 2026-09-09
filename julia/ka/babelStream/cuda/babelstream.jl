# Runs BabelStream copy, multiplication, addition, triad, and dot kernels through KernelAbstractions on the backend selected by `STENCIL_GPU`. The source currently selects Float64 with `USE_FLOAT = false`; `--float` is accepted but does not change that constant, while the CLI controls array size, iteration count, device, units, listing, and CSV output. GPU events time kernels, the dot path includes its partial-sum transfer, and host-side recurrence checks report numerical error.

using KernelAbstractions
using Printf

include(joinpath(@__DIR__, "../gpu_shim.jl"))

const IMPLEMENTATION_STRING = "KernelAbstractions.jl"
const VERSION_STRING = "5.0"

const startA = 0.1
const startB = 0.2
const startC = 0.0
const startScalar = 0.4

const TBSIZE = 1024

const USE_FLOAT = false

@kernel unsafe_indices = true function init_kernel!(a, b, c, initA, initB, initC)
    I = @index(Global)
    @inbounds begin
        a[I] = initA
        b[I] = initB
        c[I] = initC
    end
end

@kernel unsafe_indices = true function copy_kernel!(c, @Const(a))
    I = @index(Global)
    @inbounds c[I] = a[I]
end

@kernel unsafe_indices = true function mul_kernel!(b, @Const(c), scalar)
    I = @index(Global)
    @inbounds b[I] = scalar * c[I]
end

@kernel unsafe_indices = true function add_kernel!(c, @Const(a), @Const(b))
    I = @index(Global)
    @inbounds c[I] = a[I] + b[I]
end

@kernel unsafe_indices = true function triad_kernel!(a, @Const(b), @Const(c), scalar)
    I = @index(Global)
    @inbounds a[I] = b[I] + scalar * c[I]
end

@kernel unsafe_indices = true function dot_kernel!(@Const(a), @Const(b), partial, array_size, num_blocks)
    T = eltype(partial)
    tb_sum = @localmem T (1024,)

    local_i = @index(Local)
    group_i = @index(Group)

    @inbounds tb_sum[local_i] = zero(T)

    i = @index(Global)
    stride = 1024 * num_blocks
    while i <= array_size
        @inbounds tb_sum[local_i] = tb_sum[local_i] + a[i] * b[i]
        i += stride
    end

    offset = 512
    while offset > 0
        @synchronize
        if local_i <= offset
            @inbounds tb_sum[local_i] += tb_sum[local_i+offset]
        end
        offset ÷= 2
    end

    if local_i == 1
        @inbounds partial[group_i] = tb_sum[1]
    end
end

mutable struct KAStream{T}
    array_size::Int
    dot_num_blocks::Int
    backend::KernelAbstractions.Backend

    d_a::AbstractVector{T}
    d_b::AbstractVector{T}
    d_c::AbstractVector{T}
    d_sum::AbstractVector{T}

    sums::Vector{T}

    last_kernel_time::Float64
end

function KAStream{T}(array_size::Int, device_index::Int) where {T}
    if array_size % TBSIZE != 0
        error("Array size must be a multiple of $TBSIZE")
    end

    gpu_set_device!(device_index)

    println("Using $(gpu_label()) device ", gpu_name())
    println("Driver: ", gpu_driver())

    backend = gpu_backend()
    cu_count = gpu_cu_count()
    dot_num_blocks = cu_count * 4

    println("Reduction kernel config: $dot_num_blocks groups of (fixed) size $TBSIZE")

    d_a = KernelAbstractions.zeros(backend, T, array_size)
    d_b = KernelAbstractions.zeros(backend, T, array_size)
    d_c = KernelAbstractions.zeros(backend, T, array_size)
    d_sum = KernelAbstractions.zeros(backend, T, dot_num_blocks)

    sums = Vector{T}(undef, dot_num_blocks)

    s = KAStream{T}(array_size, dot_num_blocks, backend, d_a, d_b, d_c, d_sum, sums, 0.0)
    warmup!(s)
    return s
end

function warmup!(s::KAStream{T}) where {T}
    n = TBSIZE
    w_a = KernelAbstractions.zeros(s.backend, T, n)
    w_b = KernelAbstractions.zeros(s.backend, T, n)
    w_c = KernelAbstractions.zeros(s.backend, T, n)
    w_sum = KernelAbstractions.zeros(s.backend, T, s.dot_num_blocks)

    init_kernel!(s.backend, TBSIZE)(w_a, w_b, w_c, T(startA), T(startB), T(startC); ndrange=n)
    copy_kernel!(s.backend, TBSIZE)(w_c, w_a; ndrange=n)
    mul_kernel!(s.backend, TBSIZE)(w_b, w_c, T(startScalar); ndrange=n)
    add_kernel!(s.backend, TBSIZE)(w_c, w_a, w_b; ndrange=n)
    triad_kernel!(s.backend, TBSIZE)(w_a, w_b, w_c, T(startScalar); ndrange=n)
    dot_kernel!(s.backend, TBSIZE)(w_a, w_b, w_sum, n, s.dot_num_blocks;
        ndrange=TBSIZE * s.dot_num_blocks)
    copyto!(s.sums, w_sum)
    gpu_sync()
    return nothing
end

function init_arrays!(s::KAStream{T}, initA::T, initB::T, initC::T) where {T}
    s.last_kernel_time = gpu_elapsed() do
        init_kernel!(s.backend, TBSIZE)(s.d_a, s.d_b, s.d_c, initA, initB, initC;
            ndrange=s.array_size)
    end
    return nothing
end

function read_arrays!(s::KAStream{T}, a::Vector{T}, b::Vector{T}, c::Vector{T}) where {T}
    copyto!(a, s.d_a)
    copyto!(b, s.d_b)
    copyto!(c, s.d_c)
    return nothing
end

function ka_copy!(s::KAStream{T}) where {T}
    s.last_kernel_time = gpu_elapsed() do
        copy_kernel!(s.backend, TBSIZE)(s.d_c, s.d_a; ndrange=s.array_size)
    end
    return nothing
end

function ka_mul!(s::KAStream{T}) where {T}
    scalar = T(startScalar)
    s.last_kernel_time = gpu_elapsed() do
        mul_kernel!(s.backend, TBSIZE)(s.d_b, s.d_c, scalar; ndrange=s.array_size)
    end
    return nothing
end

function ka_add!(s::KAStream{T}) where {T}
    s.last_kernel_time = gpu_elapsed() do
        add_kernel!(s.backend, TBSIZE)(s.d_c, s.d_a, s.d_b; ndrange=s.array_size)
    end
    return nothing
end

function ka_triad!(s::KAStream{T}) where {T}
    scalar = T(startScalar)
    s.last_kernel_time = gpu_elapsed() do
        triad_kernel!(s.backend, TBSIZE)(s.d_a, s.d_b, s.d_c, scalar; ndrange=s.array_size)
    end
    return nothing
end

function ka_dot!(s::KAStream{T}) where {T}
    s.last_kernel_time = gpu_elapsed() do
        dot_kernel!(s.backend, TBSIZE)(s.d_a, s.d_b, s.d_sum, s.array_size, s.dot_num_blocks;
            ndrange=TBSIZE * s.dot_num_blocks)
        copyto!(s.sums, s.d_sum)
    end

    total = zero(T)
    for i in 1:s.dot_num_blocks
        total += s.sums[i]
    end
    return total
end

get_time_taken(s::KAStream) = s.last_kernel_time

mutable struct Args
    array_size::Int
    num_times::Int
    device_index::Int
    mibibytes::Bool
    csv_output::Bool
end

function parse_arguments(argv::Vector{String})
    args = Args(33554432, 1000, 0, false, false)

    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--list"
            list_devices()
            exit(0)
        elseif a == "--device"
            i += 1
            i <= length(argv) || (println(stderr, "Invalid device index."); exit(1))
            args.device_index = parse(Int, argv[i])
        elseif a == "--arraysize" || a == "-s"
            i += 1
            i <= length(argv) || (println(stderr, "Invalid array size."); exit(1))
            v = tryparse(Int, argv[i])
            (v === nothing || v <= 0) && (println(stderr, "Invalid array size."); exit(1))
            args.array_size = v
        elseif a == "--numtimes" || a == "-n"
            i += 1
            i <= length(argv) || (println(stderr, "Invalid number of times."); exit(1))
            v = tryparse(Int, argv[i])
            v === nothing && (println(stderr, "Invalid number of times."); exit(1))
            if v < 2
                println(stderr, "Number of times must be 2 or more")
                exit(1)
            end
            args.num_times = v
        elseif a == "--float"
        elseif a == "--mibibytes"
            args.mibibytes = true
        elseif a == "--csv"
            args.csv_output = true
        elseif a == "--help" || a == "-h"
            print_help()
            exit(0)
        else
            println(stderr, "Unrecognized argument '$a' (try '--help')")
            exit(1)
        end
        i += 1
    end

    return args
end

function print_help()
    println()
    println("Usage: babelstream_ka.jl [OPTIONS]")
    println()
    println("Options:")
    println("  -h  --help               Print the message")
    println("      --list               List available devices")
    println("      --device     INDEX   Select device at INDEX")
    println("  -s  --arraysize  SIZE    Use SIZE elements in the array")
    println("  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)")
    println("      --float              Use floats (no-op, float is already default)")
    println("      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)")
    println()
end

function list_devices()
    names = gpu_device_names()
    if isempty(names)
        println(stderr, "No devices found.")
    else
        println()
        println("Devices:")
        for (i, n) in enumerate(names)
            println(i - 1, ": ", n)
        end
        println()
    end
end

function kahan_sum(v::Vector{T}) where {T}
    s = 0.0
    comp = 0.0
    for x in v
        y = Float64(x) - comp
        t = s + y
        comp = (t - s) - y
        s = t
    end
    return s
end

function metrics(v::Vector{T}, gold::T) where {T}
    gold_d = Float64(gold)
    actual_sum = kahan_sum(v)
    expected_sum = gold_d * length(v)

    checksum_rel_err = expected_sum != 0.0 ?
                        abs(actual_sum - expected_sum) / abs(expected_sum) : abs(actual_sum)

    max_rel_err = 0.0
    for x in v
        abs_err = abs(Float64(x) - gold_d)
        rel_err = gold_d != 0.0 ? abs_err / abs(gold_d) : abs_err
        max_rel_err = max(max_rel_err, rel_err)
    end

    return checksum_rel_err, max_rel_err
end

function check_solution(ntimes::Int, a::Vector{T}, b::Vector{T}, c::Vector{T},
    dot_sum::T, array_size::Int, csv_output::Bool) where {T}

    goldA = T(startA)
    goldB = T(startB)
    goldC = T(startC)
    scalar = T(startScalar)

    for _ in 1:ntimes
        goldC = goldA
        goldB = scalar * goldC
        goldC = goldA + goldB
        goldA = goldB + scalar * goldC
    end

    checksumA, maxErrA = metrics(a, goldA)
    checksumB, maxErrB = metrics(b, goldB)
    checksumC, maxErrC = metrics(c, goldC)

    expected_dot = Float64(goldA) * Float64(goldB) * Float64(array_size)
    actual_dot = Float64(dot_sum)

    product_magnitude = abs(Float64(goldA) * Float64(goldB))
    dot_underflows = product_magnitude < Float64(floatmin(T))

    dot_err = expected_dot != 0.0 ?
              abs(actual_dot - expected_dot) / abs(expected_dot) : abs(actual_dot)

    out = csv_output ? stderr : stdout
    precname = sizeof(T) == 4 ? "float32" : "float64"

    println(out)
    println(out, "Correctness metrics ($precname, $ntimes iterations)")
    @printf(out, "%-10s%-18s%-18s%-18s\n", "Array", "Expected", "Checksum rel err", "Max elem rel err")

    report = (name, gold, chk, mx) -> @printf(out, "%-10s%-18.6e%-18.3e%-18.3e\n", name, gold, chk, mx)
    report("a", Float64(goldA), checksumA, maxErrA)
    report("b", Float64(goldB), checksumB, maxErrB)
    report("c", Float64(goldC), checksumC, maxErrC)

    @printf(out, "%-10s%-18.6e%-18.3e%-18s\n", "dot", expected_dot, dot_err, "-")
    @printf(out, "  (actual dot   = %.9e)\n", actual_dot)

    if dot_underflows
        @printf(out, "  (note: expected element product %.3e underflows this precision at this iteration count, so the dot relative error above is not meaningful)\n",
            product_magnitude)
    end

    return nothing
end

function run_all!(s::KAStream{T}, num_times::Int) where {T}
    timings = [Float64[] for _ in 1:5]
    sum_result = zero(T)

    for _ in 1:num_times
        ka_copy!(s); push!(timings[1], get_time_taken(s))
        ka_mul!(s); push!(timings[2], get_time_taken(s))
        ka_add!(s); push!(timings[3], get_time_taken(s))
        ka_triad!(s); push!(timings[4], get_time_taken(s))
        sum_result = ka_dot!(s); push!(timings[5], get_time_taken(s))
    end

    return timings, sum_result
end

function run(::Type{T}, args::Args) where {T}
    if !args.csv_output
        println("Running kernels ", args.num_times, " times")
        println("Precision: ", T == Float32 ? "float" : "double")

        bytes_per_elem = sizeof(T)
        if args.mibibytes
            @printf("Array size: %.1f MiB (=%.1f GiB)\n",
                args.array_size * bytes_per_elem * 2.0^-20, args.array_size * bytes_per_elem * 2.0^-30)
            @printf("Total size: %.1f MiB (=%.1f GiB)\n",
                3.0 * args.array_size * bytes_per_elem * 2.0^-20, 3.0 * args.array_size * bytes_per_elem * 2.0^-30)
        else
            @printf("Array size: %.1f MB (=%.1f GB)\n",
                args.array_size * bytes_per_elem * 1.0E-6, args.array_size * bytes_per_elem * 1.0E-9)
            @printf("Total size: %.1f MB (=%.1f GB)\n",
                3.0 * args.array_size * bytes_per_elem * 1.0E-6, 3.0 * args.array_size * bytes_per_elem * 1.0E-9)
        end
    end

    stream = KAStream{T}(args.array_size, args.device_index)

    init_arrays!(stream, T(startA), T(startB), T(startC))
    init_elapsed_s = get_time_taken(stream)

    timings, sum_result = run_all!(stream, args.num_times)

    init_bw = ((args.mibibytes ? 2.0^-20 : 1.0E-9) * (3 * sizeof(T) * args.array_size)) / init_elapsed_s

    if !args.csv_output
        @printf("Init: %7g s (=%g %s)\n", init_elapsed_s, init_bw, args.mibibytes ? "MiBytes/sec" : "GBytes/sec")
        @printf("%-12s%-12s%-12s%-12s%-12s\n", "Function",
            args.mibibytes ? "MiBytes/sec" : "GBytes/sec", "Min (sec)", "Max", "Average")
    else
        println("backend,GPU,precision,vec_size,routine,BW_GBs")
    end

    labels = ["Copy", "Mul", "Add", "Triad", "Dot"]
    sizes = [
        2 * sizeof(T) * args.array_size,
        2 * sizeof(T) * args.array_size,
        3 * sizeof(T) * args.array_size,
        3 * sizeof(T) * args.array_size,
        2 * sizeof(T) * args.array_size,
    ]

    precname = sizeof(T) == 4 ? "float32" : "float64"
    gpu_name_str = gpu_name()

    for i in 1:5
        t = timings[i][2:end]
        tmin, tmax = minimum(t), maximum(t)
        average = sum(t) / length(t)

        if !args.csv_output
            bw = ((args.mibibytes ? 2.0^-20 : 1.0E-9) * sizes[i]) / tmin
            @printf("%-12s%-12.3g%-12.5g%-12.5g%-12.5g\n", labels[i], bw, tmin, tmax, average)
        else
            for tt in timings[i]
                bw = 1.0E-9 * sizes[i] / tt
                println("$(IMPLEMENTATION_STRING),$(gpu_name_str),$(precname),$(args.array_size),$(labels[i]),$bw")
            end
        end
    end

    a = Vector{T}(undef, args.array_size)
    b = Vector{T}(undef, args.array_size)
    c = Vector{T}(undef, args.array_size)
    read_arrays!(stream, a, b, c)

    check_solution(args.num_times, a, b, c, sum_result, args.array_size, args.csv_output)

    return nothing
end

function main()
    args = parse_arguments(ARGS)
    T = USE_FLOAT ? Float32 : Float64
    run(T, args)
end

main()
