#!/usr/bin/env julia

using KernelAbstractions
using Printf

include(joinpath(@__DIR__, "backend.jl"))

const DEFAULT_SIZE = 1024
const DEFAULT_ITERATIONS = 100
const DEFAULT_WORKGROUP = (1024, 1, 1)

@kernel function test_function_kernel!(u, hx, hy, hz)
    i, j, k = @index(Global, NTuple)

    T = eltype(u)
    nx, ny, nz = size(u)

    c = T(0.5)
    x = T(i - 1) * hx
    y = T(j - 1) * hy
    z = T(k - 1) * hz
    Lx = T(nx) * hx
    Ly = T(ny) * hy
    Lz = T(nz) * hz

    @inbounds u[i, j, k] = c * x * (x - Lx) +
                            c * y * (y - Ly) +
                            c * z * (z - Lz)
end

@kernel function laplacian_kernel!(f, @Const(u), invhx2, invhy2, invhz2, invhxyz2)
    i, j, k = @index(Global, NTuple)

    nx, ny, nz = size(u)

    if !(i == 1 || i >= nx || j == 1 || j >= ny || k == 1 || k >= nz)
        @inbounds f[i, j, k] = u[i, j, k] * invhxyz2 +
                               (u[i - 1, j, k] + u[i + 1, j, k]) * invhx2 +
                               (u[i, j - 1, k] + u[i, j + 1, k]) * invhy2 +
                               (u[i, j, k - 1] + u[i, j, k + 1]) * invhz2
    end
end

function initialize_field!(backend, d_u, nx, ny, nz, hx::T, hy::T, hz::T) where {T}
    kernel = test_function_kernel!(backend, DEFAULT_WORKGROUP)
    kernel(d_u, hx, hy, hz; ndrange=(nx, ny, nz))
    return nothing
end

function run_laplacian!(backend, d_f, d_u, nx, ny, nz,
                        hx::T, hy::T, hz::T) where {T}
    invhx2 = one(T) / hx / hx
    invhy2 = one(T) / hy / hy
    invhz2 = one(T) / hz / hz
    invhxyz2 = T(-2) * (invhx2 + invhy2 + invhz2)

    kernel = laplacian_kernel!(backend, DEFAULT_WORKGROUP)
    kernel(d_f, d_u, invhx2, invhy2, invhz2, invhxyz2;
           ndrange=(nx, ny, nz))
    return nothing
end

struct CorrectnessResult
    max_abs_err::Float64
    l2_rel_err::Float64
    mean_abs_err::Float64
    num_checked::Int
end

function check_correctness(d_f, nx, ny, nz)
    h_f = Array(d_f)

    expected = 3.0
    sum_abs = 0.0
    sum_sq_err = 0.0
    sum_sq_exact = 0.0
    max_err = 0.0
    count = 0

    @inbounds for k in 2:(nz - 1), j in 2:(ny - 1), i in 2:(nx - 1)
        value = Float64(h_f[i, j, k])
        err = abs(value - expected)
        sum_abs += err
        sum_sq_err += err * err
        sum_sq_exact += expected * expected
        max_err = max(max_err, err)
        count += 1
    end

    if count == 0
        return CorrectnessResult(0.0, 0.0, 0.0, 0)
    end

    mean_abs_err = sum_abs / count
    l2_rel_err = sqrt(sum_sq_err / sum_sq_exact)
    return CorrectnessResult(max_err, l2_rel_err, mean_abs_err, count)
end

function precision_type(bits::Int)
    bits == 16 && return Float16
    bits == 32 && return Float32
    bits == 64 && return Float64
    error("Precision must be 16, 32, or 64 bits; got $bits")
end

function parse_arguments(argv)
    size = DEFAULT_SIZE
    iterations = DEFAULT_ITERATIONS
    precision_bits = 32
    csv_output = false
    show_help = false

    i = 1
    while i <= length(argv)
        arg = argv[i]

        if arg in ("-h", "--help")
            show_help = true
        elseif arg == "--csv"
            csv_output = true
        elseif arg in ("-s", "--size")
            i += 1
            i <= length(argv) || error("$arg requires an integer size")
            size = parse(Int, argv[i])
        elseif startswith(arg, "-s=") || startswith(arg, "--size=")
            size = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg in ("-n", "--iterations")
            i += 1
            i <= length(argv) || error("$arg requires an integer iteration count")
            iterations = parse(Int, argv[i])
        elseif startswith(arg, "-n=") || startswith(arg, "--iterations=")
            iterations = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg in ("-p", "--precision")
            i += 1
            i <= length(argv) || error("$arg requires 16, 32, or 64")
            precision_bits = parse(Int, argv[i])
        elseif startswith(arg, "-p=") || startswith(arg, "--precision=")
            precision_bits = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--backend" || startswith(arg, "--backend=")
            # backend.jl consumes this option; skip its separate value here.
            if arg == "--backend"
                i += 1
                i <= length(argv) || error("--backend requires cuda or hip")
            end
        else
            error("Unknown argument '$arg'. Use --help for usage.")
        end

        i += 1
    end

    size >= 3 || error("Size must be at least 3")
    iterations > 0 || error("Iteration count must be positive")
    precision = precision_type(precision_bits)
    return (; size, iterations, precision_bits, precision, csv_output, show_help)
end

function print_help()
    println("Seven-point Laplacian benchmark with KernelAbstractions")
    println()
    println("Usage:")
    println("  julia --project=env/cuda seven-point-stencil.jl --backend cuda [options]")
    println()
    println("Options:")
    println("  --backend cuda|hip   Select the GPU backend (default: cuda)")
    println("  -s SIZE               Cubic grid side length (default: $DEFAULT_SIZE)")
    println("  -n NUM                Timed kernel iterations (default: $DEFAULT_ITERATIONS)")
    println("  -p 16|32|64           Floating-point type (default: 32)")
    println("  --csv                 Write performance rows as CSV")
    println("  -h, --help            Show this help")
end

function main()
    options = parse_arguments(ARGS)
    if options.show_help
        print_help()
        return 0
    end

    T = options.precision
    nx = ny = nz = options.size
    workgroup = DEFAULT_WORKGROUP
    itemsize = sizeof(T)

    interior = (nx - 2) * (ny - 2) * (nz - 2)
    fetch_points = nx * ny * nz - 8 -
                   4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)
    theoretical_fetch_size = fetch_points * itemsize
    theoretical_write_size = interior * itemsize
    datasize = theoretical_fetch_size + theoretical_write_size

    backend = gpu_backend()
    device_name = gpu_name()

    if options.csv_output
        println("backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs")
    else
        println(device_name)
        println("Precision: Float$(options.precision_bits)")
        println("nx,ny,nz = $nx, $ny, $nz")
        println("block sizes = $(workgroup[1]), $(workgroup[2]), $(workgroup[3])")
        @printf("Theoretical fetch size (GB): %g\n", theoretical_fetch_size * 1e-9)
        @printf("Theoretical write size (GB): %g\n", theoretical_write_size * 1e-9)
    end

    d_u = KernelAbstractions.allocate(backend, T, nx, ny, nz)
    d_f = KernelAbstractions.allocate(backend, T, nx, ny, nz)

    hx = T(1) / T(nx - 1)
    hy = T(1) / T(ny - 1)
    hz = T(1) / T(nz - 1)

    initialize_field!(backend, d_u, nx, ny, nz, hx, hy, hz)
    run_laplacian!(backend, d_f, d_u, nx, ny, nz, hx, hy, hz)
    gpu_sync()

    correctness = check_correctness(d_f, nx, ny, nz)
    if options.csv_output
        println(stderr, "correctness,precision,max_abs_err,l2_rel_err,mean_abs_err,points_checked")
        @printf(stderr, "correctness,Float%d,%g,%g,%g,%d\n", options.precision_bits,
                correctness.max_abs_err, correctness.l2_rel_err,
                correctness.mean_abs_err, correctness.num_checked)
    else
        println("--- Correctness metrics (precision = Float$(options.precision_bits)) ---")
        println("Interior points checked : ", correctness.num_checked)
        @printf("Max abs error           : %g\n", correctness.max_abs_err)
        @printf("Mean abs error          : %g\n", correctness.mean_abs_err)
        @printf("Relative L2 error       : %g\n\n", correctness.l2_rel_err)
    end

    total_elapsed = 0.0
    backend_label = gpu_label()
    precision_label = "Float$(options.precision_bits)"

    for _ in 1:options.iterations
        gpu_sync()
        elapsed = gpu_elapsed(() -> run_laplacian!(backend, d_f, d_u, nx, ny, nz,
                                                   hx, hy, hz)) * 1.0e3
        if options.csv_output
            @printf("KernelAbstractions-%s,%s,%s,%d,%d,%d,%d,%g\n",
                    backend_label, device_name, precision_label, nx,
                    workgroup[1], workgroup[2], workgroup[3],
                    datasize / elapsed / 1e6)
        end
        total_elapsed += elapsed
    end

    if !options.csv_output
        @printf("Laplacian kernel took: %g ms, effective memory bandwidth: %g GB/s\n\n",
                total_elapsed / options.iterations,
                datasize * options.iterations / total_elapsed / 1e6)
    end

    return 0
end

main()
