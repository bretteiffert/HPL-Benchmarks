#!/usr/bin/env julia
# Runs a KernelAbstractions seven-point Laplacian benchmark on a backend selected by `STENCIL_GPU`. This HIP launcher defaults to Float64, accepts optional grid and workgroup dimensions plus `--csv`, validates the quadratic test field against an interior value of 3, and reports 100 event-timed iterations.

using KernelAbstractions
include(joinpath(@__DIR__, "../gpu_shim.jl"))
using Printf

const precision = Float64
const precision_str = "double"

const TBSize = 1024
const L = 1024
const NUM_ITER = 100

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
    @inbounds u[i, j, k] = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz)
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

function test_function(backend, d_u, nx, ny, nz, hx::T, hy::T, hz::T) where {T}
    kern = test_function_kernel!(backend, (TBSize, 1, 1))
    kern(d_u, hx, hy, hz; ndrange = (nx, ny, nz))
    return nothing
end

function laplacian(backend, d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z,
                   hx::T, hy::T, hz::T) where {T}
    invhx2 = one(T) / hx / hx
    invhy2 = one(T) / hy / hy
    invhz2 = one(T) / hz / hz
    invhxyz2 = T(-2) * (invhx2 + invhy2 + invhz2)

    kern = laplacian_kernel!(backend, (BLK_X, BLK_Y, BLK_Z))
    kern(d_f, d_u, invhx2, invhy2, invhz2, invhxyz2; ndrange = (nx, ny, nz))
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
        val = Float64(h_f[i, j, k])
        err = abs(val - expected)
        sum_abs += err
        sum_sq_err += err * err
        sum_sq_exact += expected * expected
        if err > max_err
            max_err = err
        end
        count += 1
    end

    if count > 0
        mean_abs_err = sum_abs / count
        l2_rel_err = sum_sq_exact > 0.0 ? sqrt(sum_sq_err / sum_sq_exact) : sqrt(sum_sq_err)
        return CorrectnessResult(max_err, l2_rel_err, mean_abs_err, count)
    end
    return CorrectnessResult(0.0, 0.0, 0.0, 0)
end

function main()
    BLK_X = TBSize
    BLK_Y = 1
    BLK_Z = 1

    nx = L; ny = L; nz = L
    csv_output = false

    length(ARGS) > 0 && (nx = parse(Int, ARGS[1]))
    length(ARGS) > 1 && (ny = parse(Int, ARGS[2]))
    length(ARGS) > 2 && (nz = parse(Int, ARGS[3]))
    length(ARGS) > 3 && (BLK_X = parse(Int, ARGS[4]))
    length(ARGS) > 4 && (BLK_Y = parse(Int, ARGS[5]))
    length(ARGS) > 5 && (BLK_Z = parse(Int, ARGS[6]))
    length(ARGS) > 6 && ARGS[7] == "--csv" && (csv_output = true)

    itemsize = sizeof(precision)

    theoretical_fetch_size = (nx * ny * nz - 8 -
                              4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)) * itemsize
    theoretical_write_size = ((nx - 2) * (ny - 2) * (nz - 2)) * itemsize
    datasize = theoretical_fetch_size + theoretical_write_size

    backend = gpu_backend()
    dev_name = gpu_name()

    if csv_output
        println("backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs")
    else
        println(dev_name)
        println("Precision: ", precision_str)
        println("nx,ny,nz = $nx, $ny, $nz")
        println("block sizes = $BLK_X, $BLK_Y, $BLK_Z")
        @printf("Theoretical fetch size (GB): %g\n", theoretical_fetch_size * 1e-9)
        @printf("Theoretical write size (GB): %g\n", theoretical_write_size * 1e-9)
    end

    d_u = KernelAbstractions.allocate(backend, precision, nx, ny, nz)
    d_f = KernelAbstractions.allocate(backend, precision, nx, ny, nz)

    hx = precision(1 / (nx - 1))
    hy = precision(1 / (ny - 1))
    hz = precision(1 / (nz - 1))

    test_function(backend, d_u, nx, ny, nz, hx, hy, hz)

    laplacian(backend, d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz)
    KernelAbstractions.synchronize(backend)

    corr = check_correctness(d_f, nx, ny, nz)

    if csv_output
        println(stderr, "correctness,precision,max_abs_err,l2_rel_err,mean_abs_err,points_checked")
        @printf(stderr, "correctness,%s,%g,%g,%g,%d\n", precision_str,
                corr.max_abs_err, corr.l2_rel_err, corr.mean_abs_err, corr.num_checked)
    else
        println("--- Correctness metrics (precision = $precision_str) ---")
        println("Interior points checked : ", corr.num_checked)
        @printf("Max abs error           : %g\n", corr.max_abs_err)
        @printf("Mean abs error          : %g\n", corr.mean_abs_err)
        @printf("Relative L2 error       : %g\n\n", corr.l2_rel_err)
    end

    total_elapsed = 0.0
    backend_label = gpu_label()

    for _ in 1:NUM_ITER
        KernelAbstractions.synchronize(backend)
        elapsed = gpu_elapsed(() -> laplacian(backend, d_f, d_u, nx, ny, nz,
                                              BLK_X, BLK_Y, BLK_Z, hx, hy, hz)) * 1.0e3
        if csv_output
            @printf("KernelAbstractions-%s,%s,%s,%d,%d,%d,%d,%g\n", backend_label, dev_name,
                    precision_str, nx, BLK_X, BLK_Y, BLK_Z, datasize / elapsed / 1e6)
        end
        total_elapsed += elapsed
    end

    if !csv_output
        @printf("Laplacian kernel took: %g ms, effective memory bandwidth: %g GB/s \n\n",
                total_elapsed / NUM_ITER,
                datasize * NUM_ITER / total_elapsed / 1e6)
    end

    return 0
end

main()
