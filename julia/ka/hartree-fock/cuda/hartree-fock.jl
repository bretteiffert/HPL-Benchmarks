#!/usr/bin/env julia
# Performs a KernelAbstractions two-electron Hartree-Fock Fock build using the backend selected by `STENCIL_GPU`. Arithmetic is configured by the shared `HF_PRECISION` and `LEGACY_SQRTF` source constants, which currently select fp64 and native square roots; the CLI accepts an input path plus check, CSV, and iteration options. The kernel uses 256-thread workgroups, atomic six-term updates, a startup GPU `erf` check, and vendor-event timing.

using KernelAbstractions
using Printf
using Atomix

include(joinpath(@__DIR__, "../gpu_shim.jl"))
include(joinpath(@__DIR__, "../../../gpu_math.jl"))
include(joinpath(@__DIR__, "../../../hf_host.jl"))

@kernel function hf_kernel!(fock, nnnn, ngauss, natoms,
                            @Const(geom), @Const(xpnt), @Const(coef),
                            @Const(dens), @Const(schwarz))
    idx = @index(Global, Linear)
    if 1 <= idx <= nnnn
        ijkl = UInt64(idx)
        nat = UInt64(natoms)
        U1 = UInt64(1)
        U2 = UInt64(2)

        @inbounds begin
            ij = idx_sqrt(U2 * ijkl)
            n = div(ij * ij + ij, U2)
            while n < ijkl
                ij += U1
                n = div(ij * ij + ij, U2)
            end
            kl = ijkl - div(ij * ij - ij, U2)

            if schwarz[ij + U1] * schwarz[kl + U1] > DTOL
                i = idx_sqrt(U2 * ij) - U1
                n = div(i * i + i, U2)
                while n < ij
                    i += U1
                    n = div(i * i + i, U2)
                end
                j = ij - div(i * i - i, U2)

                k = idx_sqrt(U2 * kl)
                n = div(k * k + k, U2)
                while n < kl
                    k += U1
                    n = div(k * k + k, U2)
                end
                l = kl - div(k * k - k, U2)
                i -= U1
                j -= U1
                k -= U1
                l -= U1

                eri = zero(RealT)
                one_T = one(RealT)
                p15 = RealT(1.5)
                pm05 = RealT(-0.5)
                for ib in 1:ngauss, jb in 1:ngauss
                    aij = one_T / (xpnt[ib] + xpnt[jb])
                    dij = coef[ib] * coef[jb] *
                          exp(-xpnt[ib] * xpnt[jb] * aij *
                              (sq(geom[i + U1] - geom[j + U1]) +
                               sq(geom[nat + i + U1] - geom[nat + j + U1]) +
                               sq(geom[U2 * nat + i + U1] - geom[U2 * nat + j + U1]))) * aij^p15
                    if abs(dij) > DTOL
                        xij = aij * (xpnt[ib] * geom[i + U1] + xpnt[jb] * geom[j + U1])
                        yij = aij * (xpnt[ib] * geom[nat + i + U1] + xpnt[jb] * geom[nat + j + U1])
                        zij = aij * (xpnt[ib] * geom[U2 * nat + i + U1] + xpnt[jb] * geom[U2 * nat + j + U1])
                        for kb in 1:ngauss, lb in 1:ngauss
                            akl = one_T / (xpnt[kb] + xpnt[lb])
                            dkl = dij * coef[kb] * coef[lb] *
                                  exp(-xpnt[kb] * xpnt[lb] * akl *
                                      (sq(geom[k + U1] - geom[l + U1]) +
                                       sq(geom[nat + k + U1] - geom[nat + l + U1]) +
                                       sq(geom[U2 * nat + k + U1] - geom[U2 * nat + l + U1]))) * akl^p15
                            if abs(dkl) > DTOL
                                aijkl = (xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb]) /
                                        (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                                tt = aijkl * (sq(xij - akl * (xpnt[kb] * geom[k + U1] + xpnt[lb] * geom[l + U1])) +
                                              sq(yij - akl * (xpnt[kb] * geom[nat + k + U1] + xpnt[lb] * geom[nat + l + U1])) +
                                              sq(zij - akl * (xpnt[kb] * geom[U2 * nat + k + U1] + xpnt[lb] * geom[U2 * nat + l + U1])))
                                f0t = SQRTPI2_DEV
                                if tt > RCUT
                                    f0t = tt^pm05 * gpu_erf(sqrt_boys(tt))
                                end
                                eri += dkl * f0t * sqrt_boys(aijkl)
                            end
                        end
                    end
                end

                i == j && (eri *= RealT(0.5))
                k == l && (eri *= RealT(0.5))
                (i == k && j == l) && (eri *= RealT(0.5))

                four = RealT(4.0)
                Atomix.@atomic fock[i * nat + j + U1] += dens[k * nat + l + U1] * eri * four
                Atomix.@atomic fock[k * nat + l + U1] += dens[i * nat + j + U1] * eri * four
                Atomix.@atomic fock[i * nat + k + U1] += -dens[j * nat + l + U1] * eri
                Atomix.@atomic fock[i * nat + l + U1] += -dens[j * nat + k + U1] * eri
                Atomix.@atomic fock[j * nat + k + U1] += -dens[i * nat + l + U1] * eri
                Atomix.@atomic fock[j * nat + l + U1] += -dens[i * nat + k + U1] * eri
            end
        end
    end
end

function to_device(backend, h::Vector{T}) where {T}
    d = KernelAbstractions.allocate(backend, T, length(h))
    copyto!(d, h)
    return d
end

function main()
    if length(ARGS) < 1
        println("ERROR: Please provide an input file")
        return 1
    end
    o = parse_args(vcat("prog", ARGS))

    backend = gpu_backend()
    S = host_setup(ARGS[1])

    d_xpnt = to_device(backend, S.h_xpnt)
    d_coef = to_device(backend, S.h_coef)
    d_geom = to_device(backend, S.h_geom)
    d_dens = to_device(backend, S.h_dens)
    d_fock = to_device(backend, S.h_fock)
    d_schwarz = to_device(backend, S.h_schwarz)
    KernelAbstractions.synchronize(backend)

    nnnn = S.nnnn
    kern = hf_kernel!(backend, BLK_SIZE)
    launch() = kern(d_fock, nnnn, S.ngauss, S.natoms,
                    d_geom, d_xpnt, d_coef, d_dens, d_schwarz; ndrange = Int(nnnn))

    launch()
    gpu_sync()

    h_fock = Array(d_fock)
    erep = contract_erep(h_fock, S.h_dens, S.natoms)

    if !o.csv_output
        report_check(S, h_fock, erep, o)
    else
        println("backend,GPU,precision,sqrt_mode,natoms,ngauss,exec_time_ms")
        for _ in 1:o.num_iter
            gpu_sync()
            elapsed_ms = gpu_elapsed(launch) * 1000.0
            @printf("KernelAbstractions,%s,%s,%s,%d,%d,%f\n",
                    gpu_name(), REAL_NAME, SQRT_MODE, S.natoms, S.ngauss, elapsed_ms)
        end
    end
    return 0
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(main())
end
