#!/usr/bin/env julia
# Hartree–Fock Fock-matrix build through KernelAbstractions.
# The only separate source file is backend.jl.

using KernelAbstractions
using Printf
using Atomix

include(joinpath(@__DIR__, "backend.jl"))

# Implements the Hartree-Fock host pipeline: input parsing, Schwarz screening data,
# an fp64 CPU reference, argument handling, and correctness reporting.
function _precision_from_args(args)
    bits = 64
    i = 1
    while i <= length(args)
        if args[i] == "-p" || args[i] == "--precision"
            i += 1
            i <= length(args) || error("-p requires 16, 32, or 64")
            bits = parse(Int, args[i])
        elseif startswith(args[i], "--precision=")
            bits = parse(Int, split(args[i], "=", limit=2)[2])
        end
        i += 1
    end
    bits in (16, 32, 64) || error("Precision must be 16, 32, or 64")
    return bits
end

const HF_PRECISION = _precision_from_args(ARGS)
const RealT = HF_PRECISION == 16 ? Float16 :
              HF_PRECISION == 32 ? Float32 : Float64
const REAL_NAME = "fp$(HF_PRECISION)"

const LEGACY_SQRTF = false
const SQRT_MODE = LEGACY_SQRTF ? "legacy-sqrtf" : "native"

const SQRTPI2_HOST = RealT(1.12837916709551255856)
const SQRTPI2_DEV = RealT(1.12837916709551255856)
const DTOL = HF_PRECISION == 16 ? RealT(1.0e-5) : RealT(1.0e-12)
const RCUT = HF_PRECISION == 16 ? RealT(1.0e-5) : RealT(1.0e-12)
const TOBOHRS = RealT(1.889725987722)

const BLK_SIZE = 256

const REF_SQRTPI2 = 1.12837916709551255856
const REF_DTOL = 1.0e-12
const REF_RCUT = 1.0e-12

@inline idx_sqrt(x::UInt64) = unsafe_trunc(UInt64, sqrt(Float32(x)))

@inline function sqrt_boys(x::T) where {T}
    LEGACY_SQRTF ? T(sqrt(Float32(x))) : sqrt(x)
end

function read_file(filename::AbstractString)
    tok = split(read(filename, String))
    p = 1
    ngauss = parse(Int, tok[p]); p += 1
    natoms = parse(Int, tok[p]); p += 1
    xpnt_d = Vector{Float64}(undef, ngauss)
    coef_d = Vector{Float64}(undef, ngauss)
    for i in 1:ngauss
        xpnt_d[i] = parse(Float64, tok[p]); p += 1
        coef_d[i] = parse(Float64, tok[p]); p += 1
    end
    geom_d = Vector{Float64}(undef, 3 * natoms)
    for i in 0:(natoms - 1)
        for j in 0:2
            geom_d[j * natoms + i + 1] = parse(Float64, tok[p]); p += 1
        end
    end
    return (ngauss, natoms,
            RealT.(xpnt_d), RealT.(coef_d), RealT.(geom_d),
            xpnt_d, coef_d, geom_d)
end

@inline sq(x) = x * x

function ssss(i::Int, j::Int, k::Int, l::Int, ngauss::Int,
              xpnt::Vector{T}, coef::Vector{T}, geom::Vector{T},
              natoms::Int) where {T}
    eri = zero(T)
    one_T = one(T)
    p15 = T(1.5)
    pm05 = T(-0.5)
    @inbounds for ib in 1:ngauss, jb in 1:ngauss
        aij = one_T / (xpnt[ib] + xpnt[jb])
        dij = coef[ib] * coef[jb] *
              exp(-xpnt[ib] * xpnt[jb] * aij *
                  (sq(geom[i + 1] - geom[j + 1]) +
                   sq(geom[natoms + i + 1] - geom[natoms + j + 1]) +
                   sq(geom[2natoms + i + 1] - geom[2natoms + j + 1]))) * aij^p15
        if abs(dij) > DTOL
            xij = aij * (xpnt[ib] * geom[i + 1] + xpnt[jb] * geom[j + 1])
            yij = aij * (xpnt[ib] * geom[natoms + i + 1] + xpnt[jb] * geom[natoms + j + 1])
            zij = aij * (xpnt[ib] * geom[2natoms + i + 1] + xpnt[jb] * geom[2natoms + j + 1])
            for kb in 1:ngauss, lb in 1:ngauss
                akl = one_T / (xpnt[kb] + xpnt[lb])
                dkl = dij * coef[kb] * coef[lb] *
                      exp(-xpnt[kb] * xpnt[lb] * akl *
                          (sq(geom[k + 1] - geom[l + 1]) +
                           sq(geom[natoms + k + 1] - geom[natoms + l + 1]) +
                           sq(geom[2natoms + k + 1] - geom[2natoms + l + 1]))) * akl^p15
                if abs(dkl) > DTOL
                    aijkl = (xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb]) /
                            (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                    tt = aijkl * (sq(xij - akl * (xpnt[kb] * geom[k + 1] + xpnt[lb] * geom[l + 1])) +
                                  sq(yij - akl * (xpnt[kb] * geom[natoms + k + 1] + xpnt[lb] * geom[natoms + l + 1])) +
                                  sq(zij - akl * (xpnt[kb] * geom[2natoms + k + 1] + xpnt[lb] * geom[2natoms + l + 1])))
                    f0t = SQRTPI2_HOST
                    if tt > RCUT
                        f0t = tt^pm05 * host_erf(sqrt_boys(tt))
                    end
                    eri += dkl * f0t * sqrt_boys(aijkl)
                end
            end
        end
    end
    return eri
end

function build_schwarz(ngauss::Int, natoms::Int,
                       xpnt::Vector{T}, coef::Vector{T}, geom::Vector{T}) where {T}
    nn = div(natoms * natoms + natoms, 2)
    schwarz = zeros(T, nn + 1)
    ij = 0
    for i in 0:(natoms - 1), j in 0:i
        ij += 1
        schwarz[ij + 1] = sqrt(abs(ssss(i, j, i, j, ngauss, xpnt, coef, geom, natoms)))
    end
    return schwarz, nn
end

function ssss_ref(i::Int, j::Int, k::Int, l::Int, ngauss::Int,
                  xpnt::Vector{Float64}, coef::Vector{Float64},
                  geom::Vector{Float64}, natoms::Int)
    eri = 0.0
    @inbounds for ib in 1:ngauss, jb in 1:ngauss
        aij = 1.0 / (xpnt[ib] + xpnt[jb])
        dij = coef[ib] * coef[jb] *
              exp(-xpnt[ib] * xpnt[jb] * aij *
                  (sq(geom[i + 1] - geom[j + 1]) +
                   sq(geom[natoms + i + 1] - geom[natoms + j + 1]) +
                   sq(geom[2natoms + i + 1] - geom[2natoms + j + 1]))) * aij^1.5
        if abs(dij) > REF_DTOL
            xij = aij * (xpnt[ib] * geom[i + 1] + xpnt[jb] * geom[j + 1])
            yij = aij * (xpnt[ib] * geom[natoms + i + 1] + xpnt[jb] * geom[natoms + j + 1])
            zij = aij * (xpnt[ib] * geom[2natoms + i + 1] + xpnt[jb] * geom[2natoms + j + 1])
            for kb in 1:ngauss, lb in 1:ngauss
                akl = 1.0 / (xpnt[kb] + xpnt[lb])
                dkl = dij * coef[kb] * coef[lb] *
                      exp(-xpnt[kb] * xpnt[lb] * akl *
                          (sq(geom[k + 1] - geom[l + 1]) +
                           sq(geom[natoms + k + 1] - geom[natoms + l + 1]) +
                           sq(geom[2natoms + k + 1] - geom[2natoms + l + 1]))) * akl^1.5
                if abs(dkl) > REF_DTOL
                    aijkl = (xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb]) /
                            (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                    tt = aijkl * (sq(xij - akl * (xpnt[kb] * geom[k + 1] + xpnt[lb] * geom[l + 1])) +
                                  sq(yij - akl * (xpnt[kb] * geom[natoms + k + 1] + xpnt[lb] * geom[natoms + l + 1])) +
                                  sq(zij - akl * (xpnt[kb] * geom[2natoms + k + 1] + xpnt[lb] * geom[2natoms + l + 1])))
                    f0t = REF_SQRTPI2
                    if tt > REF_RCUT
                        f0t = tt^-0.5 * host_erf(sqrt(tt))
                    end
                    eri += dkl * f0t * sqrt(aijkl)
                end
            end
        end
    end
    return eri
end

function fock_build_cpu(ngauss::Int, natoms::Int,
                        geom::Vector{Float64}, xpnt::Vector{Float64},
                        coef::Vector{Float64}, dens::Vector{Float64})
    fock = zeros(Float64, natoms * natoms)
    nn = div(natoms * natoms + natoms, 2)
    schwarz = zeros(Float64, nn + 1)
    ijr = 0
    for i in 0:(natoms - 1), j in 0:i
        ijr += 1
        schwarz[ijr + 1] = sqrt(abs(ssss_ref(i, j, i, j, ngauss, xpnt, coef, geom, natoms)))
    end

    @inbounds for i in 0:(natoms - 1), j in 0:i
        ij = div(i * (i + 1), 2) + j + 1
        for k in 0:(natoms - 1), l in 0:k
            kl = div(k * (k + 1), 2) + l + 1
            kl > ij && continue
            (schwarz[ij + 1] * schwarz[kl + 1] > REF_DTOL) || continue

            eri = ssss_ref(i, j, k, l, ngauss, xpnt, coef, geom, natoms)
            i == j && (eri *= 0.5)
            k == l && (eri *= 0.5)
            (i == k && j == l) && (eri *= 0.5)

            fock[i * natoms + j + 1] += dens[k * natoms + l + 1] * eri * 4.0
            fock[k * natoms + l + 1] += dens[i * natoms + j + 1] * eri * 4.0
            fock[i * natoms + k + 1] -= dens[j * natoms + l + 1] * eri
            fock[i * natoms + l + 1] -= dens[j * natoms + k + 1] * eri
            fock[j * natoms + k + 1] -= dens[i * natoms + l + 1] * eri
            fock[j * natoms + l + 1] -= dens[i * natoms + k + 1] * eri
        end
    end
    return fock
end

function energy_from_fock(fock::Vector{Float64}, dens::Vector{Float64}, natoms::Int)
    e = 0.0
    @inbounds for t in 1:(natoms * natoms)
        e += fock[t] * dens[t]
    end
    return e * 0.5
end

function energy_symmetrized(fock::Vector{Float64}, dens::Vector{Float64}, natoms::Int)
    e = 0.0
    @inbounds for i in 0:(natoms - 1), j in 0:(natoms - 1)
        f = i == j ? fock[i * natoms + j + 1] :
            fock[i * natoms + j + 1] + fock[j * natoms + i + 1]
        e += f * dens[i * natoms + j + 1]
    end
    return e * 0.5
end

mutable struct Opts
    csv_output::Bool
    run_check::Bool
    force_check::Bool
    max_check_atoms::Int
    num_iter::Int
end

function print_help()
    println("Usage: hartree-fock.jl INPUT [OPTIONS]")
    println("  --backend cuda|hip")
    println("  -p 16|32|64")
    println("  -n NUM")
    println("  --csv  --check  --no-check")
end

function parse_args(args)
    o = Opts(false, true, false, 128, 10)
    i = 3  # args[2] is the positional input file
    while i <= length(args)
        s = args[i]
        if s == "--csv"
            o.csv_output = true
        elseif s == "--check"
            o.run_check = true; o.force_check = true
        elseif s == "--no-check"
            o.run_check = false
        elseif s == "-h" || s == "--help"
            print_help()
            exit(0)
        elseif s == "--backend" || s == "-p" || s == "--precision"
            i += 1
        elseif s == "-n"
            i += 1
            i <= length(args) || error("-n requires a positive integer")
            o.num_iter = parse(Int, args[i])
            o.num_iter > 0 || error("-n must be positive")
        elseif startswith(s, "--max-check-atoms=")
            o.max_check_atoms = parse(Int, s[19:end])
        elseif startswith(s, "-")
            error("Unrecognized argument '$s'")
        end
        i += 1
    end
    return o
end

function host_setup(filename::AbstractString)
    ngauss, natoms, h_xpnt, h_coef, h_geom, xpnt_d, coef_d, geom_d = read_file(filename)
    nsq = natoms * natoms

    h_dens = fill(RealT(0.1), nsq)
    dens_d = fill(0.1, nsq)
    for i in 0:(natoms - 1)
        h_dens[i * natoms + i + 1] = RealT(1.0)
        dens_d[i * natoms + i + 1] = 1.0
    end

    for i in 1:ngauss
        h_coef[i] = h_coef[i] * (RealT(2.0) * h_xpnt[i])^RealT(0.75)
        coef_d[i] = coef_d[i] * (2.0 * xpnt_d[i])^0.75
    end
    for i in 0:(natoms - 1)
        h_geom[0natoms + i + 1] *= TOBOHRS
        h_geom[1natoms + i + 1] *= TOBOHRS
        h_geom[2natoms + i + 1] *= TOBOHRS
        geom_d[0natoms + i + 1] *= 1.889725987722
        geom_d[1natoms + i + 1] *= 1.889725987722
        geom_d[2natoms + i + 1] *= 1.889725987722
    end

    h_fock = zeros(RealT, nsq)
    h_schwarz, nn = build_schwarz(ngauss, natoms, h_xpnt, h_coef, h_geom)
    nnnn = div(nn * nn + nn, 2)

    return (; ngauss, natoms, nsq, nn, nnnn,
            h_xpnt, h_coef, h_geom, h_dens, h_fock, h_schwarz,
            xpnt_d, coef_d, geom_d, dens_d)
end

function contract_erep(h_fock::Vector{T}, h_dens::Vector{T}, natoms::Int) where {T}
    erep = zero(T)
    @inbounds for i in 0:(natoms - 1), j in 0:(natoms - 1)
        erep += h_fock[i * natoms + j + 1] * h_dens[i * natoms + j + 1]
    end
    return erep
end

function report_check(S, h_fock::Vector{RealT}, erep::RealT, o::Opts)
    natoms = S.natoms
    @printf("2e- energy=%.16f\n", Float64(erep) * 0.5)
    if !(o.run_check && (o.force_check || natoms <= o.max_check_atoms))
        o.run_check && @printf("(correctness check skipped: natoms=%d > %d; use --check to force)\n",
                               natoms, o.max_check_atoms)
        return
    end

    fock_gpu = Float64.(h_fock)
    fock_ref = fock_build_cpu(S.ngauss, natoms, S.geom_d, S.xpnt_d, S.coef_d, S.dens_d)

    e_gpu = energy_from_fock(fock_gpu, S.dens_d, natoms)
    e_ref = energy_from_fock(fock_ref, S.dens_d, natoms)
    denom = abs(e_ref) > 0.0 ? abs(e_ref) : 1.0
    rel_loss = abs(e_gpu - e_ref) / denom
    e_inpipe = Float64(erep) * 0.5
    rel_inpipe = abs(e_inpipe - e_ref) / denom

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
    for i in 0:(natoms - 1), j in 0:(natoms - 1)
        d = abs(fock_gpu[i * natoms + j + 1] - fock_gpu[j * natoms + i + 1])
        d > asym && (asym = d)
    end

    println()
    @printf("--- precision check (build: %s, sqrt: %s) ---\n", REAL_NAME, SQRT_MODE)
    @printf("2e- energy (%s, %s reduction)      = %.16f\n", REAL_NAME, REAL_NAME, e_inpipe)
    @printf("2e- energy (%s Fock, fp64 reduction) = %.16f\n", REAL_NAME, e_gpu)
    @printf("2e- energy (fp64 CPU reference)       = %.16f\n", e_ref)
    @printf("precision loss, Fock build only       = %.6e\n", rel_loss)
    @printf("precision loss, incl. %s reduction  = %.6e\n", REAL_NAME, rel_inpipe)
    @printf("max |F - F_ref| (elementwise)         = %.6e  (rel %.6e)\n", maxd, maxd / fscale)
    @printf("rms |F - F_ref| (elementwise)         = %.6e\n", rmsd)
    @printf("max |F(i,j) - F(j,i)|                 = %.6e\n", asym)
    @printf("2e- energy from symmetrized F=A+A^T   = %.16f   (diagnostic)\n",
            energy_symmetrized(fock_gpu, S.dens_d, natoms))
    println("---------------------------------------------------------------")
end

import SpecialFunctions

@inline gpu_erf(x::Float64) = SpecialFunctions.erf(x)
@inline gpu_erf(x::Float32) = SpecialFunctions.erf(x)
@inline gpu_erf(x::Float16) = Float16(SpecialFunctions.erf(Float32(x)))
@inline host_erf(x::Float64) = SpecialFunctions.erf(x)
@inline host_erf(x::Float32) = Float32(SpecialFunctions.erf(Float64(x)))
@inline host_erf(x::Float16) = Float16(SpecialFunctions.erf(Float32(x)))

function _erf_selftest_kernel!(out)
    @inbounds out[1] = SpecialFunctions.erf(3.0)
end

if SELECTED_BACKEND === :cuda
    @eval function _run_erf_selftest()
        d = CUDA.CuArray{Float64}(undef, 1)
        CUDA.@cuda threads = 1 _erf_selftest_kernel!(d)
        CUDA.synchronize()
        Array(d)[1]
    end
else
    @eval function _run_erf_selftest()
        d = AMDGPU.ROCArray{Float64}(undef, 1)
        AMDGPU.@roc groupsize = 1 gridsize = 1 _erf_selftest_kernel!(d)
        AMDGPU.synchronize()
        Array(d)[1]
    end
end

function _verify_gpu_erf()
    expected = SpecialFunctions.erf(3.0)
    got = try
        _run_erf_selftest()
    catch err
        error("Device SpecialFunctions.erf() failed for backend=$(SELECTED_BACKEND): $(err)")
    end
    isapprox(got, expected; rtol=1e-12) ||
        error("Device erf() self-test mismatch: got $(got), expected $(expected)")
end

_verify_gpu_erf()

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
    if any(s -> s == "-h" || s == "--help", ARGS)
        print_help()
        return 0
    end
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

    fill!(d_fock, zero(RealT))
    gpu_sync()
    launch()
    gpu_sync()

    h_fock = Array(d_fock)
    erep = contract_erep(h_fock, S.h_dens, S.natoms)

    if !o.csv_output
        report_check(S, h_fock, erep, o)
    else
        println("backend,GPU,precision,sqrt_mode,natoms,ngauss,exec_time_ms")
        for _ in 1:o.num_iter
            fill!(d_fock, zero(RealT))
            gpu_sync()
            elapsed_ms = gpu_elapsed(launch) * 1000.0
            @printf("%s,%s,%s,%s,%d,%d,%f\n",
                    gpu_label(), gpu_name(), REAL_NAME, SQRT_MODE,
                    S.natoms, S.ngauss, elapsed_ms)
        end
    end
    return 0
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(main())
end
