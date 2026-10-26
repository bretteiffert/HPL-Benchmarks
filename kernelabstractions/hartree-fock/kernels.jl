# kernels.jl — Hartree-Fock device code (KernelAbstractions).
#
# Math unchanged from the original hartree-fock.jl. Forced by run-time precision
# and run-time --legacy-sqrt (the original fixed both as global constants at load):
#   * RealT                -> T = eltype(fock)
#   * DTOL / RCUT / SQRTPI2_DEV -> hf_dtol(T) / hf_rcut(T) / hf_sqrtpi2(T)  (same values)
#   * sqrt_boys(x)         -> sqrt_boys(x, legacy)  with legacy::Val{Bool}, so the
#                             branch still compiles away as with the old constant.

module Kernels

using KernelAbstractions
using Atomix
import SpecialFunctions

export hf_kernel!, erf_selftest_kernel!, hf_dtol, hf_rcut, hf_sqrtpi2, hf_tobohrs,
       sq, idx_sqrt, sqrt_boys, gpu_erf, host_erf

hf_dtol(::Type{T}) where {T} = T(1.0e-12)
hf_rcut(::Type{T}) where {T} = T(1.0e-12)
hf_sqrtpi2(::Type{T}) where {T} = T(1.12837916709551255856)
hf_tobohrs(::Type{T}) where {T} = T(1.889725987722)

@inline idx_sqrt(x::UInt64) = unsafe_trunc(UInt64, sqrt(Float32(x)))

@inline sqrt_boys(x::T, ::Val{false}) where {T} = sqrt(x)
@inline sqrt_boys(x::T, ::Val{true}) where {T} = T(sqrt(Float32(x)))

@inline sq(x) = x * x

@inline gpu_erf(x::Float64) = SpecialFunctions.erf(x)
@inline gpu_erf(x::Float32) = SpecialFunctions.erf(x)
@inline host_erf(x::Float64) = SpecialFunctions.erf(x)
@inline host_erf(x::Float32) = Float32(SpecialFunctions.erf(Float64(x)))

"Writes gpu_erf(x) for each x; used to check the device erf before the run."
@kernel function erf_selftest_kernel!(out, @Const(x))
    t = @index(Global, Linear)
    @inbounds out[t] = gpu_erf(x[t])
end

@kernel function hf_kernel!(fock, nnnn, ngauss, natoms,
                            @Const(geom), @Const(xpnt), @Const(coef),
                            @Const(dens), @Const(schwarz), legacy)
    T = eltype(fock)
    DTOL = hf_dtol(T)
    RCUT = hf_rcut(T)
    SQRTPI2_DEV = hf_sqrtpi2(T)

    # Match the native zero-based launch: global lane zero is inactive and
    # lanes 1:nnnn each own one quartet.
    idx = @index(Global, Linear)
    ijkl = UInt64(idx - 1)
    if UInt64(1) <= ijkl <= UInt64(nnnn)
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

                eri = zero(T)
                one_T = one(T)
                p15 = T(1.5)
                pm05 = T(-0.5)
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
                                    f0t = tt^pm05 * gpu_erf(sqrt_boys(tt, legacy))
                                end
                                eri += dkl * f0t * sqrt_boys(aijkl, legacy)
                            end
                        end
                    end
                end

                i == j && (eri *= T(0.5))
                k == l && (eri *= T(0.5))
                (i == k && j == l) && (eri *= T(0.5))

                four = T(4.0)
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

end # module Kernels
