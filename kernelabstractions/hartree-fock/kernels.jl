# Hartree-Fock device code: precision-specific math helpers and the Fock
# build kernel. The kernel body is the HIP hartree_fock kernel line for line;
# the only changes are the ones Julia/KernelAbstractions forces:
#   - `@kernel` instead of `__global__`; the thread index
#     blockIdx.x * blockDim.x + threadIdx.x is `@index(Global, Linear) - 1`
#     (KernelAbstractions indices are 1-based, the original's are 0-based);
#   - the early `return` for thread 0 and threads past nnnn is an enclosing
#     `if` (KernelAbstractions kernels must not return early);
#   - the integer index math is unchanged and 0-based; every array access
#     adds 1 through `ix` (Julia arrays are 1-based);
#   - `for (int ib = 0; ib < (int)ngauss; ib++) ... xpnt[ib]` is
#     `for ib in 1:ngauss ... xpnt[ib]` (same Gaussians, 1-based);
#   - atomicAdd is `Atomix.@atomic :monotonic` (relaxed, as atomicAdd);
#   - the body is `@inbounds` (Julia checks bounds by default, C does not);
#   - (uint64_t)sqrtf(2*x) is unsafe_trunc(UInt64, sqrt(Float32(2x))), the
#     same uint64 -> float -> sqrtf -> truncation.
# The math helpers call the same device functions the HIP overloads did:
# Base.exp/sqrt/abs/^ and SpecialFunctions.erf, which AMDGPU.jl maps to OCML
# (__ocml_exp_f32/f64, __ocml_pow_*, __ocml_erf_*, ...) and CUDA.jl to
# libdevice (__nv_expf/__nv_exp, __nv_powf/__nv_pow, __nv_erff/__nv_erf, ...).
# Julia does not contract a*b+c into an FMA unless asked; hipcc does by
# default, so last-bit differences against the HIP build are expected.

using KernelAbstractions
import Atomix
import SpecialFunctions

@inline r_sqrt(x::T) where {T <: Union{Float32, Float64}} = sqrt(x)
@inline r_exp(x::T) where {T <: Union{Float32, Float64}} = exp(x)
@inline r_erf(x::T) where {T <: Union{Float32, Float64}} = SpecialFunctions.erf(x)
@inline r_pow(a::T, b::T) where {T <: Union{Float32, Float64}} = a^b
@inline r_abs(x::T) where {T <: Union{Float32, Float64}} = abs(x)
@inline r_sq(x) = x * x

# LEGACY_SQRTF=1 behaviour (--legacy-sqrt): the Boys-function square roots go
# through sqrtf in float even in a float64 build. Native mode uses r_sqrt.
@inline function r_sqrt_boys(::Val{LegacySqrt}, x::T) where {LegacySqrt, T}
    LegacySqrt && return T(sqrt(Float32(x)))
    return r_sqrt(x)
end

# 0-based index -> 1-based array position
@inline ix(x::Integer) = (x % Int) + 1

@kernel function hartree_fock_kernel!(nnnn::UInt64, ngauss::UInt32, natoms::UInt32,
        geom, xpnt, coef, dens, schwarz, fock, legacy::Val)
    @inbounds begin
    real_t = eltype(fock)
    # Run-time precision: the screening constants are per-type locals with the
    # same literal values as the original globals (dtol) and dev_consts.
    dtol = real_t(1.0e-12)
    dev_consts_sqrtpi2 = real_t(1.12837916709551255856)
    dev_consts_rcut = real_t(1.0e-12)

    ijkl = (@index(Global, Linear) - 1) % UInt64
    if !(ijkl == 0 || ijkl > nnnn)

    ij = unsafe_trunc(UInt64, sqrt(Float32(UInt64(2) * ijkl)))
    n = (ij*ij + ij) ÷ UInt64(2)
    while n < ijkl
        ij += UInt64(1)
        n = (ij*ij + ij) ÷ UInt64(2)
    end
    kl = ijkl - (ij * ij - ij) ÷ UInt64(2)
    if schwarz[ix(ij)] * schwarz[ix(kl)] > dtol
        i = unsafe_trunc(UInt64, sqrt(Float32(UInt64(2) * ij))) - UInt64(1)
        n = (i * i + i) ÷ UInt64(2)
        while n < ij
            i += UInt64(1)
            n = (i * i + i) ÷ UInt64(2)
        end
        j = ij - (i * i - i) ÷ UInt64(2)

        k = unsafe_trunc(UInt64, sqrt(Float32(UInt64(2) * kl)))
        n = (k * k + k) ÷ UInt64(2)
        while n < kl
            k += UInt64(1)
            n = (k * k + k ) ÷ UInt64(2)
        end
        l = kl - (k * k - k) ÷ UInt64(2)
        i -= UInt64(1)
        j -= UInt64(1)
        k -= UInt64(1)
        l -= UInt64(1)

        eri = real_t(0.0)
        for ib in 1:Int(ngauss)
            for jb in 1:Int(ngauss)
                aij = real_t(1.0) / (xpnt[ib] + xpnt[jb])
                dij = coef[ib] * coef[jb] *
                    r_exp(-xpnt[ib] * xpnt[jb] * aij *
                            (r_sq(geom[ix(i)] - geom[ix(j)]) +
                             r_sq(geom[ix(1*natoms + i)] - geom[ix(1*natoms + j)]) +
                             r_sq(geom[ix(2*natoms + i)] - geom[ix(2*natoms + j)]))) *
                    r_pow(aij, real_t(1.5))
                if r_abs(dij) > dtol
                    xij = aij * (xpnt[ib] * geom[ix(i)] + xpnt[jb] * geom[ix(j)])
                    yij = aij * (xpnt[ib] * geom[ix(1*natoms + i)] + xpnt[jb] * geom[ix(1*natoms + j)])
                    zij = aij * (xpnt[ib] * geom[ix(2*natoms + i)] + xpnt[jb] * geom[ix(2*natoms + j)])
                    for kb in 1:Int(ngauss)
                        for lb in 1:Int(ngauss)
                            akl = real_t(1.0) / (xpnt[kb] + xpnt[lb])
                            dkl = dij * coef[kb] * coef[lb] *
                                r_exp(-xpnt[kb] * xpnt[lb] * akl *
                                        (r_sq(geom[ix(k)] - geom[ix(l)]) +
                                         r_sq(geom[ix(natoms + k)] - geom[ix(natoms + l)]) +
                                         r_sq(geom[ix(2*natoms + k)] - geom[ix(2*natoms + l)]))) * r_pow(akl, real_t(1.5))
                            if r_abs(dkl) > dtol
                                aijkl = (xpnt[ib] + xpnt[jb]) *
                                        (xpnt[kb] + xpnt[lb]) /
                                        (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                                tt = aijkl * (r_sq(xij - akl * (xpnt[kb] * geom[ix(k)] + xpnt[lb] * geom[ix(l)])) +
                                        r_sq(yij - akl * (xpnt[kb] * geom[ix(natoms + k)] + xpnt[lb] * geom[ix(natoms + l)])) +
                                        r_sq(zij - akl * (xpnt[kb] * geom[ix(2*natoms + k)] + xpnt[lb] * geom[ix(2*natoms + l)])))
                                f0t = dev_consts_sqrtpi2
                                if tt > dev_consts_rcut
                                    f0t = (r_pow(tt, real_t(-0.5)) * r_erf(r_sqrt_boys(legacy, tt)))
                                end
                                eri += dkl * f0t * r_sqrt_boys(legacy, aijkl)
                            end
                        end
                    end
                end
            end
        end
        if i == j
            eri *= real_t(0.5)
        end
        if k == l
            eri *= real_t(0.5)
        end
        if i == k && j == l
            eri *= real_t(0.5)
        end
        Atomix.@atomic :monotonic fock[ix(i*natoms + j)] += dens[ix(k*natoms + l)]*eri*real_t(4.0)
        Atomix.@atomic :monotonic fock[ix(k*natoms + l)] += dens[ix(i*natoms + j)]*eri*real_t(4.0)
        Atomix.@atomic :monotonic fock[ix(i*natoms + k)] += -dens[ix(j*natoms + l)]*eri
        Atomix.@atomic :monotonic fock[ix(i*natoms + l)] += -dens[ix(j*natoms + k)]*eri
        Atomix.@atomic :monotonic fock[ix(j*natoms + k)] += -dens[ix(i*natoms + l)]*eri
        Atomix.@atomic :monotonic fock[ix(j*natoms + l)] += -dens[ix(i*natoms + k)]*eri
    end

    end  # thread 0 and threads past nnnn do nothing
    end  # @inbounds
end
