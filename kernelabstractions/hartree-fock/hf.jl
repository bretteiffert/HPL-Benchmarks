# Hartree-Fock problem: host preparation in the run precision, device buffers,
# the untimed first Fock build, event-timed builds, and the fp64 CPU
# reference. ssss (Schwarz screening), ssss_ref, fock_build_cpu and the two
# energy functions are the original code; ssss is generic over precision and
# sqrt mode. Nothing in here prints; failures throw.
#
# The first build is untimed and fills the Fock matrix that the checksum and
# the correctness check use. Every timed build adds into the same Fock
# buffer, as in the original benchmark, so the matrix is not re-zeroed.
#
# Host math goes through the C math library (glibc libm: exp/expf, pow/powf,
# erf/erff), the same functions the C++ host code calls, so the Schwarz
# screening and the fp64 reference match the HIP build bit for bit.

@enum Precision PrecisionDouble PrecisionFloat

precision_name(p::Precision) = p == PrecisionDouble ? "float64" : "float32"

@enum SqrtMode SqrtNative SqrtLegacy

# "native" or "legacy-sqrtf" (the original SQRT_MODE strings)
sqrt_mode_name(m::SqrtMode) = m == SqrtNative ? "native" : "legacy-sqrtf"

# Unique (i >= j) atom pairs.
pair_count(natoms::Integer) = ((UInt64(natoms) * UInt64(natoms)) + UInt64(natoms)) ÷ UInt64(2)

# Unique (ij >= kl) pair quartets: one kernel thread each.
function quartet_count(natoms::Integer)
    nn = pair_count(natoms)
    return ((nn * nn) + nn) ÷ UInt64(2)
end

# Blocks for threads 0..nnnn (thread 0 exits at once), as in the original.
grid_blocks(natoms::Integer, block_size::Integer) =
    ((quartet_count(natoms) + UInt64(1)) + UInt64(block_size) - UInt64(1)) ÷ UInt64(block_size)

# Fock matrix of the untimed first build compared with the fp64 CPU build.
# Numbers only; the report formats them.
Base.@kwdef mutable struct FockCheck
    e_gpu::Float64 = 0.0          # 2e- energy, device Fock summed in fp64
    e_ref::Float64 = 0.0          # 2e- energy, fp64 CPU reference
    rel_loss::Float64 = 0.0       # |e_gpu - e_ref| / |e_ref|: Fock build only
    rel_inpipe::Float64 = 0.0     # |checksum - e_ref| / |e_ref|: including the run-precision sum
    max_abs::Float64 = 0.0        # max |F - F_ref|
    max_rel::Float64 = 0.0        # max |F - F_ref| / max |F_ref|
    rms_abs::Float64 = 0.0        # rms |F - F_ref|
    asym::Float64 = 0.0           # max |F(i,j) - F(j,i)| of the device Fock
    e_symmetrized::Float64 = 0.0  # 2e- energy from F + F^T (diagnostic)
end

# ---- host math: the C library, as the C++ host code ------------------------

const LIBM = "libm.so.6"

h_exp(x::Float64) = ccall((:exp, LIBM), Float64, (Float64,), x)
h_exp(x::Float32) = ccall((:expf, LIBM), Float32, (Float32,), x)
h_pow(a::Float64, b::Float64) = ccall((:pow, LIBM), Float64, (Float64, Float64), a, b)
h_pow(a::Float32, b::Float32) = ccall((:powf, LIBM), Float32, (Float32, Float32), a, b)
h_erf(x::Float64) = ccall((:erf, LIBM), Float64, (Float64,), x)
h_erf(x::Float32) = ccall((:erff, LIBM), Float32, (Float32,), x)
h_sqrt(x::T) where {T <: Union{Float32, Float64}} = sqrt(x)    # IEEE sqrt / sqrtf (arguments are never negative)
h_abs(x::T) where {T <: Union{Float32, Float64}} = abs(x)
h_sq(x) = x * x

h_sqrt_boys(::Val{L}, x::T) where {L, T} = L ? T(h_sqrt(Float32(x))) : h_sqrt(x)

# The original host globals pi, sqrtpi2, dtol, rcut and tobohrs, per precision.
host_sqrtpi2(::Type{T}) where {T} = T(h_pow(3.1415926535897931, -0.5) * 2.0)
host_dtol(::Type{T}) where {T} = T(1.0e-12)
host_rcut(::Type{T}) where {T} = T(1.0e-12)
host_tobohrs(::Type{T}) where {T} = T(1.889725987722)

# i, j, k, l are 0-based atom indices, as in the original.
function ssss(legacy::Val, i::Integer, j::Integer, k::Integer, l::Integer,
              ngauss::Integer, xpnt::Vector{T}, coef::Vector{T},
              geom::Vector{T}, natoms::Integer) where {T}
    # Host constants of the original, typed per precision.
    sqrtpi2 = host_sqrtpi2(T)
    dtol = host_dtol(T)
    rcut = host_rcut(T)
    g(x) = geom[x + 1]
    eri = T(0.0)

    for ib in 1:Int(ngauss)
        for jb in 1:Int(ngauss)
            aij = T(1.0) / (xpnt[ib] + xpnt[jb])
            dij = coef[ib] * coef[jb] *
                  h_exp(-xpnt[ib] * xpnt[jb] * aij *
                        ((h_sq(g(i) - g(j))) +
                         (h_sq(g(natoms + i) - g(natoms + j))) +
                         (h_sq(g(2*natoms + i) - g(2*natoms + j))))) *
                  h_pow(aij, T(1.5))
            if h_abs(dij) > dtol
                xij = aij * (xpnt[ib] * g(i) + xpnt[jb] * g(j))
                yij = aij * (xpnt[ib] * g(natoms + i) + xpnt[jb] * g(natoms + j))
                zij = aij * (xpnt[ib] * g(2*natoms + i) + xpnt[jb] * g(2*natoms + j))

                for kb in 1:Int(ngauss)
                    for lb in 1:Int(ngauss)
                        akl = T(1.0) / (xpnt[kb] + xpnt[lb])
                        dkl = dij * coef[kb] * coef[lb] *
                              h_exp(-xpnt[kb] * xpnt[lb] * akl *
                                    ((h_sq(g(k) - g(l))) +
                                     (h_sq(g(natoms + k) - g(natoms + l))) +
                                     (h_sq(g(2*natoms + k) - g(2*natoms + l))))) *
                              h_pow(akl, T(1.5))

                        if h_abs(dkl) > dtol
                            aijkl = (xpnt[ib] + xpnt[jb]) *
                                    (xpnt[kb] + xpnt[lb]) /
                                    (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                            tt = aijkl * (h_sq(xij - akl * (xpnt[kb] * g(k) + xpnt[lb] * g(l))) +
                                          h_sq(yij - akl * (xpnt[kb] * g(natoms + k) + xpnt[lb] * g(natoms + l))) +
                                          h_sq(zij - akl * (xpnt[kb] * g(2*natoms + k) + xpnt[lb] * g(2*natoms + l))))
                            f0t = sqrtpi2
                            if tt > rcut
                                f0t = h_pow(tt, T(-0.5)) * h_erf(h_sqrt_boys(legacy, tt))
                            end
                            eri += (dkl * f0t * h_sqrt_boys(legacy, aijkl))
                        end
                    end
                end
            end
        end
    end
    return eri
end

const REF_SQRTPI2 = 1.12837916709551255856
const REF_DTOL    = 1.0e-12
const REF_RCUT    = 1.0e-12

d_sq(x::Float64) = x * x

function ssss_ref(i::Int, j::Int, k::Int, l::Int,
                  ngauss::Int, xpnt::Vector{Float64}, coef::Vector{Float64},
                  geom::Vector{Float64}, natoms::Int)::Float64
    @inbounds begin
    g(x) = geom[x + 1]
    eri = 0.0
    for ib in 1:ngauss
    for jb in 1:ngauss
        aij = 1.0 / (xpnt[ib] + xpnt[jb])
        dij = coef[ib] * coef[jb] *
            h_exp(-xpnt[ib] * xpnt[jb] * aij *
                  (d_sq(g(i) - g(j)) +
                   d_sq(g(natoms + i) - g(natoms + j)) +
                   d_sq(g(2*natoms + i) - g(2*natoms + j)))) *
            h_pow(aij, 1.5)
        if abs(dij) > REF_DTOL
            xij = aij * (xpnt[ib]*g(i) + xpnt[jb]*g(j))
            yij = aij * (xpnt[ib]*g(natoms + i) + xpnt[jb]*g(natoms + j))
            zij = aij * (xpnt[ib]*g(2*natoms + i) + xpnt[jb]*g(2*natoms + j))
            for kb in 1:ngauss
            for lb in 1:ngauss
                akl = 1.0 / (xpnt[kb] + xpnt[lb])
                dkl = dij * coef[kb] * coef[lb] *
                    h_exp(-xpnt[kb] * xpnt[lb] * akl *
                          (d_sq(g(k) - g(l)) +
                           d_sq(g(natoms + k) - g(natoms + l)) +
                           d_sq(g(2*natoms + k) - g(2*natoms + l)))) *
                    h_pow(akl, 1.5)
                if abs(dkl) > REF_DTOL
                    aijkl = (xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb]) /
                            (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                    tt = aijkl * (
                          d_sq(xij - akl*(xpnt[kb]*g(k) + xpnt[lb]*g(l))) +
                          d_sq(yij - akl*(xpnt[kb]*g(natoms + k) + xpnt[lb]*g(natoms + l))) +
                          d_sq(zij - akl*(xpnt[kb]*g(2*natoms + k) + xpnt[lb]*g(2*natoms + l))))
                    f0t = REF_SQRTPI2
                    if tt > REF_RCUT
                        f0t = h_pow(tt, -0.5) * h_erf(h_sqrt(tt))
                    end
                    eri += dkl * f0t * h_sqrt(aijkl)
                end
            end
            end
        end
    end
    end
    return eri
    end
end

function fock_build_cpu(ngauss::Int, natoms::Int,
                        geom::Vector{Float64}, xpnt::Vector{Float64}, coef::Vector{Float64},
                        dens::Vector{Float64}, fock_out::Vector{Float64})
    @inbounds begin
    fill!(fock_out, 0.0)
    f(x) = x + 1     # 0-based -> 1-based

    nn_ref = div(natoms * natoms + natoms, 2)
    schwarz = zeros(Float64, nn_ref + 1)
    ijr = 0
    for i in 0:natoms-1
        for j in 0:i
            ijr += 1
            schwarz[f(ijr)] = h_sqrt(abs(ssss_ref(i, j, i, j, ngauss, xpnt, coef, geom, natoms)))
        end
    end

    for i in 0:natoms-1
    for j in 0:i
        ij = div(i * (i + 1), 2) + j + 1
        for k in 0:natoms-1
        for l in 0:k
            kl = div(k * (k + 1), 2) + l + 1
            kl > ij && continue
            (schwarz[f(ij)] * schwarz[f(kl)] > REF_DTOL) || continue

            eri = ssss_ref(i, j, k, l, ngauss, xpnt, coef, geom, natoms)
            i == j && (eri *= 0.5)
            k == l && (eri *= 0.5)
            (i == k && j == l) && (eri *= 0.5)

            fock_out[f(i*natoms + j)] += dens[f(k*natoms + l)] * eri * 4.0
            fock_out[f(k*natoms + l)] += dens[f(i*natoms + j)] * eri * 4.0
            fock_out[f(i*natoms + k)] -= dens[f(j*natoms + l)] * eri
            fock_out[f(i*natoms + l)] -= dens[f(j*natoms + k)] * eri
            fock_out[f(j*natoms + k)] -= dens[f(i*natoms + l)] * eri
            fock_out[f(j*natoms + l)] -= dens[f(i*natoms + k)] * eri
        end
        end
    end
    end
    end
    return fock_out
end

function energy_from_fock(fock::Vector{Float64}, dens::Vector{Float64}, natoms::Int)
    e = 0.0
    for i in 0:natoms-1, j in 0:natoms-1
        e += fock[i*natoms + j + 1] * dens[i*natoms + j + 1]
    end
    return e * 0.5
end

function energy_symmetrized(fock::Vector{Float64}, dens::Vector{Float64}, natoms::Int)
    e = 0.0
    for i in 0:natoms-1, j in 0:natoms-1
        fv = (i == j) ? fock[i*natoms + j + 1] :
                        fock[i*natoms + j + 1] + fock[j*natoms + i + 1]
        e += fv * dens[i*natoms + j + 1]
    end
    return e * 0.5
end

# ---- the run ---------------------------------------------------------------

abstract type HartreeFock end

mutable struct KAHartreeFock{T, L, A, K} <: HartreeFock
    ngauss::UInt32
    natoms::UInt32
    nn::UInt64
    nnnn::UInt64
    nsq::Int
    blk_size::UInt32
    n_blks::UInt32

    # Run-precision host data (as the original h_* arrays).
    h_xpnt::Vector{T}; h_coef::Vector{T}; h_geom::Vector{T}
    h_dens::Vector{T}; h_fock::Vector{T}; h_schwarz::Vector{T}
    # fp64 copies for the CPU reference (as the original *_d arrays).
    xpnt_d::Vector{Float64}; coef_d::Vector{Float64}; geom_d::Vector{Float64}; dens_d::Vector{Float64}

    # Device buffers.
    xpnt::A; coef::A; geom::A; dens::A; fock::A; schwarz::A

    kernel::K
    timer::DeviceTimer

    checksum::Float64
    built::Bool
end

function device_copy(v::Vector{T}) where {T}
    d = KernelAbstractions.allocate(BACKEND, T, length(v))
    KernelAbstractions.copyto!(BACKEND, d, v)
    return d
end

# Prepares host data (normalised coefficients, geometry in bohr, density,
# Schwarz screening in the run precision), allocates and copies device buffers.
function KAHartreeFock{T, L}(deck::Deck, block_size::UInt32) where {T, L}
    ngauss = deck.ngauss
    natoms = deck.natoms
    nn = pair_count(natoms)
    nsq = Int(natoms) * Int(natoms)
    nnnn = quartet_count(natoms)
    n_blks = UInt32(grid_blocks(natoms, block_size))     # validated by main
    na = Int(natoms)

    # Same preparation as the original main().
    xpnt_d = copy(deck.xpnt); coef_d = copy(deck.coef); geom_d = copy(deck.geom)
    h_xpnt = T.(deck.xpnt); h_coef = T.(deck.coef); h_geom = T.(deck.geom)

    h_dens = Vector{T}(undef, nsq); h_fock = Vector{T}(undef, nsq)
    h_schwarz = zeros(T, Int(nn) + 1)
    dens_d = Vector{Float64}(undef, nsq)
    for i in 0:na-1
        for j in 0:na-1
            h_dens[i*na + j + 1] = T(0.1)
            dens_d[i*na + j + 1] = 0.1
        end
        h_dens[i*na + i + 1] = T(1.0)
        dens_d[i*na + i + 1] = 1.0
    end

    for i in 1:Int(ngauss)
        h_coef[i] = h_coef[i] * h_pow((T(2.0) * h_xpnt[i]), T(0.75))
        coef_d[i] = coef_d[i] * h_pow((2.0 * xpnt_d[i]), 0.75)
    end

    tobohrs = host_tobohrs(T)
    for i in 1:na
        h_geom[0 * na + i] *= tobohrs
        h_geom[1 * na + i] *= tobohrs
        h_geom[2 * na + i] *= tobohrs
        geom_d[0 * na + i] *= 1.889725987722
        geom_d[1 * na + i] *= 1.889725987722
        geom_d[2 * na + i] *= 1.889725987722
    end

    fill!(h_fock, T(0.0))

    legacy = Val(L)
    ij = 0
    for i in 0:na-1
        for j in 0:i
            ij = ij + 1
            eri = ssss(legacy, i, j, i, j, ngauss, h_xpnt, h_coef, h_geom, na)
            h_schwarz[ij + 1] = h_sqrt(h_abs(eri))
        end
    end

    xpnt = device_copy(h_xpnt)
    coef = device_copy(h_coef)
    geom = device_copy(h_geom)
    dens = device_copy(h_dens)
    fock = device_copy(h_fock)
    schwarz = device_copy(h_schwarz)
    KernelAbstractions.synchronize(BACKEND)

    # Static workgroup size, as threads(blk_size, 1, 1).
    kernel = hartree_fock_kernel!(BACKEND, Int(block_size))
    timer = make_timer()

    return KAHartreeFock{T, L, typeof(fock), typeof(kernel)}(
        ngauss, natoms, nn, nnnn, nsq, block_size, n_blks,
        h_xpnt, h_coef, h_geom, h_dens, h_fock, h_schwarz,
        xpnt_d, coef_d, geom_d, dens_d,
        xpnt, coef, geom, dens, fock, schwarz,
        kernel, timer, 0.0, false)
end

# grid(n_blks) x threads(blk_size): ndrange is exactly n_blks * blk_size work-items.
function launch!(hf::KAHartreeFock{T, L}) where {T, L}
    hf.kernel(hf.nnnn, hf.ngauss, hf.natoms,
              hf.geom, hf.xpnt, hf.coef, hf.dens,
              hf.schwarz, hf.fock, Val(L);
              ndrange = Int(hf.n_blks) * Int(hf.blk_size))
    return nothing
end

# Untimed first build; blocks until done and keeps a host copy of Fock.
function first_build!(hf::KAHartreeFock{T}) where {T}
    launch!(hf)
    KernelAbstractions.synchronize(BACKEND)

    KernelAbstractions.copyto!(BACKEND, hf.h_fock, hf.fock)
    KernelAbstractions.synchronize(BACKEND)

    na = Int(hf.natoms)
    erep = T(0.0)
    for i in 0:na-1
        for j in 0:na-1
            erep += hf.h_fock[i*na + j + 1] * hf.h_dens[i*na + j + 1]
        end
    end
    hf.checksum = Float64(erep) * 0.5
    hf.built = true
    return nothing
end

# 2e- energy checksum of the first build, summed in the run precision
# (the original "2e- energy=" line).
checksum(hf::KAHartreeFock) = hf.checksum

# One timed build (accumulates into Fock). Device-measured seconds.
# Throws if the launch or the timer fails.
function timed_build!(hf::KAHartreeFock)::Float64
    KernelAbstractions.synchronize(BACKEND)
    elapsed = time_launch(hf.timer, () -> launch!(hf))
    if !(elapsed > 0.0)
        error("device timer returned no measurement for the Fock build")
    end
    return elapsed
end

# fp64 CPU reference compared with the first build's Fock matrix.
function check(hf::KAHartreeFock)::FockCheck
    hf.built || error("check() called before first_build()")

    # Same metrics as the original precision check.
    nsq = hf.nsq
    na = Int(hf.natoms)
    fock_gpu = Float64.(hf.h_fock)
    fock_ref = Vector{Float64}(undef, nsq)

    fock_build_cpu(Int(hf.ngauss), na, hf.geom_d, hf.xpnt_d, hf.coef_d, hf.dens_d, fock_ref)

    c = FockCheck()
    c.e_gpu = energy_from_fock(fock_gpu, hf.dens_d, na)
    c.e_ref = energy_from_fock(fock_ref, hf.dens_d, na)

    denom = abs(c.e_ref) > 0.0 ? abs(c.e_ref) : 1.0
    c.rel_loss = abs(c.e_gpu - c.e_ref) / denom
    c.rel_inpipe = abs(hf.checksum - c.e_ref) / denom

    maxd = 0.0; sumsq = 0.0; fscale = 0.0
    for t in 1:nsq
        d = abs(fock_gpu[t] - fock_ref[t])
        d > maxd && (maxd = d)
        sumsq += d * d
        abs(fock_ref[t]) > fscale && (fscale = abs(fock_ref[t]))
    end
    c.rms_abs = h_sqrt(sumsq / Float64(nsq))
    fscale == 0.0 && (fscale = 1.0)
    c.max_abs = maxd
    c.max_rel = maxd / fscale

    asym = 0.0
    for i in 0:na-1, j in 0:na-1
        d = abs(fock_gpu[i*na + j + 1] - fock_gpu[j*na + i + 1])
        d > asym && (asym = d)
    end
    c.asym = asym
    c.e_symmetrized = energy_symmetrized(fock_gpu, hf.dens_d, na)
    return c
end

function make_hartree_fock(p::Precision, m::SqrtMode, deck::Deck, block_size::UInt32)::HartreeFock
    legacy = (m == SqrtLegacy)
    if p == PrecisionDouble
        return legacy ? KAHartreeFock{Float64, true}(deck, block_size) :
                        KAHartreeFock{Float64, false}(deck, block_size)
    end
    return legacy ? KAHartreeFock{Float32, true}(deck, block_size) :
                    KAHartreeFock{Float32, false}(deck, block_size)
end
