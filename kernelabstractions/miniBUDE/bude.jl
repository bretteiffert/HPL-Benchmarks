# bude.jl — miniBUDE device buffers, timed launches, throughput formulas and
# correctness metrics. No printing here.

module Bude

using KernelAbstractions
using ..Kernels
using ..Deck
using ..Backend: elapsed_seconds, device_sync

export PPWIS, BudeDevice, launch!, read_energies, throughput, Metrics, compute_metrics,
       ULP_BUCKETS, ULP_NBUCKETS, ulp_bucket_label, REPORT_TOLERANCE_PCT

const PPWIS = (1, 2, 4, 8, 16, 32, 64, 128)
const REPORT_TOLERANCE_PCT = 0.025

convert_atoms(::Type{T}, xs::Vector{DiskAtom}) where {T} =
    [AtomT{T}(T(a.x), T(a.y), T(a.z), a.type) for a in xs]

convert_forcefield(::Type{T}, xs::Vector{DiskFFParams}) where {T} =
    [FFParamsT{T}(f.hbtype, T(f.radius), T(f.hphb), T(f.elsc)) for f in xs]

struct BudeDevice{T,A,P,L,F,K,V}
    results::A
    protein::P
    ligand::L
    forcefield::F
    transforms::NTuple{6,A}
    kern::K
    ppwi::V
    natlig::Int32
    natpro::Int32
    nposes::Int32
    wgsize::Int
    ndrange::Int
    context_s::Float64       # allocation + host-to-device copies, seconds
end

function BudeDevice(::Type{T}, backend, d::DeckData, wgsize::Int, ppwi::Int) where {T}
    hProtein = convert_atoms(T, d.protein)
    hLigand = convert_atoms(T, d.ligand)
    hForcefield = convert_forcefield(T, d.forcefield)
    n = nposes(d)

    t0 = time_ns()
    dProtein = KernelAbstractions.allocate(backend, AtomT{T}, length(hProtein))
    dLigand = KernelAbstractions.allocate(backend, AtomT{T}, length(hLigand))
    dForcefield = KernelAbstractions.allocate(backend, FFParamsT{T}, length(hForcefield))
    copyto!(dProtein, hProtein)
    copyto!(dLigand, hLigand)
    copyto!(dForcefield, hForcefield)
    transforms = ntuple(6) do i
        a = KernelAbstractions.allocate(backend, T, n)
        copyto!(a, T.(d.poses[i]))
        a
    end
    dResults = KernelAbstractions.allocate(backend, T, n)
    device_sync()
    context_s = (time_ns() - t0) * 1e-9

    nblocks = cld(cld(n, ppwi), wgsize)
    kern = fasten_main!(backend, wgsize)
    vp = Val(ppwi)
    return BudeDevice{T,typeof(dResults),typeof(dProtein),typeof(dLigand),
                      typeof(dForcefield),typeof(kern),typeof(vp)}(
        dResults, dProtein, dLigand, dForcefield, transforms, kern, vp,
        Int32(natlig(d)), Int32(natpro(d)), Int32(n), wgsize, nblocks * wgsize, context_s)
end

"One launch. Returns (kernel_s from device events, wall_s from the host clock)."
function launch!(b::BudeDevice)
    w0 = time_ns()
    t = elapsed_seconds() do
        tf = b.transforms
        b.kern(b.results, b.protein, b.ligand, tf[1], tf[2], tf[3], tf[4], tf[5], tf[6],
               b.forcefield, b.natlig, b.natpro, b.nposes, Int32(b.wgsize), b.ppwi;
               ndrange = b.ndrange)
    end
    return t, (time_ns() - w0) * 1e-9
end

read_energies(b::BudeDevice) = Float64.(Array(b.results))

"(gflops, gfinsts, ginteractions) per second for an average time `avg_s`."
function throughput(d::DeckData, ppwi::Int, avg_s::Float64)
    nl, np, n = natlig(d), natpro(d), nposes(d)
    ops_per_wg = ppwi * 27 + nl * (2 + ppwi * 18 + np * (10 + ppwi * 30)) + ppwi
    total_ops = Float64(ops_per_wg) * (Float64(n) / Float64(ppwi))
    gflops = total_ops / avg_s / 1e9
    gfinsts = Float64(25 * np * nl * n) / avg_s / 1e9
    ginter = Float64(n) * nl * np / avg_s / 1e9
    return gflops, gfinsts, ginter
end

# ------------------------------------------------------------------ metrics (unchanged)

function ulp_ordered(f::Float32)
    i = reinterpret(Int32, f)
    return i < 0 ? Int64(0x80000000) - Int64(reinterpret(UInt32, i)) : Int64(i)
end

function ulp_ordered(d::Float64)
    i = reinterpret(Int64, d)
    return i < 0 ? (typemin(Int64) - i) : i
end

function ulp_distance(a::T, b::T) where {T<:AbstractFloat}
    (isnan(a) || isnan(b)) && return typemax(Int64)
    a == b && return Int64(0)
    (isinf(a) || isinf(b)) && return typemax(Int64)
    d = ulp_ordered(a) - ulp_ordered(b)
    return d < 0 ? -d : d
end

const ULP_BUCKETS = (Int64(1), Int64(2), Int64(4), Int64(8), Int64(16),
                     Int64(64), Int64(256), Int64(1024), Int64(4096))
const ULP_NBUCKETS = length(ULP_BUCKETS) + 1

function ulp_bucket_label(i::Int)
    i == 1 && return "0"
    i == ULP_NBUCKETS && return ">=" * string(ULP_BUCKETS[end])
    lo = ULP_BUCKETS[i-1]
    hi = ULP_BUCKETS[i] - 1
    return lo == hi ? string(lo) : string(lo) * "-" * string(hi)
end

mutable struct Metrics
    l2RelNorm::Float64
    rmsAbs::Float64
    meanSigned::Float64
    maxAbs::Float64
    maxAbsIdx::Int
    maxRel::Float64
    maxRelIdx::Int
    maxUlp::Int64
    maxUlpIdx::Int
    ulpHistogram::Vector{Int}
    compared::Int
    nonFinite::Int
    beyondTolerance::Int
end

Metrics() = Metrics(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0, Int64(0), 0,
                    zeros(Int, ULP_NBUCKETS), 0, 0, 0)

mutable struct Kahan
    sum::Float64
    c::Float64
end
Kahan() = Kahan(0.0, 0.0)

function kadd!(k::Kahan, x::Float64)
    y = x - k.c
    t = k.sum + y
    k.c = (t - k.sum) - y
    k.sum = t
    return nothing
end

function compute_metrics(measured::Vector{Float64}, reference::Vector{Float64},
                         precisionBits::Int)
    m = Metrics()
    n = min(length(measured), length(reference))
    errSq = Kahan(); refSq = Kahan(); signedErr = Kahan()
    for i in 1:n
        x = measured[i]
        r = reference[i]
        if !isfinite(x) || !isfinite(r)
            m.nonFinite += 1
            continue
        end
        err = x - r
        absErr = abs(err)
        kadd!(errSq, err * err)
        kadd!(refSq, r * r)
        kadd!(signedErr, err)
        if absErr > m.maxAbs
            m.maxAbs = absErr
            m.maxAbsIdx = i - 1
        end
        denom = abs(r)
        if denom > 0.0
            rel = absErr / denom
            if rel > m.maxRel
                m.maxRel = rel
                m.maxRelIdx = i - 1
            end
            if rel * 100.0 > REPORT_TOLERANCE_PCT
                m.beyondTolerance += 1
            end
        end
        u = precisionBits == 32 ? ulp_distance(Float32(x), Float32(r)) : ulp_distance(x, r)
        if u > m.maxUlp
            m.maxUlp = u
            m.maxUlpIdx = i - 1
        end
        bucket = ULP_NBUCKETS
        for b in 1:length(ULP_BUCKETS)
            if u < ULP_BUCKETS[b]
                bucket = b
                break
            end
        end
        m.ulpHistogram[bucket] += 1
        m.compared += 1
    end
    if m.compared > 0
        m.rmsAbs = sqrt(errSq.sum / m.compared)
        m.meanSigned = signedErr.sum / m.compared
        m.l2RelNorm = refSq.sum > 0.0 ? sqrt(errSq.sum) / sqrt(refSq.sum) : 0.0
    end
    return m
end

end # module Bude
