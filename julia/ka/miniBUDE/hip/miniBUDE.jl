#!/usr/bin/env julia
# Runs the KernelAbstractions miniBUDE molecular docking benchmark using binary deck data and the backend selected by `STENCIL_GPU`. `PRECISION_BITS` currently selects Float64, while CLI options control devices, iterations, pose counts, poses per work-item, workgroup sizes, deck location, output, and reporting. The harness records vendor-event and wall-clock timing, compares energies with deck data, and adds an in-process fp64 differential pass only for fp32 benchmark runs.

using KernelAbstractions
using Printf

include(joinpath(@__DIR__, "../ka_gpu_shim.jl"))

const DEFAULT_ITERS = 8
const DEFAULT_WARMUP = 2
const DEFAULT_ENERGY_ENTRIES = 8
const DEFAULT_DATA_DIR = "../../../../data/miniBUDE/bm1"

const REPORT_TOLERANCE_PCT = 0.025

const PRECISION_BITS = 64


const HBTYPE_F = Int32(70)
const HBTYPE_E = Int32(69)

const PPWIS = (1, 2, 4, 8, 16, 32, 64, 128)

struct DiskAtom
    x::Float32
    y::Float32
    z::Float32
    type::Int32
end

struct DiskFFParams
    hbtype::Int32
    radius::Float32
    hphb::Float32
    elsc::Float32
end

struct AtomT{T}
    x::T
    y::T
    z::T
    type::Int32
end

struct FFParamsT{T}
    hbtype::Int32
    radius::T
    hphb::T
    elsc::T
end

convert_atoms(::Type{T}, xs::Vector{DiskAtom}) where {T} =
    [AtomT{T}(T(a.x), T(a.y), T(a.z), a.type) for a in xs]

convert_forcefield(::Type{T}, xs::Vector{DiskFFParams}) where {T} =
    [FFParamsT{T}(f.hbtype, T(f.radius), T(f.hphb), T(f.elsc)) for f in xs]

function read_nstruct(::Type{T}, path::String) where {T}
    isfile(path) || throw(ArgumentError("Bad file: $path"))
    bytes = read(path)
    n = length(bytes) ÷ sizeof(T)
    return collect(reinterpret(T, bytes[1:n*sizeof(T)]))
end

mutable struct Params
    protein::Vector{DiskAtom}
    ligand::Vector{DiskAtom}
    forcefield::Vector{DiskFFParams}
    poses::Vector{Vector{Float32}}
    refEnergies::Vector{Float32}
    maxPoses::Int
    iterations::Int
    warmupIterations::Int
    outRows::Int
    deckDir::String
    output::String
    deviceSelector::String
    csv::Bool
    list::Bool
end

Params() = Params(DiskAtom[], DiskAtom[], DiskFFParams[], [Float32[] for _ in 1:6],
                  Float32[], 0, DEFAULT_ITERS, DEFAULT_WARMUP, DEFAULT_ENERGY_ENTRIES,
                  DEFAULT_DATA_DIR, "", "", false, false)

total_iterations(p::Params) = p.iterations + p.warmupIterations
natpro(p::Params) = length(p.protein)
natlig(p::Params) = length(p.ligand)
ntypes(p::Params) = length(p.forcefield)
nposes(p::Params) = length(p.poses[1])

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

        u = precisionBits == 32 ? ulp_distance(Float32(x), Float32(r)) :
                                  ulp_distance(x, r)
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

function compare_to_deck(measured::Vector{Float64}, refEnergies::Vector{Float32})
    n = min(length(refEnergies), length(measured))
    ref = Float64[Float64(refEnergies[i]) for i in 1:n]
    return compute_metrics(measured, ref, 64)
end

@inline function build_transforms(::Val{PPWI}, ::Type{T},
                                  t0, t1, t2, t3, t4, t5,
                                  ix::Int32, lsz::Int32) where {PPWI,T}
    return ntuple(Val(PPWI)) do i
        @inbounds begin
            idx = ix + Int32(i - 1) * lsz
            sx = sin(t0[idx]); cx = cos(t0[idx])
            sy = sin(t1[idx]); cy = cos(t1[idx])
            sz = sin(t2[idx]); cz = cos(t2[idx])
            (cy * cz,
             sx * sy * cz - cx * sz,
             cx * sy * cz + sx * sz,
             t3[idx],
             cy * sz,
             sx * sy * sz + cx * cz,
             cx * sy * sz - sx * cz,
             t4[idx],
             -sy,
             sx * cy,
             cx * cy,
             t5[idx])
        end
    end
end

@inline function ligand_positions(tf::NTuple{PPWI,NTuple{12,T}},
                                  lx::T, ly::T, lz::T) where {PPWI,T}
    return ntuple(Val(PPWI)) do i
        @inbounds tr = tf[i]
        @inbounds (tr[4] + lx * tr[1] + ly * tr[2] + lz * tr[3],
                   tr[8] + lx * tr[5] + ly * tr[6] + lz * tr[7],
                   tr[12] + lx * tr[9] + ly * tr[10] + lz * tr[11])
    end
end

@inline function accumulate_energies(etot::NTuple{PPWI,T},
                                     lpos::NTuple{PPWI,NTuple{3,T}},
                                     px::T, py::T, pz::T,
                                     radij::T, r_radij::T,
                                     elcdst::T, elcdst1::T,
                                     type_E::Bool, phphb_nz::Bool,
                                     distdslv::T, r_distdslv::T,
                                     chrg_init::T, dslv_init::T) where {PPWI,T}
    ZERO = zero(T)
    ONE = one(T)
    TWO = T(2)
    CNSTNT = T(45)
    HARDNESS = T(38)

    return ntuple(Val(PPWI)) do i
        @inbounds p = lpos[i]
        x = p[1] - px
        y = p[2] - py
        z = p[3] - pz
        distij = sqrt(x * x + y * y + z * z)

        distbb = distij - radij
        zone1 = distbb < ZERO

        e = (ONE - (distij * r_radij)) * (zone1 ? TWO * HARDNESS : ZERO)

        chrg_e = chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) *
                              (distbb < elcdst ? ONE : ZERO))
        neg_chrg_e = -abs(chrg_e)
        chrg_e = type_E ? neg_chrg_e : chrg_e
        e += chrg_e * CNSTNT

        coeff = ONE - (distbb * r_distdslv)
        dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? ONE : ZERO)
        dslv_e *= (zone1 ? ONE : coeff)
        e += dslv_e

        @inbounds etot[i] + e
    end
end

@inline function store_energies!(etotals, etot::NTuple{PPWI,T},
                                 base::Int32, lsz::Int32, half::T) where {PPWI,T}
    ntuple(Val(PPWI)) do i
        @inbounds etotals[base+Int32(i - 1)*lsz] = etot[i] * half
        nothing
    end
    return nothing
end

@kernel function fasten_main!(etotals,
                              @Const(protein), @Const(ligand),
                              @Const(t0), @Const(t1), @Const(t2),
                              @Const(t3), @Const(t4), @Const(t5),
                              @Const(forcefield),
                              natlig::Int32, natpro::Int32,
                              numTransforms::Int32, lsz::Int32,
                              ::Val{PPWI}) where {PPWI}

    @uniform T = eltype(etotals)

    ZERO = zero(T)
    ONE = one(T)
    TWO = T(2)
    FOUR = T(4)
    QUARTER = T(0.25)
    HALF = T(0.5)
    NPNPDIST = T(5.5)
    NPPDIST = T(1)
    MAXVAL = floatmax(T)

    lidx = Int32(@index(Local, Linear))
    gidx = Int32(@index(Group, Linear))

    base = (gidx - Int32(1)) * lsz * Int32(PPWI) + lidx

    ix = base <= numTransforms ? base : (numTransforms - Int32(PPWI) * lsz + lidx)

    tf = build_transforms(Val(PPWI), T, t0, t1, t2, t3, t4, t5, ix, lsz)
    etot = ntuple(_ -> ZERO, Val(PPWI))

    for il in Int32(1):natlig
        @inbounds l_atom = ligand[il]
        @inbounds l_params = forcefield[l_atom.type+Int32(1)]
        lhphb_ltz = l_params.hphb < ZERO
        lhphb_gtz = l_params.hphb > ZERO

        lpos = ligand_positions(tf, l_atom.x, l_atom.y, l_atom.z)

        for ip in Int32(1):natpro
            @inbounds p_atom = protein[ip]
            @inbounds p_params = forcefield[p_atom.type+Int32(1)]

            radij = p_params.radius + l_params.radius
            r_radij = ONE / radij

            both_f = (p_params.hbtype == HBTYPE_F) && (l_params.hbtype == HBTYPE_F)
            elcdst = both_f ? FOUR : TWO
            elcdst1 = both_f ? QUARTER : HALF
            type_E = (p_params.hbtype == HBTYPE_E) || (l_params.hbtype == HBTYPE_E)

            phphb_ltz = p_params.hphb < ZERO
            phphb_gtz = p_params.hphb > ZERO
            phphb_nz = p_params.hphb != ZERO
            p_hphb = p_params.hphb * ((phphb_ltz && lhphb_gtz) ? -ONE : ONE)
            l_hphb = l_params.hphb * ((phphb_gtz && lhphb_ltz) ? -ONE : ONE)
            distdslv = phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) :
                                   (lhphb_ltz ? NPPDIST : -MAXVAL)
            r_distdslv = ONE / distdslv

            chrg_init = l_params.elsc * p_params.elsc
            dslv_init = p_hphb + l_hphb

            etot = accumulate_energies(etot, lpos,
                                       p_atom.x, p_atom.y, p_atom.z,
                                       radij, r_radij, elcdst, elcdst1,
                                       type_E, phphb_nz,
                                       distdslv, r_distdslv,
                                       chrg_init, dslv_init)
        end
    end

    if base <= numTransforms
        store_energies!(etotals, etot, base, lsz, HALF)
    end
end

mutable struct Sample
    ppwi::Int
    wgsize::Int
    precisionBits::Int
    energies::Vector{Float64}
    kernelMillis::Vector{Float64}
    wallMillis::Vector{Float64}
    contextMillis::Float64
end

function fasten(::Type{T}, p::Params, wgsize::Int, ppwi::Int) where {T}
    backend = gpu_backend()

    hProtein = convert_atoms(T, p.protein)
    hLigand = convert_atoms(T, p.ligand)
    hForcefield = convert_forcefield(T, p.forcefield)

    n = nposes(p)

    contextStart = time_ns()
    dProtein = KernelAbstractions.allocate(backend, AtomT{T}, length(hProtein))
    dLigand = KernelAbstractions.allocate(backend, AtomT{T}, length(hLigand))
    dForcefield = KernelAbstractions.allocate(backend, FFParamsT{T}, length(hForcefield))
    copyto!(dProtein, hProtein)
    copyto!(dLigand, hLigand)
    copyto!(dForcefield, hForcefield)

    dTransforms = map(1:6) do i
        d = KernelAbstractions.allocate(backend, T, n)
        copyto!(d, T.(p.poses[i]))
        d
    end

    dResults = KernelAbstractions.allocate(backend, T, n)
    gpu_sync()
    contextEnd = time_ns()

    nblocks = cld(cld(n, ppwi), wgsize)
    ndrange = nblocks * wgsize

    kernel = fasten_main!(backend, wgsize)

    sample = Sample(ppwi, wgsize, T === Float32 ? 32 : 64, zeros(Float64, n),
                    Float64[], Float64[], (contextEnd - contextStart) * 1e-6)

    for _ in 1:total_iterations(p)
        wallStart = time_ns()
        secs = gpu_elapsed() do
            kernel(dResults, dProtein, dLigand,
                   dTransforms[1], dTransforms[2], dTransforms[3],
                   dTransforms[4], dTransforms[5], dTransforms[6],
                   dForcefield,
                   Int32(natlig(p)), Int32(natpro(p)),
                   Int32(n), Int32(wgsize), Val(ppwi);
                   ndrange = ndrange)
        end
        wallEnd = time_ns()

        push!(sample.kernelMillis, secs * 1e3)
        push!(sample.wallMillis, (wallEnd - wallStart) * 1e-6)
    end

    raw = Array(dResults)
    sample.energies .= Float64.(raw)

    gpu_sync()

    return sample
end

struct SummaryStats
    min::Float64
    max::Float64
    sum::Float64
    mean::Float64
    variance::Float64
    stdDev::Float64
end

function SummaryStats(ys::Vector{Float64})
    s = sum(ys)
    mean = s / length(ys)
    variance = sum((y - mean)^2 for y in ys) / length(ys)
    return SummaryStats(minimum(ys), maximum(ys), s, mean, variance, sqrt(variance))
end

struct Result
    sample::Sample
    ms::SummaryStats
    gflops::Float64
    ginsts::Float64
    interactionsPerSec::Float64
    launchOverheadMs::Float64
    vsReference::Metrics
    hasDifferential::Bool
    vsDeck::Metrics
end

function evaluate(p::Params, s::Sample, reference::Union{Nothing,Vector{Float64}})
    isempty(s.kernelMillis) &&
        error("Sample size is 0, this implementation did not measure kernel runtime!")

    msWithoutWarmup = s.kernelMillis[(p.warmupIterations+1):end]
    elapsed = sum(msWithoutWarmup)
    ms = elapsed / p.iterations
    runtime = ms * 1e-3

    ops_per_wg = s.ppwi * 27 +
                 natlig(p) * (2 + s.ppwi * 18 + natpro(p) * (10 + s.ppwi * 30)) +
                 s.ppwi
    total_ops = Float64(ops_per_wg) * (Float64(nposes(p)) / Float64(s.ppwi))
    gflops = (total_ops / runtime) / 1e9

    total_finsts = 25 * natpro(p) * natlig(p) * nposes(p)
    gfinsts = (Float64(total_finsts) / runtime) / 1e9

    interactions = nposes(p) * natlig(p) * natpro(p)
    interactionsPerSec = Float64(interactions) / runtime

    vsReference = Metrics()
    hasDifferential = reference !== nothing
    if hasDifferential
        vsReference = compute_metrics(s.energies, reference, s.precisionBits)
    end

    vsDeck = compare_to_deck(s.energies, p.refEnergies)

    launchOverheadMs = 0.0
    if length(s.wallMillis) == length(s.kernelMillis) &&
       length(s.kernelMillis) > p.warmupIterations
        acc = 0.0
        for i in (p.warmupIterations+1):length(s.kernelMillis)
            acc += s.wallMillis[i] - s.kernelMillis[i]
        end
        launchOverheadMs = acc / (length(s.kernelMillis) - p.warmupIterations)
    end

    return Result(s, SummaryStats(msWithoutWarmup), gflops, gfinsts,
                  interactionsPerSec, launchOverheadMs,
                  vsReference, hasDifferential, vsDeck)
end

function dump_results(p::Params, r::Result)
    isempty(p.output) && return nothing
    open(p.output, "w") do io
        for e in r.sample.energies
            @printf(io, "%.17g\n", e)
        end
    end
    return nothing
end

function show_human_readable(p::Params, r::Result, indent::Int = 1)
    prefix = " "^indent
    raw = join([@sprintf("%g", x) for x in r.sample.kernelMillis], ",")

    println(prefix, " - precision_bits:      ", r.sample.precisionBits)
    println(prefix, "   param:               { ppwi: ", r.sample.ppwi,
            ", wgsize: ", r.sample.wgsize, " }")
    println(prefix, "   raw_iterations:      [", raw, "]")
    @printf("%s   context_ms:          %.6f\n", prefix, r.sample.contextMillis)
    @printf("%s   sum_ms:              %.3f\n", prefix, r.ms.sum)
    @printf("%s   avg_ms:              %.3f\n", prefix, r.ms.mean)
    @printf("%s   min_ms:              %.3f\n", prefix, r.ms.min)
    @printf("%s   max_ms:              %.3f\n", prefix, r.ms.max)
    @printf("%s   stddev_ms:           %.3f\n", prefix, r.ms.stdDev)
    @printf("%s   giga_interactions/s: %.3f\n", prefix, r.interactionsPerSec / 1e9)
    @printf("%s   gflop/s:             %.3f\n", prefix, r.gflops)
    @printf("%s   gfinst/s:            %.3f\n", prefix, r.ginsts)
    @printf("%s   launch_overhead_ms:  %.3f\n", prefix, r.launchOverheadMs)
    println(prefix, "   energies:            ")
    for i in 1:min(p.outRows, length(r.sample.energies))
        @printf("%s     - %.6f\n", prefix, r.sample.energies[i])
    end

    if r.hasDifferential
        m = r.vsReference
        println(prefix, "   vs_fp64_reference:")
        @printf("%s     l2_rel_norm:       %.6e\n", prefix, m.l2RelNorm)
        @printf("%s     rms_abs:           %.6e\n", prefix, m.rmsAbs)
        @printf("%s     mean_signed:       %.6e\n", prefix, m.meanSigned)
        @printf("%s     max_abs:           { value: %.6e, index: %d }\n",
                prefix, m.maxAbs, m.maxAbsIdx)
        @printf("%s     max_rel:           { value: %.6e, index: %d }\n",
                prefix, m.maxRel, m.maxRelIdx)
        @printf("%s     max_ulp:           { value: %d, index: %d }\n",
                prefix, m.maxUlp, m.maxUlpIdx)
        println(prefix, "     compared:          ", m.compared)
        println(prefix, "     non_finite:        ", m.nonFinite)
        println(prefix, "     ulp_histogram:")
        for i in 1:ULP_NBUCKETS
            if m.ulpHistogram[i] > 0
                println(prefix, "       \"", ulp_bucket_label(i), "\": ", m.ulpHistogram[i])
            end
        end
    else
        println(prefix, "   vs_fp64_reference:   ~ # benchmarked precision is the reference")
    end

    m = r.vsDeck
    println(prefix, "   vs_deck_reference:   # informational; ref_energies.out is low precision")
    @printf("%s     l2_rel_norm:       %.6e\n", prefix, m.l2RelNorm)
    @printf("%s     max_rel:           { value: %.6e, index: %d }\n",
            prefix, m.maxRel, m.maxRelIdx)
    println(prefix, "     beyond_tolerance:  ", m.beyondTolerance, "/", m.compared,
            " # tolerance = ", REPORT_TOLERANCE_PCT, "%")
    flush(stdout)
    return nothing
end

function show_csv(p::Params, r::Result, header::Bool)
    if header
        println("precision_bits,ppwi,wgsize,sum_ms,avg_ms,min_ms,max_ms,stddev_ms," *
                "interactions/s,gflops/s,gfinst/s,launch_overhead_ms," *
                "l2_rel_norm,rms_abs,mean_signed,max_abs,max_rel,max_ulp,deck_l2_rel_norm")
    end
    @printf("%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
            r.sample.precisionBits, r.sample.ppwi, r.sample.wgsize,
            r.ms.sum, r.ms.mean, r.ms.min, r.ms.max, r.ms.stdDev,
            r.interactionsPerSec, r.gflops, r.ginsts, r.launchOverheadMs)
    if r.hasDifferential
        m = r.vsReference
        @printf(",%.6e,%.6e,%.6e,%.6e,%.6e,%d",
                m.l2RelNorm, m.rmsAbs, m.meanSigned, m.maxAbs, m.maxRel, m.maxUlp)
    else
        print(",,,,,")
    end
    @printf(",%.6e\n", r.vsDeck.l2RelNorm)
    flush(stdout)
    return nothing
end

function print_help()
    println()
    println("Usage: julia bude_ka.jl [COMMAND|OPTIONS]\n")
    println("Commands:")
    println("  help -h --help       Print this message")
    println("  list -l --list       List available devices")
    println("Options:")
    println("  -d --device  INDEX   Select device at INDEX from output of --list, performs a substring match of device names if INDEX is not an integer")
    println("                       [optional] default=0")
    println("  -i --iter    I       Repeat kernel I times")
    println("                       [optional] default=$DEFAULT_ITERS")
    println("  -n --poses   N       Compute energies for only N poses, use 0 for deck max")
    println("                       [optional] default=0 ")
    println("  -p --ppwi    PPWI    A CSV list of poses per work-item for the kernel, use `all` for everything")
    println("                       [optional] default=$(PPWIS[1]); available=$(join(PPWIS, ","))")
    println("  -w --wgsize  WGSIZE  A CSV list of work-group sizes, not all implementations support this parameter")
    println("                       [optional] default=1")
    println("     --deck    DIR     Use the DIR directory as input deck")
    println("                       [optional] default=`$DEFAULT_DATA_DIR`")
    println("  -o --out     PATH    Save resulting energies to PATH (no-op if more than one PPWI/WGSIZE specified)")
    println("                       [optional]")
    println("  -r --rows    N       Output first N row(s) of energy values as part of the on-screen result")
    println("                       [optional] default=$DEFAULT_ENERGY_ENTRIES")
    println("     --csv             Output results in CSV format")
    println("                       [optional] default=false")
    println()
    println("Precision is not a flag: edit the PRECISION_BITS constant at the top of")
    println("this file, as with the reference's -DPRECISION_BITS.")
    println()
end

function parse_params(args::Vector{String})
    p = Params()
    wgsizes = Int[]
    ppwis = Int[]
    nposesArg = 0
    precisionBits = PRECISION_BITS

    i = 1
    function value!(flag)
        if i + 1 > length(args)
            println(stderr, "[$flag] specified but no value was given")
            exit(1)
        end
        i += 1
        return args[i]
    end

    parseints(s, name) = [begin
                              v = tryparse(Int, t)
                              if v === nothing || v < 0
                                  println(stderr, "malformed value, positive integer required for <$name>: `$t`")
                                  exit(1)
                              end
                              v
                          end for t in split(s, ',')]

    while i <= length(args)
        a = args[i]
        if a in ("--iter", "-i")
            p.iterations = parseints(value!(a), "iter")[1]
        elseif a in ("--poses", "-n")
            nposesArg = parseints(value!(a), "poses")[1]
        elseif a in ("--device", "-d")
            p.deviceSelector = value!(a)
        elseif a == "--deck"
            p.deckDir = value!(a)
        elseif a in ("--out", "-o")
            p.output = value!(a)
        elseif a in ("--rows", "-r")
            p.outRows = parseints(value!(a), "rows")[1]
        elseif a in ("--wgsize", "-w")
            wgsizes = parseints(value!(a), "wgsize")
        elseif a in ("--ppwi", "-p")
            s = value!(a)
            if s == "all"
                ppwis = collect(PPWIS)
            else
                ppwis = parseints(s, "ppwi")
                for v in ppwis
                    if !(v in PPWIS)
                        println(stderr, "PPWI $v is not a supported value, should be one of `$(join(PPWIS, ","))`")
                        exit(1)
                    end
                end
            end
        elseif a == "--csv"
            p.csv = true
        elseif a in ("list", "--list", "-l")
            p.list = true
        elseif a in ("help", "--help", "-h")
            print_help()
            exit(0)
        else
            println("Unrecognized argument '$a' (try '--help')")
            exit(1)
        end
        i += 1
    end

    isempty(ppwis) && (ppwis = [PPWIS[1]])
    isempty(wgsizes) && (wgsizes = [1])
    precisionBits in (32, 64) || error("precision must be 32 or 64, got $precisionBits")

    p.list && return (p, wgsizes, ppwis, precisionBits)

    p.ligand = read_nstruct(DiskAtom, joinpath(p.deckDir, "ligand.in"))
    p.protein = read_nstruct(DiskAtom, joinpath(p.deckDir, "protein.in"))
    p.forcefield = read_nstruct(DiskFFParams, joinpath(p.deckDir, "forcefield.in"))

    poses = read_nstruct(Float32, joinpath(p.deckDir, "poses.in"))
    length(poses) % 6 == 0 ||
        throw(ArgumentError("Pose size ($(length(poses))) not divisible my 6!"))
    maxPoses = length(poses) ÷ 6
    n = nposesArg == 0 ? maxPoses : nposesArg
    n <= maxPoses ||
        throw(ArgumentError("Requested poses ($n) exceeded max poses ($maxPoses) for deck"))
    for k in 1:6
        p.poses[k] = poses[((k-1)*maxPoses+1):((k-1)*maxPoses+n)]
    end

    for line in eachline(joinpath(p.deckDir, "ref_energies.out"))
        isempty(strip(line)) && continue
        push!(p.refEnergies, parse(Float32, strip(line)))
    end
    nposes(p) <= length(p.refEnergies) ||
        throw(ArgumentError("Size of reference energies ($(length(p.refEnergies))) is less than poses ($(nposes(p)))"))

    p.maxPoses = maxPoses
    return (p, wgsizes, ppwis, precisionBits)
end

function select_device(needle::String, haystack::Vector{Tuple{Int,String}})
    isempty(needle) && return haystack[1]
    idx = tryparse(Int, needle)
    if idx !== nothing
        1 <= idx + 1 <= length(haystack) && return haystack[idx+1]
    end
    println(stderr, "# Attempting to match device with substring  `$needle`")
    for d in haystack
        occursin(needle, d[2]) && return d
    end
    if length(haystack) == 1
        println(stderr, "# No matching device but there's only one device, using it anyway")
        return haystack[1]
    end
    println(stderr, "# No matching devices")
    return (-1, "")
end

function run(p::Params, wgsizes::Vector{Int}, ppwis::Vector{Int}, precisionBits::Int)
    devices = gpu_devices()
    isempty(devices) && println(stderr, " # (no devices available)")

    if p.list
        println(p.csv ? "index,name" : "devices:")
        for (i, d) in enumerate(devices)
            p.csv ? println(i - 1, ",", d[2]) : println("  ", i - 1, ": \"", d[2], "\"")
        end
        return true
    end

    dev = select_device(p.deviceSelector, devices)
    dev[1] < 0 && return false
    gpu_select_device!(dev[1])

    p.csv || println("device: { index: ", dev[1], ",  name: \"", dev[2], "\" }")

    FP = precisionBits == 32 ? Float32 : Float64

    wantDifferential = precisionBits != 64
    golden = Float64[]
    goldenReady = false

    dumped = false
    results = Result[]
    for ppwi in ppwis, wgsize in wgsizes
        n = nposes(p)
        if n < ppwi * wgsize
            println(" # WARNING: pose count $n <= ($wgsize (wgsize) * $ppwi (ppwi)), skipping")
            continue
        end
        if n % (ppwi * wgsize) != 0
            println(" # WARNING: pose count $n % ($wgsize (wgsize) * $ppwi (ppwi)) != 0, skipping")
            continue
        end

        if wantDifferential && !goldenReady
            println("# computing fp64 reference pass (ppwi=$ppwi, wgsize=$wgsize)")
            golden = fasten(Float64, p, wgsize, ppwi).energies
            goldenReady = true
        end

        sample = fasten(FP, p, wgsize, ppwi)
        result = evaluate(p, sample, goldenReady ? golden : nothing)
        push!(results, result)

        if !dumped
            dumped = true
            dump_results(p, result)
        end
    end

    if isempty(results)
        println(stderr, "# no runnable configurations")
        return false
    end

    p.csv || println("results:")
    for (i, r) in enumerate(results)
        p.csv ? show_csv(p, r, i == 1) : show_human_readable(p, r)
    end

    best = results[argmin([r.ms.sum for r in results])]
    @printf("%sbest: { min_ms: %.3f, max_ms: %.3f, sum_ms: %.3f, avg_ms: %.3f, ppwi: %d, wgsize: %d }\n",
            p.csv ? "# " : "", best.ms.min, best.ms.max, best.ms.sum, best.ms.mean,
            best.sample.ppwi, best.sample.wgsize)
    return true
end

function julia_main()
    args = String[a for a in ARGS]
    (p, wgsizes, ppwis, precisionBits) = parse_params(args)
    if !p.csv
        println("miniBUDE:  kernelabstractions")
        println("backend:   ", gpu_label())
        println("precision: ", precisionBits, " # PRECISION_BITS constant in source")
        println("ppwi:      [", join(ppwis, ","), "]")
        println("wgsize:    [", join(wgsizes, ","), "]")
    end
    return run(p, wgsizes, ppwis, precisionBits) ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(julia_main())
end
