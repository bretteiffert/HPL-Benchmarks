# Implements the Hartree-Fock host pipeline: input parsing, Schwarz screening data, an fp64 CPU reference, argument handling, and correctness reporting. The benchmark arithmetic defaults to Float64 through `HF_PRECISION = 64`, while `LEGACY_SQRTF = false` selects native square roots. Inputs are parsed in Float64 before benchmark arrays are converted to the selected arithmetic type.
const HF_PRECISION = 64
const RealT = HF_PRECISION == 32 ? Float32 :
              HF_PRECISION == 64 ? Float64 :
              error("HF_PRECISION must be 32 or 64")
const REAL_NAME = HF_PRECISION == 32 ? "fp32" : "fp64"

const LEGACY_SQRTF = false
const SQRT_MODE = LEGACY_SQRTF ? "legacy-sqrtf" : "native"

const SQRTPI2_HOST = RealT(3.1415926535897931^-0.5 * 2.0)
const SQRTPI2_DEV = RealT(1.12837916709551255856)
const DTOL = RealT(1.0e-12)
const RCUT = RealT(1.0e-12)
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

function parse_args(args)
    o = Opts(false, true, false, 128, 10)
    for s in args[2:end]
        if s == "--csv"
            o.csv_output = true
        elseif s == "--check"
            o.run_check = true; o.force_check = true
        elseif s == "--no-check"
            o.run_check = false
        elseif startswith(s, "--max-check-atoms=")
            o.max_check_atoms = parse(Int, s[19:end])
        elseif startswith(s, "--iters=")
            o.num_iter = parse(Int, s[9:end])
        end
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
