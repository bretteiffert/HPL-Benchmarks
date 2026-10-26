# hf.jl — Hartree-Fock input parsing, host setup (Schwarz screening), device
# buffers, timed launches and the fp64 CPU reference. No printing here.
#
# Host math is unchanged from the original hartree-fock.jl; the precision-generic
# constants come from Kernels (same values as the old RealT constants).

module HF

using KernelAbstractions
using ..Kernels
using ..Backend: elapsed_seconds, device_sync

export InputError, read_input, host_setup, HFDevice, launch_fock!, timed_fock!, read_fock,
       contract_erep, fock_build_cpu, energy_from_fock, energy_symmetrized

struct InputError <: Exception
    msg::String
end
Base.showerror(io::IO, e::InputError) = print(io, e.msg)

"Parse an input file: ngauss, natoms, (xpnt, coef) × ngauss, (x, y, z) × natoms."
function read_input(filename::AbstractString)
    isfile(filename) || throw(InputError("input file '$filename' not found"))
    text = try
        read(filename, String)
    catch e
        throw(InputError("cannot read input file '$filename': $(sprint(showerror, e))"))
    end
    tok = split(text)
    p = Ref(1)
    function next_num(::Type{N}, what) where {N}
        p[] <= length(tok) ||
            throw(InputError("input file '$filename' ends early (expected $what)"))
        v = tryparse(N, tok[p[]])
        v === nothing &&
            throw(InputError("input file '$filename': '$(tok[p[]])' is not a valid $what"))
        p[] += 1
        return v
    end
    ngauss = next_num(Int, "Gaussian count")
    natoms = next_num(Int, "atom count")
    ngauss >= 1 || throw(InputError("input file '$filename': Gaussian count must be >= 1"))
    natoms >= 1 || throw(InputError("input file '$filename': atom count must be >= 1"))
    xpnt = Vector{Float64}(undef, ngauss)
    coef = Vector{Float64}(undef, ngauss)
    for i in 1:ngauss
        xpnt[i] = next_num(Float64, "exponent")
        coef[i] = next_num(Float64, "coefficient")
    end
    geom = Vector{Float64}(undef, 3 * natoms)
    for i in 0:(natoms - 1), j in 0:2
        geom[j * natoms + i + 1] = next_num(Float64, "coordinate")
    end
    return ngauss, natoms, xpnt, coef, geom
end

function ssss(i::Int, j::Int, k::Int, l::Int, ngauss::Int,
              xpnt::Vector{T}, coef::Vector{T}, geom::Vector{T},
              natoms::Int, legacy) where {T}
    DTOL = hf_dtol(T)
    RCUT = hf_rcut(T)
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
                    f0t = hf_sqrtpi2(T)
                    if tt > RCUT
                        f0t = tt^pm05 * host_erf(sqrt_boys(tt, legacy))
                    end
                    eri += dkl * f0t * sqrt_boys(aijkl, legacy)
                end
            end
        end
    end
    return eri
end

function build_schwarz(ngauss::Int, natoms::Int, xpnt::Vector{T}, coef::Vector{T},
                       geom::Vector{T}, legacy) where {T}
    nn = div(natoms * natoms + natoms, 2)
    schwarz = zeros(T, nn + 1)
    ij = 0
    for i in 0:(natoms - 1), j in 0:i
        ij += 1
        schwarz[ij + 1] = sqrt(abs(ssss(i, j, i, j, ngauss, xpnt, coef, geom, natoms, legacy)))
    end
    return schwarz, nn
end

"Host data in T (device inputs) and Float64 (CPU reference)."
function host_setup(::Type{T}, filename::AbstractString, legacy) where {T}
    ngauss, natoms, xpnt_d, coef_d, geom_d = read_input(filename)
    h_xpnt, h_coef, h_geom = T.(xpnt_d), T.(coef_d), T.(geom_d)
    nsq = natoms * natoms

    h_dens = fill(T(0.1), nsq)
    dens_d = fill(0.1, nsq)
    for i in 0:(natoms - 1)
        h_dens[i * natoms + i + 1] = T(1.0)
        dens_d[i * natoms + i + 1] = 1.0
    end

    for i in 1:ngauss
        h_coef[i] = h_coef[i] * (T(2.0) * h_xpnt[i])^T(0.75)
        coef_d[i] = coef_d[i] * (2.0 * xpnt_d[i])^0.75
    end
    tob = hf_tobohrs(T)
    for i in 0:(natoms - 1)
        h_geom[0natoms + i + 1] *= tob
        h_geom[1natoms + i + 1] *= tob
        h_geom[2natoms + i + 1] *= tob
        geom_d[0natoms + i + 1] *= 1.889725987722
        geom_d[1natoms + i + 1] *= 1.889725987722
        geom_d[2natoms + i + 1] *= 1.889725987722
    end

    h_schwarz, nn = build_schwarz(ngauss, natoms, h_xpnt, h_coef, h_geom, legacy)
    nnnn = div(nn * nn + nn, 2)

    return (; ngauss, natoms, nsq, nn, nnnn,
            h_xpnt, h_coef, h_geom, h_dens, h_schwarz,
            xpnt_d, coef_d, geom_d, dens_d)
end

# ------------------------------------------------------------------ device

struct HFDevice{T,B,A<:AbstractVector{T},K,L}
    backend::B
    fock::A
    geom::A
    xpnt::A
    coef::A
    dens::A
    schwarz::A
    kern::K
    nnnn::Int
    ngauss::Int
    natoms::Int
    legacy::L
end

function _to_device(backend, h::Vector{T}) where {T}
    d = KernelAbstractions.allocate(backend, T, length(h))
    copyto!(d, h)
    return d
end

function HFDevice(backend, S, block_size::Int, legacy)
    T = eltype(S.h_xpnt)
    fock = KernelAbstractions.zeros(backend, T, S.nsq)
    geom = _to_device(backend, S.h_geom)
    xpnt = _to_device(backend, S.h_xpnt)
    coef = _to_device(backend, S.h_coef)
    dens = _to_device(backend, S.h_dens)
    schwarz = _to_device(backend, S.h_schwarz)
    KernelAbstractions.synchronize(backend)
    kern = hf_kernel!(backend, block_size)
    return HFDevice{T,typeof(backend),typeof(fock),typeof(kern),typeof(legacy)}(
        backend, fock, geom, xpnt, coef, dens, schwarz, kern,
        S.nnnn, S.ngauss, S.natoms, legacy)
end

_launch(d::HFDevice) = d.kern(d.fock, d.nnnn, d.ngauss, d.natoms, d.geom, d.xpnt, d.coef,
                             d.dens, d.schwarz, d.legacy; ndrange = d.nnnn)

"Untimed run: zero the Fock matrix, build it, wait."
function launch_fock!(d::HFDevice{T}) where {T}
    fill!(d.fock, zero(T))
    device_sync()
    _launch(d)
    device_sync()
    return nothing
end

"Timed run: zero the Fock matrix (untimed), then time the build (seconds)."
function timed_fock!(d::HFDevice{T}) where {T}
    fill!(d.fock, zero(T))
    device_sync()
    return elapsed_seconds(() -> _launch(d))
end

read_fock(d::HFDevice) = Array(d.fock)

function contract_erep(h_fock::Vector{T}, h_dens::Vector{T}, natoms::Int) where {T}
    erep = zero(T)
    @inbounds for i in 0:(natoms - 1), j in 0:(natoms - 1)
        erep += h_fock[i * natoms + j + 1] * h_dens[i * natoms + j + 1]
    end
    return erep
end

# ------------------------------------------------------------------ fp64 CPU reference

const REF_SQRTPI2 = 1.12837916709551255856
const REF_DTOL = 1.0e-12
const REF_RCUT = 1.0e-12

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

end # module HF
