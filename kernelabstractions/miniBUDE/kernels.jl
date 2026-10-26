# kernels.jl — miniBUDE device code (KernelAbstractions). Unchanged from the
# original miniBUDE.jl: the kernel was already generic in T (eltype(etotals))
# and in PPWI (Val), so run-time precision needed no kernel change.

module Kernels

using KernelAbstractions

export AtomT, FFParamsT, fasten_main!, HBTYPE_F, HBTYPE_E

const HBTYPE_F = Int32(70)
const HBTYPE_E = Int32(69)

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

end # module Kernels
