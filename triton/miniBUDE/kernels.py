# miniBUDE device code (Triton). The kernel is unchanged from the original
# miniBUDE.py; BLOCK = ppwi * wgsize, num_warps = wgsize / warp size.

import triton
import triton.language as tl

HBTYPE_F = tl.constexpr(70)
HBTYPE_E = tl.constexpr(69)


@triton.jit
def fasten_main(
    etotals_ptr,
    prot_x, prot_y, prot_z, prot_t,
    lig_x, lig_y, lig_z, lig_t,
    t0, t1, t2, t3, t4, t5,
    ff_hbtype, ff_radius, ff_hphb, ff_elsc,
    natlig, natpro, num_transforms,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
    MAXVAL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_transforms

    a0 = tl.load(t0 + offs, mask=mask, other=0.0)
    a1 = tl.load(t1 + offs, mask=mask, other=0.0)
    a2 = tl.load(t2 + offs, mask=mask, other=0.0)

    sx = tl.sin(a0)
    cx = tl.cos(a0)
    sy = tl.sin(a1)
    cy = tl.cos(a1)
    sz = tl.sin(a2)
    cz = tl.cos(a2)

    m00 = cy * cz
    m01 = sx * sy * cz - cx * sz
    m02 = cx * sy * cz + sx * sz
    m03 = tl.load(t3 + offs, mask=mask, other=0.0)
    m10 = cy * sz
    m11 = sx * sy * sz + cx * cz
    m12 = cx * sy * sz - sx * cz
    m13 = tl.load(t4 + offs, mask=mask, other=0.0)
    m20 = -sy
    m21 = sx * cy
    m22 = cx * cy
    m23 = tl.load(t5 + offs, mask=mask, other=0.0)

    etot = tl.zeros([BLOCK], dtype=DTYPE)

    for il in range(natlig):
        lx = tl.load(lig_x + il)
        ly = tl.load(lig_y + il)
        lz = tl.load(lig_z + il)
        ltype = tl.load(lig_t + il)

        l_radius = tl.load(ff_radius + ltype)
        l_hphb = tl.load(ff_hphb + ltype)
        l_elsc = tl.load(ff_elsc + ltype)
        l_hbtype = tl.load(ff_hbtype + ltype)

        lhphb_ltz = l_hphb < 0.0
        lhphb_gtz = l_hphb > 0.0

        lpos_x = m03 + lx * m00 + ly * m01 + lz * m02
        lpos_y = m13 + lx * m10 + ly * m11 + lz * m12
        lpos_z = m23 + lx * m20 + ly * m21 + lz * m22

        for ip in range(natpro):
            px = tl.load(prot_x + ip)
            py = tl.load(prot_y + ip)
            pz = tl.load(prot_z + ip)
            ptype = tl.load(prot_t + ip)

            p_radius = tl.load(ff_radius + ptype)
            p_hphb = tl.load(ff_hphb + ptype)
            p_elsc = tl.load(ff_elsc + ptype)
            p_hbtype = tl.load(ff_hbtype + ptype)

            radij = p_radius + l_radius
            r_radij = 1.0 / radij

            both_f = (p_hbtype == HBTYPE_F) & (l_hbtype == HBTYPE_F)
            elcdst = tl.where(both_f, 4.0, 2.0).to(DTYPE)
            elcdst1 = tl.where(both_f, 0.25, 0.5).to(DTYPE)
            type_E = (p_hbtype == HBTYPE_E) | (l_hbtype == HBTYPE_E)

            phphb_ltz = p_hphb < 0.0
            phphb_gtz = p_hphb > 0.0
            phphb_nz = p_hphb != 0.0
            p_hphb_s = p_hphb * tl.where(phphb_ltz & lhphb_gtz, -1.0, 1.0).to(DTYPE)
            l_hphb_s = l_hphb * tl.where(phphb_gtz & lhphb_ltz, -1.0, 1.0).to(DTYPE)

            distdslv = tl.where(
                phphb_ltz,
                tl.where(lhphb_ltz, 5.5, 1.0),
                tl.where(lhphb_ltz, 1.0, -MAXVAL),
            ).to(DTYPE)
            r_distdslv = 1.0 / distdslv

            chrg_init = l_elsc * p_elsc
            dslv_init = p_hphb_s + l_hphb_s

            x = lpos_x - px
            y = lpos_y - py
            z = lpos_z - pz
            distij = tl.sqrt(x * x + y * y + z * z)

            distbb = distij - radij
            zone1 = distbb < 0.0

            etot += (1.0 - (distij * r_radij)) * tl.where(zone1, 2.0 * 38.0, 0.0)

            chrg_e = chrg_init * (
                tl.where(zone1, 1.0, 1.0 - distbb * elcdst1)
                * tl.where(distbb < elcdst, 1.0, 0.0)
            )
            neg_chrg_e = -tl.abs(chrg_e)
            chrg_e = tl.where(type_E, neg_chrg_e, chrg_e)
            etot += chrg_e * 45.0

            coeff = 1.0 - (distbb * r_distdslv)
            dslv_e = dslv_init * tl.where((distbb < distdslv) & phphb_nz, 1.0, 0.0)
            dslv_e *= tl.where(zone1, 1.0, coeff)
            etot += dslv_e

    tl.store(etotals_ptr + offs, etot * 0.5, mask=mask)
