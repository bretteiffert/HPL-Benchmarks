//! CubeCL miniBUDE kernel operating on structure-of-arrays inputs with compile-time poses per work item.
//! It evaluates the reference interaction terms for each transform and uses CubeCL select operations for conditional expressions.
//! Tail units are remapped for safe indexing and only in-range results are written.

use cubecl::prelude::*;

pub const HBTYPE_F: u32 = 70;
pub const HBTYPE_E: u32 = 69;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn fasten_main<F: Float>(
    etotals: &mut Array<F>,
    prot_x: &Array<F>,
    prot_y: &Array<F>,
    prot_z: &Array<F>,
    prot_t: &Array<u32>,
    lig_x: &Array<F>,
    lig_y: &Array<F>,
    lig_z: &Array<F>,
    lig_t: &Array<u32>,
    t0: &Array<F>,
    t1: &Array<F>,
    t2: &Array<F>,
    t3: &Array<F>,
    t4: &Array<F>,
    t5: &Array<F>,
    ff_hbtype: &Array<u32>,
    ff_radius: &Array<F>,
    ff_hphb: &Array<F>,
    ff_elsc: &Array<F>,
    natlig: u32,
    natpro: u32,
    num_transforms: u32,
    #[comptime] ppwi: usize,
) {
    let zero = F::new(0.0f32);
    let quarter = F::new(0.25f32);
    let half = F::new(0.5f32);
    let one = F::new(1.0f32);
    let two = F::new(2.0f32);
    let four = F::new(4.0f32);
    let cnstnt = F::new(45.0f32);
    let hardness = F::new(38.0f32);
    let npnpdist = F::new(5.5f32);
    let nppdist = F::new(1.0f32);

    let maxval = F::new(f32::MAX);

    let lsz = usize::cast_from(CUBE_DIM_X);
    let unit = usize::cast_from(UNIT_POS_X);
    let cube = usize::cast_from(CUBE_POS_X);
    let ntrans = usize::cast_from(num_transforms);

    let base = cube * lsz * ppwi + unit;

    let ix = select(base < ntrans, base, ntrans - ppwi * lsz + unit);

    let mut etot = Array::<F>::new(ppwi);
    let mut tf = Array::<F>::new(comptime!(ppwi * 12));

    #[unroll]
    for i in 0..ppwi {
        let index = ix + i * lsz;

        let sx = F::sin(t0[index]);
        let cx = F::cos(t0[index]);
        let sy = F::sin(t1[index]);
        let cy = F::cos(t1[index]);
        let sz = F::sin(t2[index]);
        let cz = F::cos(t2[index]);

        let b = i * 12;
        tf[b] = cy * cz;
        tf[b + 1] = sx * sy * cz - cx * sz;
        tf[b + 2] = cx * sy * cz + sx * sz;
        tf[b + 3] = t3[index];
        tf[b + 4] = cy * sz;
        tf[b + 5] = sx * sy * sz + cx * cz;
        tf[b + 6] = cx * sy * sz - sx * cz;
        tf[b + 7] = t4[index];
        tf[b + 8] = zero - sy;
        tf[b + 9] = sx * cy;
        tf[b + 10] = cx * cy;
        tf[b + 11] = t5[index];

        etot[i] = zero;
    }

    let mut lpos = Array::<F>::new(comptime!(ppwi * 3));

    for il in 0..usize::cast_from(natlig) {
        let lx = lig_x[il];
        let ly = lig_y[il];
        let lz = lig_z[il];
        let ltype = usize::cast_from(lig_t[il]);

        let l_radius = ff_radius[ltype];
        let l_hphb = ff_hphb[ltype];
        let l_elsc = ff_elsc[ltype];
        let l_hbtype = ff_hbtype[ltype];

        let lhphb_ltz = l_hphb < zero;
        let lhphb_gtz = l_hphb > zero;

        #[unroll]
        for i in 0..ppwi {
            let b = i * 12;
            let o = i * 3;
            lpos[o] = tf[b + 3] + lx * tf[b] + ly * tf[b + 1] + lz * tf[b + 2];
            lpos[o + 1] = tf[b + 7] + lx * tf[b + 4] + ly * tf[b + 5] + lz * tf[b + 6];
            lpos[o + 2] = tf[b + 11] + lx * tf[b + 8] + ly * tf[b + 9] + lz * tf[b + 10];
        }

        for ip in 0..usize::cast_from(natpro) {
            let px = prot_x[ip];
            let py = prot_y[ip];
            let pz = prot_z[ip];
            let ptype = usize::cast_from(prot_t[ip]);

            let p_radius = ff_radius[ptype];
            let p_hphb = ff_hphb[ptype];
            let p_elsc = ff_elsc[ptype];
            let p_hbtype = ff_hbtype[ptype];

            let radij = p_radius + l_radius;
            let r_radij = one / radij;

            let both_f = (p_hbtype == HBTYPE_F) && (l_hbtype == HBTYPE_F);
            let elcdst = select(both_f, four, two);
            let elcdst1 = select(both_f, quarter, half);
            let type_e = (p_hbtype == HBTYPE_E) || (l_hbtype == HBTYPE_E);

            let phphb_ltz = p_hphb < zero;
            let phphb_gtz = p_hphb > zero;
            let phphb_nz = p_hphb != zero;

            let p_hphb_s = p_hphb * select(phphb_ltz && lhphb_gtz, zero - one, one);
            let l_hphb_s = l_hphb * select(phphb_gtz && lhphb_ltz, zero - one, one);

            let distdslv = select(
                phphb_ltz,
                select(lhphb_ltz, npnpdist, nppdist),
                select(lhphb_ltz, nppdist, zero - maxval),
            );
            let r_distdslv = one / distdslv;

            let chrg_init = l_elsc * p_elsc;
            let dslv_init = p_hphb_s + l_hphb_s;

            #[unroll]
            for i in 0..ppwi {
                let o = i * 3;
                let x = lpos[o] - px;
                let y = lpos[o + 1] - py;
                let z = lpos[o + 2] - pz;
                let distij = F::sqrt(x * x + y * y + z * z);

                let distbb = distij - radij;
                let zone1 = distbb < zero;

                etot[i] += (one - (distij * r_radij)) * select(zone1, two * hardness, zero);

                let mut chrg_e = chrg_init
                    * (select(zone1, one, one - distbb * elcdst1)
                        * select(distbb < elcdst, one, zero));
                let neg_chrg_e = zero - F::abs(chrg_e);
                chrg_e = select(type_e, neg_chrg_e, chrg_e);
                etot[i] += chrg_e * cnstnt;

                let coeff = one - (distbb * r_distdslv);
                let mut dslv_e = dslv_init * select(distbb < distdslv && phphb_nz, one, zero);
                dslv_e *= select(zone1, one, coeff);
                etot[i] += dslv_e;
            }
        }
    }

    if base < ntrans {
        #[unroll]
        for i in 0..ppwi {
            etotals[base + i * lsz] = etot[i] * half;
        }
    }
}
