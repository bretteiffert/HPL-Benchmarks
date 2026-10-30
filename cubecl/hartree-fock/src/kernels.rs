//! Two-electron Fock-build kernel: one unit per (ij|kl) integral quartet.
//!
//! CubeCL 0.10 kernels use `Array` launch arguments.

use cubecl::prelude::*;

#[cube]
fn idx_sqrt(x: u64) -> u64 {
    let widened: f32 = f32::cast_from(x);
    let rooted: f32 = f32::sqrt(widened);
    let out: u64 = u64::cast_from(rooted);
    out
}

#[cube(launch_unchecked)]
pub fn hartree_fock_kernel<F: Float + ScalarArgSettings>(
    geom: &Array<F>,
    xpnt: &Array<F>,
    coef: &Array<F>,
    dens: &Array<F>,
    schwarz: &Array<F>,
    fock: &Array<Atomic<F>>,
    nnnn: u64,
    ngauss: u32,
    natoms: u32,
    dtol: F,
    rcut: F,
    sqrtpi2: F,
    #[comptime] legacy: bool,
) {
    let ijkl: u64 = u64::cast_from(ABSOLUTE_POS);

    if ijkl >= 1u64 && ijkl <= nnnn {
        let seed_ij: u64 = 2u64 * ijkl;
        let mut ij = idx_sqrt(seed_ij);
        let mut n = (ij * ij + ij) / 2u64;
        while n < ijkl {
            ij += 1u64;
            n = (ij * ij + ij) / 2u64;
        }
        let kl: u64 = ijkl - (ij * ij - ij) / 2u64;

        let si: usize = usize::cast_from(ij);
        let sk: usize = usize::cast_from(kl);
        let s_ij: F = schwarz[si];
        let s_kl: F = schwarz[sk];
        let s_prod: F = s_ij * s_kl;

        if s_prod > dtol {
            let seed_i: u64 = 2u64 * ij;
            let root_i: u64 = idx_sqrt(seed_i);
            let mut i = root_i - 1u64;
            let mut ni = (i * i + i) / 2u64;
            while ni < ij {
                i += 1u64;
                ni = (i * i + i) / 2u64;
            }
            let mut j = ij - (i * i - i) / 2u64;

            let seed_k: u64 = 2u64 * kl;
            let mut k = idx_sqrt(seed_k);
            let mut nk = (k * k + k) / 2u64;
            while nk < kl {
                k += 1u64;
                nk = (k * k + k) / 2u64;
            }
            let mut l = kl - (k * k - k) / 2u64;

            i -= 1u64;
            j -= 1u64;
            k -= 1u64;
            l -= 1u64;

            let gi: usize = usize::cast_from(i);
            let gj: usize = usize::cast_from(j);
            let gk: usize = usize::cast_from(k);
            let gl: usize = usize::cast_from(l);
            let na: usize = usize::cast_from(natoms);
            let ng: usize = usize::cast_from(ngauss);
            let na2: usize = 2 * na;

            let gx_i: F = geom[gi];
            let gy_i: F = geom[na + gi];
            let gz_i: F = geom[na2 + gi];
            let gx_j: F = geom[gj];
            let gy_j: F = geom[na + gj];
            let gz_j: F = geom[na2 + gj];
            let gx_k: F = geom[gk];
            let gy_k: F = geom[na + gk];
            let gz_k: F = geom[na2 + gk];
            let gx_l: F = geom[gl];
            let gy_l: F = geom[na + gl];
            let gz_l: F = geom[na2 + gl];

            let one_f: F = F::new(1.0f32);
            let p15: F = F::new(1.5f32);
            let pm05: F = F::new(-0.5f32);
            let half: F = F::new(0.5f32);
            let four: F = F::new(4.0f32);

            let dxij: F = gx_i - gx_j;
            let dyij: F = gy_i - gy_j;
            let dzij: F = gz_i - gz_j;
            let dxij2: F = dxij * dxij;
            let dyij2: F = dyij * dyij;
            let dzij2: F = dzij * dzij;
            let rij_p: F = dxij2 + dyij2;
            let rij: F = rij_p + dzij2;

            let dxkl: F = gx_k - gx_l;
            let dykl: F = gy_k - gy_l;
            let dzkl: F = gz_k - gz_l;
            let dxkl2: F = dxkl * dxkl;
            let dykl2: F = dykl * dykl;
            let dzkl2: F = dzkl * dzkl;
            let rkl_p: F = dxkl2 + dykl2;
            let rkl: F = rkl_p + dzkl2;

            let mut eri = F::new(0.0f32);

            for ib in 0..ng {
                let xa: F = xpnt[ib];
                let ca: F = coef[ib];
                for jb in 0..ng {
                    let xb: F = xpnt[jb];
                    let cb: F = coef[jb];

                    let sab: F = xa + xb;
                    let aij: F = one_f / sab;

                    let xab: F = xa * xb;
                    let earg_a: F = -xab;
                    let earg_b: F = earg_a * aij;
                    let earg_ij: F = earg_b * rij;
                    let eterm_ij: F = F::exp(earg_ij);
                    let pterm_ij: F = F::powf(aij, p15);
                    let cab: F = ca * cb;
                    let dij_p: F = cab * eterm_ij;
                    let dij: F = dij_p * pterm_ij;

                    let dij_abs: F = F::abs(dij);
                    if dij_abs > dtol {
                        let wx_a: F = xa * gx_i;
                        let wx_b: F = xb * gx_j;
                        let wx_s: F = wx_a + wx_b;
                        let xij: F = aij * wx_s;

                        let wy_a: F = xa * gy_i;
                        let wy_b: F = xb * gy_j;
                        let wy_s: F = wy_a + wy_b;
                        let yij: F = aij * wy_s;

                        let wz_a: F = xa * gz_i;
                        let wz_b: F = xb * gz_j;
                        let wz_s: F = wz_a + wz_b;
                        let zij: F = aij * wz_s;

                        for kb in 0..ng {
                            let xc: F = xpnt[kb];
                            let cc: F = coef[kb];
                            for lb in 0..ng {
                                let xd: F = xpnt[lb];
                                let cd: F = coef[lb];

                                let scd: F = xc + xd;
                                let akl: F = one_f / scd;

                                let xcd: F = xc * xd;
                                let karg_a: F = -xcd;
                                let karg_b: F = karg_a * akl;
                                let karg_kl: F = karg_b * rkl;
                                let eterm_kl: F = F::exp(karg_kl);
                                let pterm_kl: F = F::powf(akl, p15);
                                let ccd: F = cc * cd;
                                let dkl_p: F = dij * ccd;
                                let dkl_q: F = dkl_p * eterm_kl;
                                let dkl: F = dkl_q * pterm_kl;

                                let dkl_abs: F = F::abs(dkl);
                                if dkl_abs > dtol {
                                    let sprod: F = sab * scd;
                                    let ssum: F = sab + scd;
                                    let aijkl: F = sprod / ssum;

                                    let qx_a: F = xc * gx_k;
                                    let qx_b: F = xd * gx_l;
                                    let qx_s: F = qx_a + qx_b;
                                    let qx_w: F = akl * qx_s;
                                    let px: F = xij - qx_w;
                                    let px2: F = px * px;

                                    let qy_a: F = xc * gy_k;
                                    let qy_b: F = xd * gy_l;
                                    let qy_s: F = qy_a + qy_b;
                                    let qy_w: F = akl * qy_s;
                                    let py: F = yij - qy_w;
                                    let py2: F = py * py;

                                    let qz_a: F = xc * gz_k;
                                    let qz_b: F = xd * gz_l;
                                    let qz_s: F = qz_a + qz_b;
                                    let qz_w: F = akl * qz_s;
                                    let pz: F = zij - qz_w;
                                    let pz2: F = pz * pz;

                                    let tsum_p: F = px2 + py2;
                                    let tsum: F = tsum_p + pz2;
                                    let tt: F = aijkl * tsum;

                                    let mut f0t = sqrtpi2;
                                    if tt > rcut {
                                        let boys: F = if legacy {
                                            let tt_n: F = tt;
                                            let tt_f: f32 = f32::cast_from(tt_n);
                                            let tt_r: f32 = f32::sqrt(tt_f);
                                            F::cast_from(tt_r)
                                        } else {
                                            F::sqrt(tt)
                                        };
                                        let erf_v: F = F::erf(boys);
                                        let pow_v: F = F::powf(tt, pm05);
                                        f0t = pow_v * erf_v;
                                    }

                                    let root: F = if legacy {
                                        let ak_n: F = aijkl;
                                        let ak_f: f32 = f32::cast_from(ak_n);
                                        let ak_r: f32 = f32::sqrt(ak_f);
                                        F::cast_from(ak_r)
                                    } else {
                                        F::sqrt(aijkl)
                                    };

                                    let contrib_p: F = dkl * f0t;
                                    let contrib: F = contrib_p * root;
                                    eri += contrib;
                                }
                            }
                        }
                    }
                }
            }

            if i == j {
                eri *= half;
            }
            if k == l {
                eri *= half;
            }
            if i == k && j == l {
                eri *= half;
            }

            let idx_ij: usize = gi * na + gj;
            let idx_kl: usize = gk * na + gl;
            let idx_ik: usize = gi * na + gk;
            let idx_il: usize = gi * na + gl;
            let idx_jk: usize = gj * na + gk;
            let idx_jl: usize = gj * na + gl;

            let d_ij: F = dens[idx_ij];
            let d_kl: F = dens[idx_kl];
            let d_ik: F = dens[idx_ik];
            let d_il: F = dens[idx_il];
            let d_jk: F = dens[idx_jk];
            let d_jl: F = dens[idx_jl];

            let e4: F = eri * four;
            let upd0: F = d_kl * e4;
            let upd1: F = d_ij * e4;
            let upd2_p: F = d_jl * eri;
            let upd2: F = -upd2_p;
            let upd3_p: F = d_jk * eri;
            let upd3: F = -upd3_p;
            let upd4_p: F = d_il * eri;
            let upd4: F = -upd4_p;
            let upd5_p: F = d_ik * eri;
            let upd5: F = -upd5_p;

            let _o0: F = fock[idx_ij].fetch_add(upd0);
            let _o1: F = fock[idx_kl].fetch_add(upd1);
            let _o2: F = fock[idx_ik].fetch_add(upd2);
            let _o3: F = fock[idx_il].fetch_add(upd3);
            let _o4: F = fock[idx_jk].fetch_add(upd4);
            let _o5: F = fock[idx_jl].fetch_add(upd5);
        }
    }
}
