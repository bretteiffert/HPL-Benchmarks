//! CubeCL Hartree-Fock two-electron Fock-build benchmark with device profiling and optional fp64 CPU correctness checks.
//! Precision and legacy square-root behavior are controlled by in-source constants that default to fp64 and native square root.
//! The runtime is selected at build time with the CUDA or HIP Cargo feature, and the device name comes from CubeCL at runtime.

use cubecl::prelude::*;
use std::env;
use std::time::Duration;

#[cfg(all(feature = "cuda", not(feature = "hip")))]
type Rt = cubecl::cuda::CudaRuntime;
#[cfg(feature = "hip")]
type Rt = cubecl::hip::HipRuntime;

const PRECISION: u32 = 64;

const LEGACY_SQRTF: bool = false;

const BLK_SIZE: u32 = 256;

const REF_SQRTPI2: f64 = 1.12837916709551255856;
const REF_DTOL: f64 = 1.0e-12;
const REF_RCUT: f64 = 1.0e-12;

#[cube]
fn idx_sqrt(x: u64) -> u64 {
    let widened: f32 = f32::cast_from(x);
    let rooted: f32 = f32::sqrt(widened);
    let out: u64 = u64::cast_from(rooted);
    out
}

#[cube(launch_unchecked)]
fn hartree_fock_kernel<F: Float + CubeElement>(
    geom: &Array<F>,
    xpnt: &Array<F>,
    coef: &Array<F>,
    dens: &Array<F>,
    schwarz: &Array<F>,
    fock: &mut Array<Atomic<F>>,
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
        let mut ij: u64 = idx_sqrt(seed_ij);
        let mut n: u64 = (ij * ij + ij) / 2u64;
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
            let mut i: u64 = root_i - 1u64;
            let mut ni: u64 = (i * i + i) / 2u64;
            while ni < ij {
                i += 1u64;
                ni = (i * i + i) / 2u64;
            }
            let mut j: u64 = ij - (i * i - i) / 2u64;

            let seed_k: u64 = 2u64 * kl;
            let mut k: u64 = idx_sqrt(seed_k);
            let mut nk: u64 = (k * k + k) / 2u64;
            while nk < kl {
                k += 1u64;
                nk = (k * k + k) / 2u64;
            }
            let mut l: u64 = kl - (k * k - k) / 2u64;

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

            let mut eri: F = F::new(0.0f32);

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

                                    let mut f0t: F = sqrtpi2;
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

fn erf_f64(x: f64) -> f64 {
    unsafe extern "C" {
        fn erf(x: f64) -> f64;
    }
    unsafe { erf(x) }
}

trait Real:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::MulAssign
    + PartialOrd
{
    const NAME: &'static str;
    fn h_from_f64(x: f64) -> Self;
    fn h_to_f64(self) -> f64;
    fn h_exp(self) -> Self;
    fn h_erf(self) -> Self;
    fn h_sqrt(self) -> Self;
    fn h_abs(self) -> Self;
    fn h_powf(self, e: Self) -> Self;
    fn h_sqrt_f32(self) -> Self;
}

impl Real for f32 {
    const NAME: &'static str = "fp32";
    fn h_from_f64(x: f64) -> Self { x as f32 }
    fn h_to_f64(self) -> f64 { self as f64 }
    fn h_exp(self) -> Self { self.exp() }
    fn h_erf(self) -> Self { erf_f64(self as f64) as f32 }
    fn h_sqrt(self) -> Self { self.sqrt() }
    fn h_abs(self) -> Self { self.abs() }
    fn h_powf(self, e: Self) -> Self { self.powf(e) }
    fn h_sqrt_f32(self) -> Self { self.sqrt() }
}

impl Real for f64 {
    const NAME: &'static str = "fp64";
    fn h_from_f64(x: f64) -> Self { x }
    fn h_to_f64(self) -> f64 { self }
    fn h_exp(self) -> Self { self.exp() }
    fn h_erf(self) -> Self { erf_f64(self) }
    fn h_sqrt(self) -> Self { self.sqrt() }
    fn h_abs(self) -> Self { self.abs() }
    fn h_powf(self, e: Self) -> Self { self.powf(e) }
    fn h_sqrt_f32(self) -> Self { (self as f32).sqrt() as f64 }
}

struct Consts<T> {
    sqrtpi2_host: T,
    sqrtpi2_dev: T,
    dtol: T,
    rcut: T,
    tobohrs: T,
    legacy: bool,
}

impl<T: Real> Consts<T> {
    fn new(legacy: bool) -> Self {
        Self {
            sqrtpi2_host: T::h_from_f64(3.1415926535897931f64.powf(-0.5) * 2.0),
            sqrtpi2_dev: T::h_from_f64(1.12837916709551255856),
            dtol: T::h_from_f64(1.0e-12),
            rcut: T::h_from_f64(1.0e-12),
            tobohrs: T::h_from_f64(1.889725987722),
            legacy,
        }
    }
    fn sqrt_boys(&self, x: T) -> T {
        if self.legacy { x.h_sqrt_f32() } else { x.h_sqrt() }
    }
}

fn sqh<T: Real>(x: T) -> T {
    x * x
}

#[allow(clippy::too_many_arguments)]
fn ssss_host<T: Real>(
    i: usize, j: usize, k: usize, l: usize, ngauss: usize,
    xpnt: &[T], coef: &[T], geom: &[T], natoms: usize, c: &Consts<T>,
) -> T {
    let mut eri = T::h_from_f64(0.0);
    let one_t = T::h_from_f64(1.0);
    let p15 = T::h_from_f64(1.5);
    let pm05 = T::h_from_f64(-0.5);
    for ib in 0..ngauss {
        for jb in 0..ngauss {
            let aij = one_t / (xpnt[ib] + xpnt[jb]);
            let dij = coef[ib] * coef[jb]
                * (-xpnt[ib] * xpnt[jb] * aij
                    * (sqh(geom[i] - geom[j])
                        + sqh(geom[natoms + i] - geom[natoms + j])
                        + sqh(geom[2 * natoms + i] - geom[2 * natoms + j])))
                    .h_exp()
                * aij.h_powf(p15);
            if dij.h_abs() > c.dtol {
                let xij = aij * (xpnt[ib] * geom[i] + xpnt[jb] * geom[j]);
                let yij = aij * (xpnt[ib] * geom[natoms + i] + xpnt[jb] * geom[natoms + j]);
                let zij =
                    aij * (xpnt[ib] * geom[2 * natoms + i] + xpnt[jb] * geom[2 * natoms + j]);
                for kb in 0..ngauss {
                    for lb in 0..ngauss {
                        let akl = one_t / (xpnt[kb] + xpnt[lb]);
                        let dkl = dij * coef[kb] * coef[lb]
                            * (-xpnt[kb] * xpnt[lb] * akl
                                * (sqh(geom[k] - geom[l])
                                    + sqh(geom[natoms + k] - geom[natoms + l])
                                    + sqh(geom[2 * natoms + k] - geom[2 * natoms + l])))
                                .h_exp()
                            * akl.h_powf(p15);
                        if dkl.h_abs() > c.dtol {
                            let aijkl = (xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb])
                                / (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb]);
                            let tt = aijkl
                                * (sqh(xij - akl * (xpnt[kb] * geom[k] + xpnt[lb] * geom[l]))
                                    + sqh(yij
                                        - akl
                                            * (xpnt[kb] * geom[natoms + k]
                                                + xpnt[lb] * geom[natoms + l]))
                                    + sqh(zij
                                        - akl
                                            * (xpnt[kb] * geom[2 * natoms + k]
                                                + xpnt[lb] * geom[2 * natoms + l])));
                            let mut f0t = c.sqrtpi2_host;
                            if tt > c.rcut {
                                f0t = tt.h_powf(pm05) * c.sqrt_boys(tt).h_erf();
                            }
                            eri += dkl * f0t * c.sqrt_boys(aijkl);
                        }
                    }
                }
            }
        }
    }
    eri
}

fn sqd(x: f64) -> f64 {
    x * x
}

#[allow(clippy::too_many_arguments)]
fn ssss_ref(
    i: usize, j: usize, k: usize, l: usize, ngauss: usize,
    xpnt: &[f64], coef: &[f64], geom: &[f64], natoms: usize,
) -> f64 {
    let mut eri = 0.0f64;
    for ib in 0..ngauss {
        for jb in 0..ngauss {
            let aij = 1.0 / (xpnt[ib] + xpnt[jb]);
            let dij = coef[ib] * coef[jb]
                * (-xpnt[ib] * xpnt[jb] * aij
                    * (sqd(geom[i] - geom[j])
                        + sqd(geom[natoms + i] - geom[natoms + j])
                        + sqd(geom[2 * natoms + i] - geom[2 * natoms + j])))
                    .exp()
                * aij.powf(1.5);
            if dij.abs() > REF_DTOL {
                let xij = aij * (xpnt[ib] * geom[i] + xpnt[jb] * geom[j]);
                let yij = aij * (xpnt[ib] * geom[natoms + i] + xpnt[jb] * geom[natoms + j]);
                let zij =
                    aij * (xpnt[ib] * geom[2 * natoms + i] + xpnt[jb] * geom[2 * natoms + j]);
                for kb in 0..ngauss {
                    for lb in 0..ngauss {
                        let akl = 1.0 / (xpnt[kb] + xpnt[lb]);
                        let dkl = dij * coef[kb] * coef[lb]
                            * (-xpnt[kb] * xpnt[lb] * akl
                                * (sqd(geom[k] - geom[l])
                                    + sqd(geom[natoms + k] - geom[natoms + l])
                                    + sqd(geom[2 * natoms + k] - geom[2 * natoms + l])))
                                .exp()
                            * akl.powf(1.5);
                        if dkl.abs() > REF_DTOL {
                            let aijkl = (xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb])
                                / (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb]);
                            let tt = aijkl
                                * (sqd(xij - akl * (xpnt[kb] * geom[k] + xpnt[lb] * geom[l]))
                                    + sqd(yij
                                        - akl
                                            * (xpnt[kb] * geom[natoms + k]
                                                + xpnt[lb] * geom[natoms + l]))
                                    + sqd(zij
                                        - akl
                                            * (xpnt[kb] * geom[2 * natoms + k]
                                                + xpnt[lb] * geom[2 * natoms + l])));
                            let mut f0t = REF_SQRTPI2;
                            if tt > REF_RCUT {
                                f0t = tt.powf(-0.5) * erf_f64(tt.sqrt());
                            }
                            eri += dkl * f0t * aijkl.sqrt();
                        }
                    }
                }
            }
        }
    }
    eri
}

fn fock_build_cpu(
    ngauss: usize, natoms: usize,
    geom: &[f64], xpnt: &[f64], coef: &[f64], dens: &[f64],
) -> Vec<f64> {
    let mut fock = vec![0.0f64; natoms * natoms];
    let nn = (natoms * natoms + natoms) / 2;
    let mut schwarz = vec![0.0f64; nn + 1];
    let mut ijr = 0usize;
    for i in 0..natoms {
        for j in 0..=i {
            ijr += 1;
            schwarz[ijr] = ssss_ref(i, j, i, j, ngauss, xpnt, coef, geom, natoms).abs().sqrt();
        }
    }
    for i in 0..natoms {
        for j in 0..=i {
            let ij = i * (i + 1) / 2 + j + 1;
            for k in 0..natoms {
                for l in 0..=k {
                    let kl = k * (k + 1) / 2 + l + 1;
                    if kl > ij {
                        continue;
                    }
                    if !(schwarz[ij] * schwarz[kl] > REF_DTOL) {
                        continue;
                    }
                    let mut eri = ssss_ref(i, j, k, l, ngauss, xpnt, coef, geom, natoms);
                    if i == j { eri *= 0.5; }
                    if k == l { eri *= 0.5; }
                    if i == k && j == l { eri *= 0.5; }
                    fock[i * natoms + j] += dens[k * natoms + l] * eri * 4.0;
                    fock[k * natoms + l] += dens[i * natoms + j] * eri * 4.0;
                    fock[i * natoms + k] -= dens[j * natoms + l] * eri;
                    fock[i * natoms + l] -= dens[j * natoms + k] * eri;
                    fock[j * natoms + k] -= dens[i * natoms + l] * eri;
                    fock[j * natoms + l] -= dens[i * natoms + k] * eri;
                }
            }
        }
    }
    fock
}

fn energy_from_fock(fock: &[f64], dens: &[f64], natoms: usize) -> f64 {
    let mut e = 0.0;
    for t in 0..natoms * natoms {
        e += fock[t] * dens[t];
    }
    e * 0.5
}

fn energy_symmetrized(fock: &[f64], dens: &[f64], natoms: usize) -> f64 {
    let mut e = 0.0;
    for i in 0..natoms {
        for j in 0..natoms {
            let f = if i == j {
                fock[i * natoms + j]
            } else {
                fock[i * natoms + j] + fock[j * natoms + i]
            };
            e += f * dens[i * natoms + j];
        }
    }
    e * 0.5
}

struct Opts {
    csv_output: bool,
    run_check: bool,
    force_check: bool,
    max_check_atoms: usize,
    num_iter: usize,
}

fn time_once<R: Runtime, Fun: FnOnce() + Send>(client: &ComputeClient<R>, f: Fun) -> Duration {
    let (_, profile_duration) = client
        .profile(f, "hartree_fock")
        .expect("device profiling failed");
    let ticks = cubecl::future::block_on(profile_duration.resolve());
    ticks.duration()
}

fn to_bytes<T: CubeElement>(data: &[T]) -> cubecl::bytes::Bytes {
    cubecl::bytes::Bytes::from_bytes_vec(<T as CubeElement>::as_bytes(data).to_vec())
}

fn run<F: Float + CubeElement + Real>(path: &str, o: &Opts, legacy: bool) {
    let c = Consts::<F>::new(legacy);

    let text = std::fs::read_to_string(path).expect("ERROR: Could not read file!");
    let mut tok = text.split_ascii_whitespace();
    let ngauss: usize = tok.next().expect("ERROR: Could not read # of gaussians").parse().unwrap();
    let natoms: usize = tok.next().expect("ERROR: Could not read # of atoms").parse().unwrap();

    let mut xpnt_d = vec![0.0f64; ngauss];
    let mut coef_d = vec![0.0f64; ngauss];
    for i in 0..ngauss {
        xpnt_d[i] = tok.next().unwrap().parse().unwrap();
        coef_d[i] = tok.next().unwrap().parse().unwrap();
    }
    let mut geom_d = vec![0.0f64; 3 * natoms];
    for i in 0..natoms {
        for j in 0..3 {
            geom_d[j * natoms + i] = tok.next().unwrap().parse().unwrap();
        }
    }

    let nsq = natoms * natoms;

    let h_xpnt: Vec<F> = xpnt_d.iter().map(|&x| F::h_from_f64(x)).collect();
    let mut h_coef: Vec<F> = coef_d.iter().map(|&x| F::h_from_f64(x)).collect();
    let mut h_geom: Vec<F> = geom_d.iter().map(|&x| F::h_from_f64(x)).collect();

    let mut h_dens = vec![F::h_from_f64(0.1); nsq];
    let mut dens_d = vec![0.1f64; nsq];
    for i in 0..natoms {
        h_dens[i * natoms + i] = F::h_from_f64(1.0);
        dens_d[i * natoms + i] = 1.0;
    }

    for i in 0..ngauss {
        h_coef[i] = h_coef[i] * (F::h_from_f64(2.0) * h_xpnt[i]).h_powf(F::h_from_f64(0.75));
        coef_d[i] *= (2.0 * xpnt_d[i]).powf(0.75);
    }
    for i in 0..natoms {
        for a in 0..3 {
            h_geom[a * natoms + i] *= c.tobohrs;
            geom_d[a * natoms + i] *= 1.889725987722;
        }
    }

    let mut h_fock = vec![F::h_from_f64(0.0); nsq];

    let nn = (natoms * natoms + natoms) / 2;
    let mut h_schwarz = vec![F::h_from_f64(0.0); nn + 1];
    let mut ij = 0usize;
    for i in 0..natoms {
        for j in 0..=i {
            ij += 1;
            h_schwarz[ij] =
                ssss_host(i, j, i, j, ngauss, &h_xpnt, &h_coef, &h_geom, natoms, &c)
                    .h_abs()
                    .h_sqrt();
        }
    }

    let nnnn = ((nn as u64) * (nn as u64) + nn as u64) / 2;
    let n_blks = ((nnnn + 1 + BLK_SIZE as u64 - 1) / BLK_SIZE as u64) as u32;
    assert!(
        nnnn + 1 <= u32::MAX as u64,
        "nnnn = {} exceeds CubeCL's 32-bit global index space (natoms too large)",
        nnnn
    );

    let client = Rt::client(&Default::default());
    let gpu_name = Rt::name(&client);

    let d_xpnt = client.create(to_bytes(&h_xpnt));
    let d_coef = client.create(to_bytes(&h_coef));
    let d_geom = client.create(to_bytes(&h_geom));
    let d_dens = client.create(to_bytes(&h_dens));
    let d_fock = client.create(to_bytes(&h_fock));
    let d_schwarz = client.create(to_bytes(&h_schwarz));

    let cube_count = CubeCount::Static(n_blks, 1, 1);
    let cube_dim = CubeDim::new_1d(BLK_SIZE);

    let dev_dtol = c.dtol;
    let dev_rcut = c.rcut;
    let dev_sqrtpi2 = c.sqrtpi2_dev;

    let launch = || unsafe {
        hartree_fock_kernel::launch_unchecked::<F, Rt>(
            &client,
            cube_count.clone(),
            cube_dim,
            ArrayArg::from_raw_parts(d_geom.clone(), 3 * natoms),
            ArrayArg::from_raw_parts(d_xpnt.clone(), ngauss),
            ArrayArg::from_raw_parts(d_coef.clone(), ngauss),
            ArrayArg::from_raw_parts(d_dens.clone(), nsq),
            ArrayArg::from_raw_parts(d_schwarz.clone(), nn + 1),
            ArrayArg::from_raw_parts(d_fock.clone(), nsq),
            nnnn,
            ngauss as u32,
            natoms as u32,
            dev_dtol,
            dev_rcut,
            dev_sqrtpi2,
            legacy,
        )
    };

    launch();
    cubecl::future::block_on(client.sync());

    let bytes = client
        .read_one(d_fock.clone())
        .expect("failed to read fock buffer back from device");
    let dev_out: &[F] = <F as CubeElement>::from_bytes(&bytes);
    h_fock.copy_from_slice(dev_out);

    let mut erep = F::h_from_f64(0.0);
    for i in 0..natoms {
        for j in 0..natoms {
            erep += h_fock[i * natoms + j] * h_dens[i * natoms + j];
        }
    }

    let sqrt_mode = if legacy { "legacy-sqrtf" } else { "native" };

    if !o.csv_output {
        println!("2e- energy={:.16}", erep.h_to_f64() * 0.5);
        if o.run_check && (o.force_check || natoms <= o.max_check_atoms) {
            let fock_gpu: Vec<f64> = h_fock.iter().map(|&x| x.h_to_f64()).collect();
            let fock_ref = fock_build_cpu(ngauss, natoms, &geom_d, &xpnt_d, &coef_d, &dens_d);

            let e_gpu = energy_from_fock(&fock_gpu, &dens_d, natoms);
            let e_ref = energy_from_fock(&fock_ref, &dens_d, natoms);
            let denom = if e_ref.abs() > 0.0 { e_ref.abs() } else { 1.0 };
            let rel_loss = (e_gpu - e_ref).abs() / denom;
            let e_inpipe = erep.h_to_f64() * 0.5;
            let rel_inpipe = (e_inpipe - e_ref).abs() / denom;

            let (mut maxd, mut sumsq, mut fscale) = (0.0f64, 0.0f64, 0.0f64);
            for t in 0..nsq {
                let d = (fock_gpu[t] - fock_ref[t]).abs();
                if d > maxd { maxd = d; }
                sumsq += d * d;
                if fock_ref[t].abs() > fscale { fscale = fock_ref[t].abs(); }
            }
            let rmsd = (sumsq / nsq as f64).sqrt();
            if fscale == 0.0 { fscale = 1.0; }

            let mut asym = 0.0f64;
            for i in 0..natoms {
                for j in 0..natoms {
                    let d = (fock_gpu[i * natoms + j] - fock_gpu[j * natoms + i]).abs();
                    if d > asym { asym = d; }
                }
            }

            println!();
            println!("--- precision check (build: {}, sqrt: {}) ---", F::NAME, sqrt_mode);
            println!("2e- energy ({}, {} reduction)      = {:.16}", F::NAME, F::NAME, e_inpipe);
            println!("2e- energy ({} Fock, fp64 reduction) = {:.16}", F::NAME, e_gpu);
            println!("2e- energy (fp64 CPU reference)       = {:.16}", e_ref);
            println!("precision loss, Fock build only       = {:.6e}", rel_loss);
            println!("precision loss, incl. {} reduction  = {:.6e}", F::NAME, rel_inpipe);
            println!("max |F - F_ref| (elementwise)         = {:.6e}  (rel {:.6e})", maxd, maxd / fscale);
            println!("rms |F - F_ref| (elementwise)         = {:.6e}", rmsd);
            println!("max |F(i,j) - F(j,i)|                 = {:.6e}", asym);
            println!("2e- energy from symmetrized F=A+A^T   = {:.16}   (diagnostic)",
                     energy_symmetrized(&fock_gpu, &dens_d, natoms));
            println!("---------------------------------------------------------------");
        } else if o.run_check {
            println!("(correctness check skipped: natoms={} > {}; use --check to force)",
                     natoms, o.max_check_atoms);
        }
    } else {
        println!("backend,GPU,precision,sqrt_mode,natoms,ngauss,exec_time_ms");
        for _ in 0..o.num_iter {
            cubecl::future::block_on(client.sync());
            let d = time_once::<Rt, _>(&client, || {
                launch();
            });
            let ms = d.as_secs_f64() * 1000.0;
            println!("CubeCL,{},{},{},{},{},{:.6}",
                     gpu_name, F::NAME, sqrt_mode, natoms, ngauss, ms);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("ERROR: Please provide an input file");
        return;
    }
    let mut o = Opts {
        csv_output: false,
        run_check: true,
        force_check: false,
        max_check_atoms: 128,
        num_iter: 10,
    };
    for s in &args[2..] {
        if s == "--csv" {
            o.csv_output = true;
        } else if s == "--check" {
            o.run_check = true;
            o.force_check = true;
        } else if s == "--no-check" {
            o.run_check = false;
        } else if let Some(v) = s.strip_prefix("--max-check-atoms=") {
            o.max_check_atoms = v.parse().unwrap();
        } else if let Some(v) = s.strip_prefix("--iters=") {
            o.num_iter = v.parse().unwrap();
        }
    }

    let legacy = LEGACY_SQRTF;
    match PRECISION {
        32 => run::<f32>(&args[1], &o, legacy),
        64 => run::<f64>(&args[1], &o, legacy),
        other => panic!("PRECISION must be 32 or 64, got {other}"),
    }
}
