//! Host side: input parsing, problem setup, and the fp64 CPU reference.
//!
//! The integral, reference and energy code below is the original
//! implementation moved here unchanged (only `pub` added, and precision names
//! now read float32/float64 to match the other benchmarks).

const REF_SQRTPI2: f64 = 1.12837916709551255856;
const REF_DTOL: f64 = 1.0e-12;
const REF_RCUT: f64 = 1.0e-12;

pub fn erf_f64(x: f64) -> f64 {
    unsafe extern "C" {
        fn erf(x: f64) -> f64;
    }
    unsafe { erf(x) }
}

pub trait Real:
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
    const NAME: &'static str = "float32";
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
    const NAME: &'static str = "float64";
    fn h_from_f64(x: f64) -> Self { x }
    fn h_to_f64(self) -> f64 { self }
    fn h_exp(self) -> Self { self.exp() }
    fn h_erf(self) -> Self { erf_f64(self) }
    fn h_sqrt(self) -> Self { self.sqrt() }
    fn h_abs(self) -> Self { self.abs() }
    fn h_powf(self, e: Self) -> Self { self.powf(e) }
    fn h_sqrt_f32(self) -> Self { (self as f32).sqrt() as f64 }
}

pub struct Consts<T> {
    pub sqrtpi2_host: T,
    pub sqrtpi2_dev: T,
    pub dtol: T,
    pub rcut: T,
    pub tobohrs: T,
    pub legacy: bool,
}

impl<T: Real> Consts<T> {
    pub fn new(legacy: bool) -> Self {
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
pub fn ssss_host<T: Real>(
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
pub fn ssss_ref(
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

pub fn fock_build_cpu(
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

pub fn energy_from_fock(fock: &[f64], dens: &[f64], natoms: usize) -> f64 {
    let mut e = 0.0;
    for t in 0..natoms * natoms {
        e += fock[t] * dens[t];
    }
    e * 0.5
}

pub fn energy_symmetrized(fock: &[f64], dens: &[f64], natoms: usize) -> f64 {
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


/// One parsed input: basis exponents/coefficients and atom positions (Å),
/// stored structure-of-arrays as `geom[axis * natoms + atom]`.
pub struct Input {
    pub ngauss: usize,
    pub natoms: usize,
    pub xpnt: Vec<f64>,
    pub coef: Vec<f64>,
    pub geom: Vec<f64>,
}

/// Parses the whitespace-separated input format: `ngauss natoms`, then
/// `ngauss` (exponent, coefficient) pairs, then `natoms` (x, y, z) positions.
pub fn read_input(path: &str) -> Result<Input, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read input file '{}': {}", path, e))?;
    let mut tok = text.split_ascii_whitespace();
    let mut word = |what: &str| -> Result<&str, String> {
        tok.next()
            .ok_or_else(|| format!("{}: file ended while reading {}", path, what))
    };
    let count = |s: &str, what: &str| -> Result<usize, String> {
        s.parse::<usize>()
            .map_err(|e| format!("{}: invalid {}: {}", path, what, e))
    };
    let ngauss = count(word("number of gaussians")?, "number of gaussians")?;
    let natoms = count(word("number of atoms")?, "number of atoms")?;
    let mut next = |what: &str| -> Result<f64, String> {
        word(what)?
            .parse::<f64>()
            .map_err(|e| format!("{}: invalid {}: {}", path, what, e))
    };
    let mut xpnt = vec![0.0f64; ngauss];
    let mut coef = vec![0.0f64; ngauss];
    for i in 0..ngauss {
        xpnt[i] = next("gaussian exponent")?;
        coef[i] = next("gaussian coefficient")?;
    }
    let mut geom = vec![0.0f64; 3 * natoms];
    for i in 0..natoms {
        for j in 0..3 {
            geom[j * natoms + i] = next("atom coordinate")?;
        }
    }
    Ok(Input { ngauss, natoms, xpnt, coef, geom })
}

/// Device-precision arrays plus their fp64 twins for the CPU reference.
pub struct Problem<T> {
    pub ngauss: usize,
    pub natoms: usize,
    pub xpnt: Vec<T>,
    pub coef: Vec<T>,
    pub geom: Vec<T>,
    pub dens: Vec<T>,
    pub schwarz: Vec<T>,
    pub xpnt_d: Vec<f64>,
    pub coef_d: Vec<f64>,
    pub geom_d: Vec<f64>,
    pub dens_d: Vec<f64>,
}

impl<T: Real> Problem<T> {
    /// Same setup as the original: density 0.1 off-diagonal and 1.0 on the
    /// diagonal, normalized coefficients, geometry converted to bohr, and the
    /// Schwarz screening table computed on the host in precision `T`.
    pub fn new(input: &Input, c: &Consts<T>) -> Self {
        let (ngauss, natoms) = (input.ngauss, input.natoms);
        let nsq = natoms * natoms;
        let xpnt: Vec<T> = input.xpnt.iter().map(|&x| T::h_from_f64(x)).collect();
        let mut coef: Vec<T> = input.coef.iter().map(|&x| T::h_from_f64(x)).collect();
        let mut geom: Vec<T> = input.geom.iter().map(|&x| T::h_from_f64(x)).collect();
        let mut coef_d = input.coef.clone();
        let mut geom_d = input.geom.clone();

        let mut dens = vec![T::h_from_f64(0.1); nsq];
        let mut dens_d = vec![0.1f64; nsq];
        for i in 0..natoms {
            dens[i * natoms + i] = T::h_from_f64(1.0);
            dens_d[i * natoms + i] = 1.0;
        }
        for i in 0..ngauss {
            coef[i] = coef[i] * (T::h_from_f64(2.0) * xpnt[i]).h_powf(T::h_from_f64(0.75));
            coef_d[i] *= (2.0 * input.xpnt[i]).powf(0.75);
        }
        for i in 0..natoms {
            for a in 0..3 {
                geom[a * natoms + i] *= c.tobohrs;
                geom_d[a * natoms + i] *= 1.889725987722;
            }
        }

        let nn = (natoms * natoms + natoms) / 2;
        let mut schwarz = vec![T::h_from_f64(0.0); nn + 1];
        let mut ij = 0usize;
        for i in 0..natoms {
            for j in 0..=i {
                ij += 1;
                schwarz[ij] = ssss_host(i, j, i, j, ngauss, &xpnt, &coef, &geom, natoms, c)
                    .h_abs()
                    .h_sqrt();
            }
        }

        Self {
            ngauss,
            natoms,
            xpnt,
            coef,
            geom,
            dens,
            schwarz,
            xpnt_d: input.xpnt.clone(),
            coef_d,
            geom_d,
            dens_d,
        }
    }

    /// Number of (ij) pairs, n(n+1)/2.
    pub fn pairs(&self) -> usize {
        (self.natoms * self.natoms + self.natoms) / 2
    }

    /// Number of unique (ij|kl) quartets, one kernel unit each.
    pub fn quartets(&self) -> u64 {
        let nn = self.pairs() as u64;
        (nn * nn + nn) / 2
    }
}

