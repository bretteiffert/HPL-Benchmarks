//! Backend-agnostic miniBUDE correctness metrics, ULP histograms, and C-compatible numeric formatting.
//! Comparisons report numerical values without a pass/fail gate and round ULP operands to the tested precision.
//! Deck-reference comparisons use norms because the text energies have limited precision.

pub const REPORT_TOLERANCE_PCT: f64 = 0.025;

fn ordered_f32(f: f32) -> i64 {
    let i = f.to_bits() as i32;
    if i < 0 {
        0x8000_0000i64 - (i as u32 as i64)
    } else {
        i as i64
    }
}

fn ordered_f64(d: f64) -> i64 {
    let i = d.to_bits() as i64;
    if i < 0 {
        i64::MIN.wrapping_sub(i)
    } else {
        i
    }
}

pub fn ulp_distance_f32(a: f32, b: f32) -> i64 {
    if a.is_nan() || b.is_nan() {
        return i64::MAX;
    }
    if a == b {
        return 0;
    }
    if a.is_infinite() || b.is_infinite() {
        return i64::MAX;
    }
    let d = ordered_f32(a).wrapping_sub(ordered_f32(b));
    if d < 0 {
        -d
    } else {
        d
    }
}

pub fn ulp_distance_f64(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() {
        return i64::MAX;
    }
    if a == b {
        return 0;
    }
    if a.is_infinite() || b.is_infinite() {
        return i64::MAX;
    }
    let d = ordered_f64(a).wrapping_sub(ordered_f64(b));
    if d < 0 {
        -d
    } else {
        d
    }
}

pub const ULP_BUCKETS: [i64; 9] = [1, 2, 4, 8, 16, 64, 256, 1024, 4096];
pub const ULP_NBUCKETS: usize = ULP_BUCKETS.len() + 1;

pub fn ulp_bucket_label(i: usize) -> String {
    if i == 0 {
        return "0".to_string();
    }
    if i == ULP_NBUCKETS - 1 {
        return format!(">={}", ULP_BUCKETS[ULP_BUCKETS.len() - 1]);
    }
    let lo = ULP_BUCKETS[i - 1];
    let hi = ULP_BUCKETS[i] - 1;
    if lo == hi {
        format!("{}", lo)
    } else {
        format!("{}-{}", lo, hi)
    }
}

#[derive(Clone, Debug)]
pub struct Metrics {
    pub l2_rel_norm: f64,
    pub rms_abs: f64,
    pub mean_signed: f64,
    pub max_abs: f64,
    pub max_abs_idx: usize,
    pub max_rel: f64,
    pub max_rel_idx: usize,
    pub max_ulp: i64,
    pub max_ulp_idx: usize,
    pub ulp_histogram: [usize; ULP_NBUCKETS],
    pub compared: usize,
    pub non_finite: usize,
    pub beyond_tolerance: usize,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            l2_rel_norm: 0.0,
            rms_abs: 0.0,
            mean_signed: 0.0,
            max_abs: 0.0,
            max_abs_idx: 0,
            max_rel: 0.0,
            max_rel_idx: 0,
            max_ulp: 0,
            max_ulp_idx: 0,
            ulp_histogram: [0; ULP_NBUCKETS],
            compared: 0,
            non_finite: 0,
            beyond_tolerance: 0,
        }
    }
}

#[derive(Default)]
struct Kahan {
    sum: f64,
    c: f64,
}

impl Kahan {
    fn add(&mut self, x: f64) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }
}

pub fn compute_metrics(measured: &[f64], reference: &[f64], precision_bits: usize) -> Metrics {
    let mut m = Metrics::default();
    let n = measured.len().min(reference.len());

    let mut err_sq = Kahan::default();
    let mut ref_sq = Kahan::default();
    let mut signed_err = Kahan::default();

    for i in 0..n {
        let x = measured[i];
        let r = reference[i];

        if !x.is_finite() || !r.is_finite() {
            m.non_finite += 1;
            continue;
        }

        let err = x - r;
        let abs_err = err.abs();

        err_sq.add(err * err);
        ref_sq.add(r * r);
        signed_err.add(err);

        if abs_err > m.max_abs {
            m.max_abs = abs_err;
            m.max_abs_idx = i;
        }

        let denom = r.abs();
        if denom > 0.0 {
            let rel = abs_err / denom;
            if rel > m.max_rel {
                m.max_rel = rel;
                m.max_rel_idx = i;
            }
            if rel * 100.0 > REPORT_TOLERANCE_PCT {
                m.beyond_tolerance += 1;
            }
        }

        let u = if precision_bits == 32 {
            ulp_distance_f32(x as f32, r as f32)
        } else {
            ulp_distance_f64(x, r)
        };
        if u > m.max_ulp {
            m.max_ulp = u;
            m.max_ulp_idx = i;
        }
        let mut bucket = ULP_NBUCKETS - 1;
        for (b, bound) in ULP_BUCKETS.iter().enumerate() {
            if u < *bound {
                bucket = b;
                break;
            }
        }
        m.ulp_histogram[bucket] += 1;

        m.compared += 1;
    }

    if m.compared > 0 {
        m.rms_abs = (err_sq.sum / m.compared as f64).sqrt();
        m.mean_signed = signed_err.sum / m.compared as f64;
        m.l2_rel_norm = if ref_sq.sum > 0.0 {
            err_sq.sum.sqrt() / ref_sq.sum.sqrt()
        } else {
            0.0
        };
    }

    m
}

pub fn compare_to_deck(measured: &[f64], ref_energies: &[f32]) -> Metrics {
    let n = ref_energies.len().min(measured.len());
    let reference: Vec<f64> = ref_energies[..n].iter().map(|&x| x as f64).collect();
    compute_metrics(measured, &reference, 64)
}

pub fn sci(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let s = format!("{:.*e}", prec, x);
    match s.split_once('e') {
        Some((mantissa, exp)) => {
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exp.strip_prefix('+').unwrap_or(exp)),
            };
            format!("{}e{}{:0>2}", mantissa, sign, digits)
        }
        None => s,
    }
}

pub fn gfmt(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let exp = x.abs().log10().floor() as i32;
    if exp < -4 || exp >= 6 {
        let s = sci(x, 5);
        let (mantissa, e) = s.split_once('e').unwrap();
        let m = if mantissa.contains('.') {
            mantissa.trim_end_matches('0').trim_end_matches('.')
        } else {
            mantissa
        };
        format!("{}e{}", m, e)
    } else {
        let decimals = (5 - exp).max(0) as usize;
        let s = format!("{:.*}", decimals, x);
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_basics() {
        assert_eq!(ulp_distance_f32(1.0, 1.0), 0);
        assert_eq!(ulp_distance_f32(1.0, 1.0000001), 1);
        assert_eq!(ulp_distance_f32(-0.0, 0.0), 0);
        assert_eq!(ulp_distance_f32(f32::NAN, 1.0), i64::MAX);
        assert_eq!(ulp_distance_f32(f32::INFINITY, f32::INFINITY), 0);
        assert_eq!(ulp_distance_f64(1.0, 1.0 + f64::EPSILON), 1);
        assert_eq!(ulp_distance_f64(-2.0, -2.0), 0);
    }

    #[test]
    fn bucket_labels() {
        assert_eq!(ulp_bucket_label(0), "0");
        assert_eq!(ulp_bucket_label(1), "1");
        assert_eq!(ulp_bucket_label(2), "2-3");
        assert_eq!(ulp_bucket_label(ULP_NBUCKETS - 1), ">=4096");
    }

    #[test]
    fn c_style_formats() {
        assert_eq!(sci(2.45507e-7, 6), "2.455070e-07");
        assert_eq!(sci(0.0, 6), "0.000000e+00");
        assert_eq!(sci(-1.5e10, 6), "-1.500000e+10");
        assert_eq!(gfmt(1.05), "1.05");
        assert_eq!(gfmt(1.0), "1");
        assert_eq!(gfmt(0.000123456789), "0.000123457");
    }
}
