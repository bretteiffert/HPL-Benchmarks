//! Run results and every output format. The benchmark fills a `RunReport`; the
//! text report and both CSV modes are printed from that one struct, so all modes
//! always show the same numbers. One run sweeps several (ppwi, wgsize)
//! configurations, so each table has one row per configuration.
//!
//! Streams: in text mode everything goes to stdout. In the CSV modes only CSV
//! goes to stdout; the header, energy and correctness sections go to stderr.

use crate::metrics::{ulp_bucket_label, Metrics, REPORT_TOLERANCE_PCT, ULP_NBUCKETS};
use std::io::{self, Write};

pub const IMPLEMENTATION: &str = "CubeCL";

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Text,
    Csv,
    CsvRaw,
}

pub struct Meta {
    pub backend: &'static str,
    pub device: String,
    pub driver: String,
    pub precision: &'static str,
    /// Whether an fp64 reference pass was run for comparison.
    pub differential: bool,
    /// Deck name (last path component).
    pub deck: String,
    pub natlig: usize,
    pub natpro: usize,
    pub nposes: usize,
    pub max_poses: usize,
    pub ppwis: Vec<usize>,
    pub wgsizes: Vec<usize>,
    pub iterations: usize,
    pub warmup: usize,
    pub timing: String,
    pub energy_enabled: bool,
    /// Warnings about the configuration, printed under the header.
    pub notes: Vec<String>,
}

pub struct EnergyReport {
    pub total_j: f64,
    pub samples: u32,
    pub time_s: f64,
    pub iterations: usize,
}

impl EnergyReport {
    pub fn per_iter_j(&self) -> f64 {
        if self.iterations > 0 {
            self.total_j / self.iterations as f64
        } else {
            0.0
        }
    }
    pub fn avg_power_w(&self) -> f64 {
        if self.time_s > 0.0 {
            self.total_j / self.time_s
        } else {
            0.0
        }
    }
    pub fn total_wh(&self) -> f64 {
        self.total_j / 3600.0
    }
}

pub struct ConfigResult {
    pub ppwi: usize,
    pub wgsize: usize,
    /// Kernel time per measured iteration, seconds.
    pub times: Vec<f64>,
    pub gflops: f64,
    pub gfinsts: f64,
    pub interactions_per_s: f64,
    pub launch_overhead_s: f64,
    /// Deck upload time, seconds.
    pub context_s: f64,
    pub energy: Option<EnergyReport>,
    /// Against the fp64 reference pass; `None` when the run is fp64 itself.
    pub vs_reference: Option<Metrics>,
    pub vs_deck: Metrics,
}

impl ConfigResult {
    pub fn min(&self) -> f64 {
        self.times.iter().copied().fold(f64::INFINITY, f64::min)
    }
    pub fn max(&self) -> f64 {
        self.times.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }
    pub fn avg(&self) -> f64 {
        self.times.iter().sum::<f64>() / self.times.len() as f64
    }
    /// Population standard deviation, as upstream miniBUDE reports it.
    pub fn stddev(&self) -> f64 {
        let mean = self.avg();
        let var = self.times.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / self.times.len() as f64;
        var.sqrt()
    }
}

pub struct RunReport {
    pub meta: Meta,
    pub configs: Vec<ConfigResult>,
    /// The first `--rows` pose energies, from the first configuration.
    pub energies_head: Vec<f64>,
}

impl RunReport {
    /// Index of the configuration with the lowest average kernel time.
    pub fn best(&self) -> usize {
        (0..self.configs.len())
            .min_by(|&a, &b| self.configs[a].avg().total_cmp(&self.configs[b].avg()))
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Printed before the run starts, so the configuration is visible while it runs.
pub fn emit_header(meta: &Meta, mode: OutputMode) {
    let r = match mode {
        OutputMode::Text => write_header(meta, &mut io::stdout().lock()),
        OutputMode::Csv | OutputMode::CsvRaw => write_header(meta, &mut io::stderr().lock()),
    };
    r.expect("failed to write output");
}

/// Printed after the run.
pub fn emit_results(report: &RunReport, mode: OutputMode) {
    try_emit_results(report, mode).expect("failed to write output");
}

fn try_emit_results(report: &RunReport, mode: OutputMode) -> io::Result<()> {
    let stdout = io::stdout();
    let mut data = stdout.lock();
    match mode {
        OutputMode::Text => {
            write_results(report, &mut data)?;
            write_energy(report, &mut data)?;
            write_correctness(report, &mut data)?;
        }
        OutputMode::Csv | OutputMode::CsvRaw => {
            if mode == OutputMode::Csv {
                write_csv_summary(report, &mut data)?;
            } else {
                write_csv_raw(report, &mut data)?;
            }
            data.flush()?;
            let stderr = io::stderr();
            let mut diag = stderr.lock();
            write_best(report, &mut diag)?;
            writeln!(diag)?;
            write_energy(report, &mut diag)?;
            write_correctness(report, &mut diag)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Text report
// ---------------------------------------------------------------------------

fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn joined(v: &[usize]) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
}

fn kv(out: &mut dyn Write, key: &str, value: impl std::fmt::Display) -> io::Result<()> {
    writeln!(out, "  {:<13}{}", key, value)
}

fn note(out: &mut dyn Write, text: &str) -> io::Result<()> {
    writeln!(out, "  note: {}", text)
}

fn write_header(m: &Meta, out: &mut dyn Write) -> io::Result<()> {
    writeln!(out, "miniBUDE ({})", IMPLEMENTATION)?;
    kv(out, "Backend", m.backend)?;
    kv(out, "Device", &m.device)?;
    kv(out, "Driver", &m.driver)?;
    kv(
        out,
        "Precision",
        if m.differential {
            format!("{} (plus an fp64 reference pass for comparison)", m.precision)
        } else {
            m.precision.to_string()
        },
    )?;
    kv(
        out,
        "Deck",
        format!(
            "{} ({} ligand atoms, {} protein atoms, {} of {} poses)",
            m.deck,
            m.natlig,
            m.natpro,
            thousands(m.nposes),
            thousands(m.max_poses)
        ),
    )?;
    kv(out, "PPWI", joined(&m.ppwis))?;
    kv(out, "Work-group", joined(&m.wgsizes))?;
    kv(
        out,
        "Iterations",
        format!("{} timed per configuration, after {} warm-up", m.iterations, m.warmup),
    )?;
    kv(out, "Timing", &m.timing)?;
    kv(
        out,
        "Energy",
        if m.energy_enabled {
            "libkernelenergy (board power)"
        } else {
            "disabled (--no-energy)"
        },
    )?;
    for n in &m.notes {
        note(out, n)?;
    }
    writeln!(out)
}

fn write_best(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    if r.configs.len() > 1 {
        let b = &r.configs[r.best()];
        note(
            out,
            &format!(
                "best: ppwi {}, wgsize {} ({:.8} s average, {:.3} GFLOP/s)",
                b.ppwi,
                b.wgsize,
                b.avg(),
                b.gflops
            ),
        )?;
    }
    Ok(())
}

fn write_results(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "{:<6}{:<6}{:>14}{:>14}{:>14}{:>14}{:>12}{:>12}",
        "PPWI", "WG", "Min (s)", "Max (s)", "Avg (s)", "Stddev (s)", "GFLOP/s", "GInter/s"
    )?;
    for c in &r.configs {
        writeln!(
            out,
            "{:<6}{:<6}{:>14.8}{:>14.8}{:>14.8}{:>14.8}{:>12.3}{:>12.3}",
            c.ppwi,
            c.wgsize,
            c.min(),
            c.max(),
            c.avg(),
            c.stddev(),
            c.gflops,
            c.interactions_per_s / 1e9
        )?;
    }
    write_best(r, out)?;
    writeln!(out)
}

fn write_energy(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    if !r.meta.energy_enabled {
        writeln!(out, "{:<10}disabled (--no-energy)", "Energy")?;
        return writeln!(out);
    }
    writeln!(
        out,
        "{:<8}{:<6}{:<6}{:>12}{:>14}{:>16}{:>12}{:>13}{:>10}",
        "Energy", "PPWI", "WG", "J/iter", "Total (Wh)", "Avg power (W)", "Time (s)", "Iterations", "Samples"
    )?;
    let mut low = Vec::new();
    for c in &r.configs {
        let Some(e) = &c.energy else { continue };
        writeln!(
            out,
            "{:<8}{:<6}{:<6}{:>12.6}{:>14.6}{:>16.2}{:>12.6}{:>13}{:>10}",
            "board",
            c.ppwi,
            c.wgsize,
            e.per_iter_j(),
            e.total_wh(),
            e.avg_power_w(),
            e.time_s,
            e.iterations,
            e.samples
        )?;
        if e.samples < 2 {
            low.push(format!("{}/{}", c.ppwi, c.wgsize));
        }
    }
    if !low.is_empty() {
        let which = if low.len() == r.configs.len() {
            "every configuration".to_string()
        } else {
            format!("ppwi/wgsize {}", low.join(", "))
        };
        note(
            out,
            &format!(
                "fewer than two power samples for {}; energy integration needs at least two. \
                 Increase --iter or use more poses.",
                which
            ),
        )?;
    }
    writeln!(out)
}

fn write_correctness(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    let m = &r.meta;
    writeln!(out, "Correctness ({})", m.precision)?;

    if let Some(first) = r.configs.first() {
        writeln!(
            out,
            "  Pose energies, first {} (ppwi {}, wgsize {}):",
            r.energies_head.len(),
            first.ppwi,
            first.wgsize
        )?;
        for (i, e) in r.energies_head.iter().enumerate() {
            writeln!(out, "    {:>6}  {:.6}", i, e)?;
        }
    }

    writeln!(out)?;
    if m.differential {
        writeln!(out, "  vs fp64 reference pass")?;
        writeln!(
            out,
            "  {:<6}{:<6}{:>14}{:>14}{:>14}{:>14}{:>10}{:>12}",
            "PPWI", "WG", "L2 rel", "RMS abs", "Max abs", "Max rel", "Max ULP", "Non-finite"
        )?;
        for c in &r.configs {
            let Some(x) = &c.vs_reference else { continue };
            writeln!(
                out,
                "  {:<6}{:<6}{:>14.6e}{:>14.6e}{:>14.6e}{:>14.6e}{:>10}{:>12}",
                c.ppwi, c.wgsize, x.l2_rel_norm, x.rms_abs, x.max_abs, x.max_rel, x.max_ulp, x.non_finite
            )?;
        }
        // Configurations usually produce identical energies, so print each
        // distinct histogram once with the configurations that share it.
        let mut groups: Vec<([usize; ULP_NBUCKETS], Vec<String>)> = Vec::new();
        for c in &r.configs {
            let Some(x) = &c.vs_reference else { continue };
            let label = format!("{}/{}", c.ppwi, c.wgsize);
            match groups.iter_mut().find(|(h, _)| *h == x.ulp_histogram) {
                Some((_, labels)) => labels.push(label),
                None => groups.push((x.ulp_histogram, vec![label])),
            }
        }
        for (hist, labels) in &groups {
            let buckets: Vec<String> = (0..ULP_NBUCKETS)
                .filter(|&i| hist[i] > 0)
                .map(|i| format!("{}: {}", ulp_bucket_label(i), hist[i]))
                .collect();
            let which = if groups.len() == 1 {
                "all configurations".to_string()
            } else {
                format!("ppwi/wgsize {}", labels.join(", "))
            };
            note(out, &format!("ULP histogram ({}): {}", which, buckets.join(", ")))?;
        }
    } else {
        note(out, "float64 is the reference precision, so there is no differential pass")?;
    }

    writeln!(out)?;
    writeln!(out, "  vs deck reference energies")?;
    writeln!(
        out,
        "  {:<6}{:<6}{:>14}{:>14}{:>18}",
        "PPWI",
        "WG",
        "L2 rel",
        "Max rel",
        format!("Beyond {}%", REPORT_TOLERANCE_PCT)
    )?;
    for c in &r.configs {
        let x = &c.vs_deck;
        writeln!(
            out,
            "  {:<6}{:<6}{:>14.6e}{:>14.6e}{:>18}",
            c.ppwi,
            c.wgsize,
            x.l2_rel_norm,
            x.max_rel,
            format!("{}/{}", x.beyond_tolerance, x.compared)
        )?;
    }
    note(out, "informational: ref_energies.out is stored at low precision")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn csv_prefix(m: &Meta) -> String {
    format!(
        "{},{},{},{},{},{},{},{}",
        IMPLEMENTATION,
        m.backend,
        csv_field(&m.device),
        m.precision,
        csv_field(&m.deck),
        m.natlig,
        m.natpro,
        m.nposes
    )
}

fn write_csv_summary(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    let m = &r.meta;
    writeln!(
        out,
        "implementation,backend,device,precision,deck,natlig,natpro,nposes,ppwi,wgsize,iterations,\
         min_s,max_s,avg_s,stddev_s,gflops,gfinsts,ginteractions,launch_overhead_s,context_s,\
         energy_j_per_iter,avg_power_w,ref_l2_rel,ref_max_rel,ref_max_ulp,deck_l2_rel,deck_beyond_tol"
    )?;
    let prefix = csv_prefix(m);
    for c in &r.configs {
        let (energy_j, power_w) = match &c.energy {
            Some(e) => (format!("{:.6}", e.per_iter_j()), format!("{:.3}", e.avg_power_w())),
            None => (String::new(), String::new()),
        };
        let (ref_l2, ref_rel, ref_ulp) = match &c.vs_reference {
            Some(x) => (
                format!("{:.6e}", x.l2_rel_norm),
                format!("{:.6e}", x.max_rel),
                x.max_ulp.to_string(),
            ),
            None => (String::new(), String::new(), String::new()),
        };
        writeln!(
            out,
            "{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.3},{:.3},{:.6},{:.9},{:.6},{},{},{},{},{},{:.6e},{}",
            prefix,
            c.ppwi,
            c.wgsize,
            m.iterations,
            c.min(),
            c.max(),
            c.avg(),
            c.stddev(),
            c.gflops,
            c.gfinsts,
            c.interactions_per_s / 1e9,
            c.launch_overhead_s,
            c.context_s,
            energy_j,
            power_w,
            ref_l2,
            ref_rel,
            ref_ulp,
            c.vs_deck.l2_rel_norm,
            c.vs_deck.beyond_tolerance
        )?;
    }
    Ok(())
}

fn write_csv_raw(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "implementation,backend,device,precision,deck,natlig,natpro,nposes,ppwi,wgsize,iteration,time_s"
    )?;
    let prefix = csv_prefix(&r.meta);
    for c in &r.configs {
        for (i, t) in c.times.iter().enumerate() {
            // Warm-up runs are not recorded, so timed iterations start at 1.
            writeln!(out, "{},{},{},{},{:.9}", prefix, c.ppwi, c.wgsize, i + 1, t)?;
        }
    }
    Ok(())
}
