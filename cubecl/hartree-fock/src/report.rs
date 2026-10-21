//! Run results and every output format. The benchmark fills a `RunReport`; the
//! text report and both CSV modes are printed from that one struct, so all modes
//! always show the same numbers.
//!
//! Streams: in text mode everything goes to stdout. In the CSV modes only CSV
//! goes to stdout; the header, energy and correctness sections go to stderr.

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
    /// "native" or "legacy-sqrtf".
    pub sqrt_mode: &'static str,
    /// Input file name (no directory).
    pub input: String,
    pub natoms: usize,
    pub ngauss: usize,
    pub quartets: u64,
    pub block_size: u32,
    pub cubes: u32,
    pub num_times: u32,
    pub timing: String,
    pub energy_enabled: bool,
    /// Warnings about the configuration, printed under the header.
    pub notes: Vec<String>,
}

pub struct Routine {
    pub name: &'static str,
    /// One entry per timed launch.
    pub times: Vec<f64>,
}

impl Routine {
    pub fn min(&self) -> f64 {
        self.times.iter().copied().fold(f64::INFINITY, f64::min)
    }
    pub fn max(&self) -> f64 {
        self.times.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }
    pub fn avg(&self) -> f64 {
        self.times.iter().sum::<f64>() / self.times.len() as f64
    }
}

pub struct EnergyReport {
    pub total_j: f64,
    pub samples: u32,
    pub time_s: f64,
    pub iterations: u32,
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

/// Comparison against the fp64 CPU reference Fock build.
pub struct ReferenceCheck {
    /// 2e- energy from the device Fock matrix, reduced on the host in fp64.
    pub e_fock_fp64_reduce: f64,
    pub e_ref: f64,
    /// Relative energy error from the Fock build alone.
    pub rel_err_fock: f64,
    /// Relative energy error including the in-precision reduction.
    pub rel_err_total: f64,
    pub max_abs_diff: f64,
    pub max_rel_diff: f64,
    pub rms_diff: f64,
    pub max_asymmetry: f64,
    /// Diagnostic: energy from the symmetrized Fock matrix.
    pub e_symmetrized: f64,
}

pub struct Correctness {
    /// 2e- energy from the device Fock matrix, reduced in the build precision.
    pub e2: f64,
    pub reference: Option<ReferenceCheck>,
    /// Why the reference check did not run, when it didn't.
    pub skipped: Option<String>,
}

pub struct RunReport {
    pub meta: Meta,
    pub fock: Routine,
    pub energy: Option<EnergyReport>,
    pub correctness: Correctness,
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
            write_energy(report, &mut diag)?;
            write_correctness(report, &mut diag)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Text report
// ---------------------------------------------------------------------------

fn thousands(n: u64) -> String {
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

fn kv(out: &mut dyn Write, key: &str, value: impl std::fmt::Display) -> io::Result<()> {
    writeln!(out, "  {:<13}{}", key, value)
}

/// Key/value with a wider key column, for the correctness section.
fn kv_wide(out: &mut dyn Write, key: &str, value: impl std::fmt::Display) -> io::Result<()> {
    writeln!(out, "  {:<34}{}", key, value)
}

fn note(out: &mut dyn Write, text: &str) -> io::Result<()> {
    writeln!(out, "  note: {}", text)
}

fn write_header(m: &Meta, out: &mut dyn Write) -> io::Result<()> {
    writeln!(out, "Hartree-Fock ({})", IMPLEMENTATION)?;
    kv(out, "Backend", m.backend)?;
    kv(out, "Device", &m.device)?;
    kv(out, "Driver", &m.driver)?;
    kv(out, "Precision", m.precision)?;
    kv(out, "Sqrt mode", m.sqrt_mode)?;
    kv(
        out,
        "Input",
        format!("{} ({} atoms, {} gaussians)", m.input, m.natoms, m.ngauss),
    )?;
    kv(out, "Quartets", format!("{} (one unit each)", thousands(m.quartets)))?;
    kv(out, "Block size", m.block_size)?;
    kv(out, "Cubes", thousands(m.cubes as u64))?;
    kv(out, "Iterations", format!("{} timed, after 1 untimed run", m.num_times))?;
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

fn write_results(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    let k = &r.fock;
    writeln!(
        out,
        "{:<12}{:>14}{:>14}{:>14}",
        "Function", "Min (s)", "Max (s)", "Avg (s)"
    )?;
    writeln!(
        out,
        "{:<12}{:>14.8}{:>14.8}{:>14.8}",
        k.name,
        k.min(),
        k.max(),
        k.avg()
    )?;
    writeln!(out)
}

fn write_energy(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    let Some(e) = &r.energy else {
        writeln!(out, "{:<10}disabled (--no-energy)", "Energy")?;
        return writeln!(out);
    };
    writeln!(
        out,
        "{:<10}{:>12}{:>14}{:>16}{:>12}{:>13}{:>10}",
        "Energy", "J/iter", "Total (Wh)", "Avg power (W)", "Time (s)", "Iterations", "Samples"
    )?;
    writeln!(
        out,
        "{:<10}{:>12.6}{:>14.6}{:>16.2}{:>12.6}{:>13}{:>10}",
        "board",
        e.per_iter_j(),
        e.total_wh(),
        e.avg_power_w(),
        e.time_s,
        e.iterations,
        e.samples
    )?;
    if e.samples < 2 {
        note(
            out,
            &format!(
                "only {} power sample(s); energy integration needs at least two. \
                 Increase --numtimes or use a larger input.",
                e.samples
            ),
        )?;
    }
    writeln!(out)
}

fn write_correctness(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    let c = &r.correctness;
    let p = r.meta.precision;
    writeln!(out, "Correctness ({}, sqrt: {})", p, r.meta.sqrt_mode)?;
    kv_wide(out, &format!("2e- energy ({} reduction)", p), format!("{:.16}", c.e2))?;
    match &c.reference {
        Some(x) => {
            kv_wide(out, "2e- energy (fp64 reduction)", format!("{:.16}", x.e_fock_fp64_reduce))?;
            kv_wide(out, "2e- energy (fp64 CPU reference)", format!("{:.16}", x.e_ref))?;
            kv_wide(out, "Rel err, Fock build only", format!("{:.6e}", x.rel_err_fock))?;
            kv_wide(out, &format!("Rel err, incl. {} reduction", p), format!("{:.6e}", x.rel_err_total))?;
            kv_wide(
                out,
                "Max |F - F_ref|",
                format!("{:.6e}  (rel {:.6e})", x.max_abs_diff, x.max_rel_diff),
            )?;
            kv_wide(out, "RMS |F - F_ref|", format!("{:.6e}", x.rms_diff))?;
            kv_wide(out, "Max |F(i,j) - F(j,i)|", format!("{:.6e}", x.max_asymmetry))?;
            kv_wide(
                out,
                "2e- energy, symmetrized F",
                format!("{:.16}  (diagnostic)", x.e_symmetrized),
            )?;
        }
        None => {
            if let Some(why) = &c.skipped {
                note(out, why)?;
            }
        }
    }
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
        "{},{},{},{},{},{},{},{},{}",
        IMPLEMENTATION,
        m.backend,
        csv_field(&m.device),
        m.precision,
        m.sqrt_mode,
        csv_field(&m.input),
        m.natoms,
        m.ngauss,
        m.block_size
    )
}

fn write_csv_summary(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    let m = &r.meta;
    writeln!(
        out,
        "implementation,backend,device,precision,sqrt_mode,input,natoms,ngauss,block_size,\
         num_times,routine,min_s,max_s,avg_s,e2_energy,rel_err_ref,energy_j_per_iter,avg_power_w"
    )?;
    let (energy_j, power_w) = match &r.energy {
        Some(e) => (format!("{:.6}", e.per_iter_j()), format!("{:.3}", e.avg_power_w())),
        None => (String::new(), String::new()),
    };
    let rel_err = r
        .correctness
        .reference
        .as_ref()
        .map(|x| format!("{:.6e}", x.rel_err_total))
        .unwrap_or_default();
    let k = &r.fock;
    writeln!(
        out,
        "{},{},{},{:.9},{:.9},{:.9},{:.16},{},{},{}",
        csv_prefix(m),
        m.num_times,
        k.name,
        k.min(),
        k.max(),
        k.avg(),
        r.correctness.e2,
        rel_err,
        energy_j,
        power_w
    )
}

fn write_csv_raw(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "implementation,backend,device,precision,sqrt_mode,input,natoms,ngauss,block_size,\
         routine,iteration,time_s"
    )?;
    let prefix = csv_prefix(&r.meta);
    for (i, t) in r.fock.times.iter().enumerate() {
        // The untimed first run is not recorded, so timed iterations start at 1.
        writeln!(out, "{},{},{},{:.9}", prefix, r.fock.name, i + 1, t)?;
    }
    Ok(())
}
