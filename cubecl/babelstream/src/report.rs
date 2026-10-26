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
    pub elem_bytes: usize,
    pub array_size: usize,
    pub block_size: u32,
    pub dot_groups: u32,
    pub dot_groups_detail: String,
    pub num_times: u32,
    pub use_fma: bool,
    pub timing: String,
    pub energy_enabled: bool,
    pub mibibytes: bool,
    /// Warnings about the configuration, printed under the header.
    pub notes: Vec<String>,
}

pub struct Routine {
    pub name: &'static str,
    /// Bytes moved per launch, for bandwidth.
    pub bytes: usize,
    /// One entry per measured launch (BabelStream's iteration 0 is already dropped).
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

pub struct ArrayCheck {
    pub name: &'static str,
    pub expected: f64,
    pub checksum_rel_err: f64,
    /// `None` for the dot product, which has no per-element error.
    pub max_elem_rel_err: Option<f64>,
}

pub struct Correctness {
    pub rows: Vec<ArrayCheck>,
    pub actual_dot: f64,
    /// Expected per-element product when it underflows the precision.
    pub dot_underflow: Option<f64>,
    /// Expected dot product when it exceeds the precision's finite range.
    pub dot_overflow: Option<f64>,
}

pub struct RunReport {
    pub meta: Meta,
    pub init: Routine,
    pub routines: Vec<Routine>,
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

fn bw_scale_and_unit(mibibytes: bool) -> (f64, &'static str) {
    if mibibytes {
        (2f64.powi(-20), "MiB/s")
    } else {
        (1.0e-9, "GB/s")
    }
}

fn size_string(bytes: usize, mibibytes: bool) -> String {
    if mibibytes {
        format!("{:.1} MiB", bytes as f64 * 2f64.powi(-20))
    } else {
        format!("{:.1} MB", bytes as f64 * 1.0e-6)
    }
}

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

fn kv(out: &mut dyn Write, key: &str, value: impl std::fmt::Display) -> io::Result<()> {
    writeln!(out, "  {:<13}{}", key, value)
}

fn note(out: &mut dyn Write, text: &str) -> io::Result<()> {
    writeln!(out, "  note: {}", text)
}

fn write_header(m: &Meta, out: &mut dyn Write) -> io::Result<()> {
    let array_bytes = m.array_size * m.elem_bytes;
    writeln!(out, "BabelStream ({})", IMPLEMENTATION)?;
    kv(out, "Backend", m.backend)?;
    kv(out, "Device", &m.device)?;
    kv(out, "Driver", &m.driver)?;
    kv(out, "Precision", m.precision)?;
    kv(
        out,
        "Array size",
        format!(
            "{} elements ({} per array, {} total)",
            thousands(m.array_size),
            size_string(array_bytes, m.mibibytes),
            size_string(3 * array_bytes, m.mibibytes)
        ),
    )?;
    kv(out, "Block size", m.block_size)?;
    kv(out, "Dot group size", m.block_size)?;
    kv(out, "Dot groups", format!("{} ({})", m.dot_groups, m.dot_groups_detail))?;
    kv(
        out,
        "Iterations",
        format!("{} (first excluded from statistics)", m.num_times),
    )?;
    kv(out, "FMAD", if m.use_fma { "1" } else { "0" })?;
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
    let (scale, unit) = bw_scale_and_unit(r.meta.mibibytes);
    writeln!(
        out,
        "{:<10}{:>18}{:>14}{:>14}{:>14}",
        "Function",
        format!("Bandwidth ({})", unit),
        "Min (s)",
        "Max (s)",
        "Avg (s)"
    )?;
    // Init runs once, so it has a single time and no spread.
    let t = r.init.min();
    writeln!(
        out,
        "{:<10}{:>18.3}{:>14.8}{:>14}{:>14}",
        r.init.name,
        scale * r.init.bytes as f64 / t,
        t,
        "—",
        "—"
    )?;
    for k in &r.routines {
        let tmin = k.min();
        writeln!(
            out,
            "{:<10}{:>18.3}{:>14.8}{:>14.8}{:>14.8}",
            k.name,
            scale * k.bytes as f64 / tmin,
            tmin,
            k.max(),
            k.avg()
        )?;
    }
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
                 Increase --numtimes or use a larger --arraysize.",
                e.samples
            ),
        )?;
    }
    writeln!(out)
}

fn write_correctness(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    let c = &r.correctness;
    writeln!(
        out,
        "Correctness ({}, {} iterations)",
        r.meta.precision, r.meta.num_times
    )?;
    writeln!(
        out,
        "{:<10}{:>16}{:>20}{:>20}",
        "Array", "Expected", "Checksum rel err", "Max elem rel err"
    )?;
    for row in &c.rows {
        let max = row
            .max_elem_rel_err
            .map(|v| format!("{:.3e}", v))
            .unwrap_or_else(|| "—".to_string());
        writeln!(
            out,
            "{:<10}{:>16.6e}{:>20.3e}{:>20}",
            row.name, row.expected, row.checksum_rel_err, max
        )?;
    }
    note(out, &format!("actual dot = {:.9e}", c.actual_dot))?;
    if let Some(p) = c.dot_underflow {
        note(
            out,
            &format!(
                "expected element product {:.3e} underflows {} at this iteration count, \
                 so the dot relative error is not meaningful",
                p, r.meta.precision
            ),
        )?;
    }
    if let Some(d) = c.dot_overflow {
        note(
            out,
            &format!(
                "expected dot {:.3e} exceeds the {} range, so the device result overflows \
                 and the dot relative error is not meaningful; use a smaller --arraysize",
                d, r.meta.precision
            ),
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CSV (bandwidth is always GB/s here, whatever --mibibytes says)
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
        "{},{},{},{},{},{}",
        IMPLEMENTATION,
        m.backend,
        csv_field(&m.device),
        m.precision,
        m.array_size,
        m.block_size
    )
}

fn write_csv_summary(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    let m = &r.meta;
    writeln!(
        out,
        "implementation,backend,device,precision,array_size,block_size,dot_group_size,dot_groups,num_times,\
         routine,bw_gbs,min_s,max_s,avg_s,energy_j_per_iter,avg_power_w"
    )?;
    let prefix = csv_prefix(m);
    let (energy_j, power_w) = match &r.energy {
        Some(e) => (format!("{:.6}", e.per_iter_j()), format!("{:.3}", e.avg_power_w())),
        None => (String::new(), String::new()),
    };
    // Init runs once and outside the energy window: no max/avg, no energy.
    let t = r.init.min();
    writeln!(
        out,
        "{},{},{},{},{},{:.4},{:.9},,,,",
        prefix,
        m.block_size,
        m.dot_groups,
        m.num_times,
        r.init.name,
        1.0e-9 * r.init.bytes as f64 / t,
        t
    )?;
    for k in &r.routines {
        let tmin = k.min();
        writeln!(
            out,
            "{},{},{},{},{},{:.4},{:.9},{:.9},{:.9},{},{}",
            prefix,
            m.block_size,
            m.dot_groups,
            m.num_times,
            k.name,
            1.0e-9 * k.bytes as f64 / tmin,
            tmin,
            k.max(),
            k.avg(),
            energy_j,
            power_w
        )?;
    }
    Ok(())
}

fn write_csv_raw(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "implementation,backend,device,precision,array_size,block_size,dot_group_size,dot_groups,routine,iteration,time_s,bw_gbs"
    )?;
    let prefix = csv_prefix(&r.meta);
    for k in &r.routines {
        for (i, t) in k.times.iter().enumerate() {
            // Iteration 0 is the unmeasured warm-up, so measured ones start at 1.
            writeln!(
                out,
                "{},{},{},{},{},{:.9},{:.4}",
                prefix,
                r.meta.block_size,
                r.meta.dot_groups,
                k.name,
                i + 1,
                t,
                1.0e-9 * k.bytes as f64 / t
            )?;
        }
    }
    Ok(())
}
