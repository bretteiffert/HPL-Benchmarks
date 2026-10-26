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
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub cubes: (u32, u32, u32),
    /// Theoretical bytes read and written by one Laplacian.
    pub fetch_bytes: usize,
    pub write_bytes: usize,
    pub num_times: u32,
    pub timing: String,
    pub energy_enabled: bool,
    pub mibibytes: bool,
    /// Warnings about the configuration, printed under the header.
    pub notes: Vec<String>,
}

impl Meta {
    pub fn points(&self) -> usize {
        self.grid.0 as usize * self.grid.1 as usize * self.grid.2 as usize
    }
    pub fn bytes_per_iter(&self) -> usize {
        self.fetch_bytes + self.write_bytes
    }
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

pub struct Correctness {
    pub expected: f64,
    pub points: usize,
    pub max_abs_err: f64,
    pub mean_abs_err: f64,
    pub l2_rel_err: f64,
}

pub struct RunReport {
    pub meta: Meta,
    pub laplacian: Routine,
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

fn dims(d: (u32, u32, u32)) -> String {
    format!("{} × {} × {}", d.0, d.1, d.2)
}

fn kv(out: &mut dyn Write, key: &str, value: impl std::fmt::Display) -> io::Result<()> {
    writeln!(out, "  {:<13}{}", key, value)
}

fn note(out: &mut dyn Write, text: &str) -> io::Result<()> {
    writeln!(out, "  note: {}", text)
}

fn write_header(m: &Meta, out: &mut dyn Write) -> io::Result<()> {
    let (bx, by, bz) = m.block;
    writeln!(out, "Seven-point stencil ({})", IMPLEMENTATION)?;
    kv(out, "Backend", m.backend)?;
    kv(out, "Device", &m.device)?;
    kv(out, "Driver", &m.driver)?;
    kv(out, "Precision", m.precision)?;
    kv(
        out,
        "Grid",
        format!(
            "{} ({} points, {} per array)",
            dims(m.grid),
            thousands(m.points()),
            size_string(m.points() * m.elem_bytes, m.mibibytes)
        ),
    )?;
    kv(out, "Block", format!("{} ({} units)", dims(m.block), bx * by * bz))?;
    kv(out, "Cubes", dims(m.cubes))?;
    kv(
        out,
        "Traffic",
        format!(
            "{} read + {} written per iteration (theoretical)",
            size_string(m.fetch_bytes, m.mibibytes),
            size_string(m.write_bytes, m.mibibytes)
        ),
    )?;
    kv(out, "Iterations", format!("{} timed, after 1 warm-up run", m.num_times))?;
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
    let k = &r.laplacian;
    let tmin = k.min();
    writeln!(
        out,
        "{:<10}{:>18}{:>14}{:>14}{:>14}",
        "Function",
        format!("Bandwidth ({})", unit),
        "Min (s)",
        "Max (s)",
        "Avg (s)"
    )?;
    writeln!(
        out,
        "{:<10}{:>18.3}{:>14.8}{:>14.8}{:>14.8}",
        k.name,
        scale * r.meta.bytes_per_iter() as f64 / tmin,
        tmin,
        k.max(),
        k.avg()
    )?;
    note(
        out,
        &format!(
            "bandwidth uses the fastest iteration; the average iteration gives {:.3} {}",
            scale * r.meta.bytes_per_iter() as f64 / k.avg(),
            unit
        ),
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
                 Increase --numtimes or use a larger grid.",
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
        "Correctness ({}, expected Laplacian = {} at every interior point)",
        r.meta.precision, c.expected
    )?;
    kv(out, "Points", thousands(c.points))?;
    kv(out, "Max abs err", format!("{:.3e}", c.max_abs_err))?;
    kv(out, "Mean abs err", format!("{:.3e}", c.mean_abs_err))?;
    kv(out, "Rel L2 err", format!("{:.3e}", c.l2_rel_err))?;
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
        "{},{},{},{},{},{},{},{},{},{}",
        IMPLEMENTATION,
        m.backend,
        csv_field(&m.device),
        m.precision,
        m.grid.0,
        m.grid.1,
        m.grid.2,
        m.block.0,
        m.block.1,
        m.block.2
    )
}

fn write_csv_summary(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    let m = &r.meta;
    writeln!(
        out,
        "implementation,backend,device,precision,nx,ny,nz,block_x,block_y,block_z,num_times,\
         routine,bw_gbs,min_s,max_s,avg_s,energy_j_per_iter,avg_power_w"
    )?;
    let (energy_j, power_w) = match &r.energy {
        Some(e) => (format!("{:.6}", e.per_iter_j()), format!("{:.3}", e.avg_power_w())),
        None => (String::new(), String::new()),
    };
    let k = &r.laplacian;
    let tmin = k.min();
    writeln!(
        out,
        "{},{},{},{:.4},{:.9},{:.9},{:.9},{},{}",
        csv_prefix(m),
        m.num_times,
        k.name,
        1.0e-9 * m.bytes_per_iter() as f64 / tmin,
        tmin,
        k.max(),
        k.avg(),
        energy_j,
        power_w
    )
}

fn write_csv_raw(r: &RunReport, out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "implementation,backend,device,precision,nx,ny,nz,block_x,block_y,block_z,\
         routine,iteration,time_s,bw_gbs"
    )?;
    let prefix = csv_prefix(&r.meta);
    let bytes = r.meta.bytes_per_iter() as f64;
    for (i, t) in r.laplacian.times.iter().enumerate() {
        // The warm-up run is not recorded, so timed iterations start at 1.
        writeln!(
            out,
            "{},{},{},{:.9},{:.4}",
            prefix,
            r.laplacian.name,
            i + 1,
            t,
            1.0e-9 * bytes / t
        )?;
    }
    Ok(())
}
