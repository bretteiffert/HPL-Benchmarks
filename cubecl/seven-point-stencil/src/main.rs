//! CubeCL seven-point Laplacian benchmark with build-time CUDA or HIP backend
//! selection (`--features cuda` or `--features hip`).
//!
//! - Fills `u` with a quadratic whose Laplacian is exactly 3, runs the stencil
//!   once untimed (warm-up and correctness), then times `--numtimes` runs.
//! - Precision: float64 by default, float32 with `--float`, float16 with `--half`.
//! - Timing: each run is profiled with `Client::profile()`; on CUDA and HIP in
//!   CubeCL 0.10 this uses device events. The method used is in the header.
//! - Bandwidth uses the theoretical traffic of one Laplacian: every point except
//!   corners and edges read once, every interior point written once.

mod backend;
mod cli;
mod energy;
mod kernels;
mod report;
mod stencil;

use backend::BACKEND_LABEL;
use cli::{Args, Precision};
use energy::EnergyProfiler;
use report::{Correctness, EnergyReport, Meta, Routine, RunReport};
use std::time::Instant;
use stencil::{coefficients, Elem, Geometry, Stencil};

/// Units per cube for the (1D) initialization kernel, capped by the device.
const INIT_BLOCK: u32 = 1024;
/// ∇²u for the test function, at every interior point.
const EXPECTED_LAPLACIAN: f64 = 3.0;

fn fail(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(1);
}

fn list_devices() {
    println!("Devices ({}):", BACKEND_LABEL);
    let devices = backend::devices();
    if devices.is_empty() {
        println!("  none found");
    }
    for (i, _device) in devices.iter().enumerate() {
        println!("  {}: {}", i, backend::placeholder_name(i));
    }
}

/// Checks the block shape and both launch grids against the device limits.
fn validate_launch(geo: &Geometry, init_block: u32, max_units: u32, max_dim: (u32, u32, u32), max_count: (u32, u32, u32)) {
    let (bx, by, bz) = geo.block;
    let units = bx as u64 * by as u64 * bz as u64;
    if units > max_units as u64 {
        fail(&format!(
            "--block-size {}x{}x{} is {} units; the device allows at most {} per cube",
            bx, by, bz, units, max_units
        ));
    }
    if bx > max_dim.0 || by > max_dim.1 || bz > max_dim.2 {
        fail(&format!(
            "--block-size {}x{}x{} exceeds the device's per-axis limit {}x{}x{}",
            bx, by, bz, max_dim.0, max_dim.1, max_dim.2
        ));
    }
    let (cx, cy, cz) = geo.cubes();
    if cx > max_count.0 || cy > max_count.1 || cz > max_count.2 {
        fail(&format!(
            "the grid needs {}x{}x{} cubes with this block size; the device allows at most {}x{}x{}",
            cx, cy, cz, max_count.0, max_count.1, max_count.2
        ));
    }
    // The init kernel launches (nx / init_block) x ny x nz cubes.
    if geo.ny > max_count.1 || geo.nz > max_count.2 || geo.nx.div_ceil(init_block) > max_count.0 {
        fail(&format!(
            "grid {}x{}x{} is too large for the initialization launch on this device",
            geo.nx, geo.ny, geo.nz
        ));
    }
}

/// The stencil coefficients scale as 1/h², so a fine grid can overflow a narrow
/// precision before any data is touched.
fn validate_coefficients<F: Elem>(geo: &Geometry) {
    let (_, _, _, centre) = coefficients(geo.spacing());
    if centre.abs() > F::max_finite() {
        fail(&format!(
            "the stencil's centre coefficient {:.3e} overflows {} (max {:.3e}) on this grid; \
             use a coarser grid or a wider precision",
            centre.abs(),
            F::PRECISION,
            F::max_finite()
        ));
    }
}

fn check_correctness<F: Elem>(f: &[F], geo: &Geometry) -> Correctness {
    let (nx, ny, nz) = (geo.nx as usize, geo.ny as usize, geo.nz as usize);
    let expected = EXPECTED_LAPLACIAN;
    let mut sum_abs = 0.0f64;
    let mut sum_sq_err = 0.0f64;
    let mut max_err = 0.0f64;
    let mut count = 0usize;

    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            let row = nx * (j + ny * k);
            for v in &f[row + 1..row + nx - 1] {
                let err = (v.to_host_f64() - expected).abs();
                sum_abs += err;
                sum_sq_err += err * err;
                if err > max_err {
                    max_err = err;
                }
                count += 1;
            }
        }
    }

    // Exact solution is the constant `expected`, so ||exact||² = count · expected².
    let sum_sq_exact = count as f64 * expected * expected;
    Correctness {
        expected,
        points: count,
        max_abs_err: max_err,
        mean_abs_err: if count > 0 { sum_abs / count as f64 } else { 0.0 },
        l2_rel_err: if sum_sq_exact > 0.0 {
            (sum_sq_err / sum_sq_exact).sqrt()
        } else {
            sum_sq_err.sqrt()
        },
    }
}

fn run<F: Elem>(args: &Args) {
    let device = backend::device(args.device_index)
        .unwrap_or_else(|e| fail(&format!("device {}: {}", args.device_index, e)));
    let client = backend::client(&device);
    let props = client.properties();
    let hw = &props.hardware;
    if !props.supports_type(F::as_type_native().expect("native element type")) {
        fail(&format!("{} is not supported on this device", F::PRECISION));
    }

    let geo = Geometry {
        nx: args.grid.0,
        ny: args.grid.1,
        nz: args.grid.2,
        block: args.block,
    };
    let init_block = INIT_BLOCK.min(hw.max_units_per_cube);
    validate_launch(&geo, init_block, hw.max_units_per_cube, hw.max_cube_dim, hw.max_cube_count);
    validate_coefficients::<F>(&geo);

    let (device, name_note) = backend::resolve_name(args.device_index, &client);
    let bpe = F::elem_size();
    let meta = Meta {
        backend: BACKEND_LABEL,
        device,
        driver: backend::resolve_driver(),
        precision: F::PRECISION,
        elem_bytes: bpe,
        grid: args.grid,
        block: args.block,
        cubes: geo.cubes(),
        fetch_bytes: geo.fetch_bytes(bpe),
        write_bytes: geo.write_bytes(bpe),
        num_times: args.num_times,
        timing: match props.timing_method {
            cubecl::profile::TimingMethod::Device => "device timestamps".to_string(),
            cubecl::profile::TimingMethod::System => {
                "host clock, device synced before and after each kernel".to_string()
            }
        },
        energy_enabled: args.energy,
        mibibytes: args.mibibytes,
        notes: name_note.into_iter().collect(),
    };
    report::emit_header(&meta, args.output);

    let stencil = Stencil::<F>::new(client.clone(), geo, init_block);
    stencil.init();
    // Untimed warm-up: compiles the kernel and produces the result checked below.
    stencil.laplacian();
    let correctness = stencil.with_result(|f| check_correctness(f, &geo));

    let mut profiler = if args.energy {
        Some(EnergyProfiler::new(args.device_index).unwrap_or_else(|e| fail(&e)))
    } else {
        None
    };
    let mut laplacian = Routine {
        name: "Laplacian",
        times: Vec::with_capacity(args.num_times as usize),
    };

    if let Some(e) = profiler.as_mut() {
        e.start();
    }
    let started = Instant::now();
    for _ in 0..args.num_times {
        laplacian.times.push(stencil.timed_laplacian());
    }
    let time_s = started.elapsed().as_secs_f64();

    let energy = profiler.as_mut().map(|e| {
        let (total_j, samples) = e.stop_and_read();
        EnergyReport {
            total_j,
            samples,
            time_s,
            iterations: args.num_times,
        }
    });

    let report = RunReport {
        meta,
        laplacian,
        energy,
        correctness,
    };
    report::emit_results(&report, args.output);
}

fn main() {
    let args = cli::parse_arguments();
    if args.list {
        list_devices();
        return;
    }
    match args.precision {
        Precision::Float16 => run::<half::f16>(&args),
        Precision::Float32 => run::<f32>(&args),
        Precision::Float64 => run::<f64>(&args),
    }
}
