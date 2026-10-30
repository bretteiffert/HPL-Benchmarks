//! CubeCL Hartree-Fock two-electron Fock-build benchmark with build-time CUDA
//! or HIP backend selection (`--features cuda` or `--features hip`).
//!
//! - One kernel unit per unique (ij|kl) integral quartet, accumulating into the
//!   Fock matrix with atomic adds. One untimed run produces the Fock matrix that
//!   is checked; the timed runs that follow keep accumulating into it.
//! - Precision: float64 by default, float32 with `--float`. `--legacy-sqrt`
//!   takes the Boys-function square roots in float32 (the old behaviour).
//! - Correctness: the 2e- energy is always reported; an fp64 CPU reference
//!   Fock build is compared against it up to `--max-check-atoms` atoms (it is
//!   O(natoms⁴)), or always with `--check`.
//! - Timing: each run is profiled with `Client::profile()`; on CUDA and HIP in
//!   CubeCL 0.10 this uses device events. The method used is in the header.

mod backend;
mod cli;
mod energy;
mod hf;
mod kernels;
mod report;

use backend::BACKEND_LABEL;
use cli::{Args, Check, Precision};
use backend::{BackendRuntime, Client};
use cubecl::features::AtomicUsage;
use cubecl::ir::StorageType;
use cubecl::prelude::*;
use energy::EnergyProfiler;
use hf::{Consts, Problem, Real};
use kernels::hartree_fock_kernel;
use report::{Correctness, EnergyReport, Meta, ReferenceCheck, Routine, RunReport};
use std::time::Instant;

/// Device element types this benchmark runs in.
trait Elem: Float + CubeElement + ScalarArgSettings + Real {}
impl<T: Float + CubeElement + ScalarArgSettings + Real> Elem for T {}

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

/// Fails unless the device supports `F` and atomic adds on it.
fn validate_device<F: Elem>(client: &Client) {
    let props = client.properties();
    let native = F::as_type_native().expect("native element type");
    if !props.supports_type(native) {
        fail(&format!("{} is not supported on this device", F::NAME));
    }
    let atomic = Type::new(StorageType::Atomic(native.elem_type()));
    if !props.atomic_type_usage(atomic).contains(AtomicUsage::Add) {
        fail(&format!(
            "this device does not support atomic add on {}, which the Fock build needs",
            F::NAME
        ));
    }
}

/// Checks the launch against the 32-bit global index space and device limits.
fn validate_launch(quartets: u64, block_size: u32, client: &Client) -> u32 {
    let hw = &client.properties().hardware;
    if quartets + 1 > u32::MAX as u64 {
        fail(&format!(
            "{} quartets exceed CubeCL's 32-bit global index space (too many atoms)",
            quartets
        ));
    }
    if block_size > hw.max_units_per_cube {
        fail(&format!(
            "--block-size {} exceeds the device limit of {} units per cube",
            block_size, hw.max_units_per_cube
        ));
    }
    let cubes = (quartets + 1).div_ceil(block_size as u64) as u32;
    if cubes > hw.max_cube_count.0 {
        fail(&format!(
            "{} cubes exceed the device limit of {}; use a larger --block-size",
            cubes, hw.max_cube_count.0
        ));
    }
    cubes
}

fn reference_check<F: Elem>(p: &Problem<F>, fock_dev: &[F], e2: f64) -> ReferenceCheck {
    let n = p.natoms;
    let nsq = n * n;
    let fock_gpu: Vec<f64> = fock_dev.iter().map(|&x| x.h_to_f64()).collect();
    let fock_ref = hf::fock_build_cpu(p.ngauss, n, &p.geom_d, &p.xpnt_d, &p.coef_d, &p.dens_d);

    let e_gpu = hf::energy_from_fock(&fock_gpu, &p.dens_d, n);
    let e_ref = hf::energy_from_fock(&fock_ref, &p.dens_d, n);
    let denom = if e_ref.abs() > 0.0 { e_ref.abs() } else { 1.0 };

    let (mut maxd, mut sumsq, mut fscale) = (0.0f64, 0.0f64, 0.0f64);
    for t in 0..nsq {
        let d = (fock_gpu[t] - fock_ref[t]).abs();
        if d > maxd {
            maxd = d;
        }
        sumsq += d * d;
        if fock_ref[t].abs() > fscale {
            fscale = fock_ref[t].abs();
        }
    }
    if fscale == 0.0 {
        fscale = 1.0;
    }
    let mut asym = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let d = (fock_gpu[i * n + j] - fock_gpu[j * n + i]).abs();
            if d > asym {
                asym = d;
            }
        }
    }

    ReferenceCheck {
        e_fock_fp64_reduce: e_gpu,
        e_ref,
        rel_err_fock: (e_gpu - e_ref).abs() / denom,
        rel_err_total: (e2 - e_ref).abs() / denom,
        max_abs_diff: maxd,
        max_rel_diff: maxd / fscale,
        rms_diff: (sumsq / nsq as f64).sqrt(),
        max_asymmetry: asym,
        e_symmetrized: hf::energy_symmetrized(&fock_gpu, &p.dens_d, n),
    }
}

fn run<F: Elem>(args: &Args) {
    let path = args.input.as_deref().expect("input checked by the CLI");
    let input = hf::read_input(path).unwrap_or_else(|e| fail(&e));
    let legacy = args.legacy_sqrt;
    let c = Consts::<F>::new(legacy);
    let p = Problem::<F>::new(&input, &c);
    let (natoms, ngauss, nsq) = (p.natoms, p.ngauss, p.natoms * p.natoms);
    let nn = p.pairs();
    let nnnn = p.quartets();

    let device = backend::device(args.device_index)
        .unwrap_or_else(|e| fail(&format!("device {}: {}", args.device_index, e)));
    let client = backend::client(&device);
    validate_device::<F>(&client);
    let n_blks = validate_launch(nnnn, args.block_size, &client);

    let props = client.properties();
    let (device, name_note) = backend::resolve_name(args.device_index, &client);
    let sqrt_mode = if legacy { "legacy-sqrtf" } else { "native" };
    let meta = Meta {
        backend: BACKEND_LABEL,
        device,
        driver: backend::resolve_driver(),
        precision: F::NAME,
        sqrt_mode,
        input: std::path::Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string()),
        natoms,
        ngauss,
        quartets: nnnn,
        block_size: args.block_size,
        cubes: n_blks,
        num_times: args.num_times,
        timing: match props.timing_method {
            cubecl::profile::TimingMethod::Device => "device timestamps".to_string(),
            cubecl::profile::TimingMethod::System => {
                "host clock, device synced before and after each kernel".to_string()
            }
        },
        energy_enabled: args.energy,
        notes: name_note.into_iter().collect(),
    };
    report::emit_header(&meta, args.output);

    let d_xpnt = client.create_from_slice(F::as_bytes(&p.xpnt));
    let d_coef = client.create_from_slice(F::as_bytes(&p.coef));
    let d_geom = client.create_from_slice(F::as_bytes(&p.geom));
    let d_dens = client.create_from_slice(F::as_bytes(&p.dens));
    let d_fock = client.create_from_slice(F::as_bytes(&vec![F::h_from_f64(0.0); nsq]));
    let d_schwarz = client.create_from_slice(F::as_bytes(&p.schwarz));

    // The launch closure takes ownership of the buffer handles; keep a second
    // handle to the Fock matrix for reading it back.
    let d_fock_read = d_fock.clone();
    let (dtol, rcut, sqrtpi2) = (c.dtol, c.rcut, c.sqrtpi2_dev);
    let block_size = args.block_size;
    let launch = move |client: &Client| unsafe {
        hartree_fock_kernel::launch_unchecked::<F, BackendRuntime>(
            client,
            CubeCount::Static(n_blks, 1, 1),
            CubeDim::new_1d(block_size),
            ArrayArg::from_raw_parts(d_geom.clone(), 3 * natoms),
            ArrayArg::from_raw_parts(d_xpnt.clone(), ngauss),
            ArrayArg::from_raw_parts(d_coef.clone(), ngauss),
            ArrayArg::from_raw_parts(d_dens.clone(), nsq),
            ArrayArg::from_raw_parts(d_schwarz.clone(), nn + 1),
            ArrayArg::from_raw_parts(d_fock.clone(), nsq),
            nnnn,
            ngauss as u32,
            natoms as u32,
            dtol,
            rcut,
            sqrtpi2,
            legacy,
        );
    };

    // Untimed run: compiles the kernel and produces the Fock matrix checked below.
    launch(&client);
    let fock_bytes = client
        .read_one(d_fock_read)
        .expect("failed to read fock buffer back from device");
    let fock_dev: Vec<F> = F::from_bytes(&fock_bytes).to_vec();
    drop(fock_bytes);

    let mut erep = F::h_from_f64(0.0);
    for t in 0..nsq {
        erep += fock_dev[t] * p.dens[t];
    }
    let e2 = erep.h_to_f64() * 0.5;

    let run_check = match args.check {
        Check::Always => true,
        Check::Never => false,
        Check::Auto => natoms <= args.max_check_atoms,
    };
    let correctness = Correctness {
        e2,
        reference: run_check.then(|| reference_check(&p, &fock_dev, e2)),
        skipped: match args.check {
            Check::Never => Some("fp64 CPU reference check disabled (--no-check)".to_string()),
            Check::Auto if !run_check => Some(format!(
                "fp64 CPU reference check skipped: {} atoms > {} (--max-check-atoms); --check forces it",
                natoms, args.max_check_atoms
            )),
            _ => None,
        },
    };

    let mut profiler = if args.energy {
        Some(EnergyProfiler::new(args.device_index).unwrap_or_else(|e| fail(&e)))
    } else {
        None
    };
    let mut fock = Routine {
        name: "Fock build",
        times: Vec::with_capacity(args.num_times as usize),
    };

    if let Some(e) = profiler.as_mut() {
        e.start();
    }
    let started = Instant::now();
    for _ in 0..args.num_times {
        let inner = client.clone();
        let launch = launch.clone();
        let (_, profile) = client
            .profile(move || launch(&inner), "hartree_fock")
            .expect("cubecl profile() call failed");
        // `None` means the device recorded no measurement for this window; a
        // zero would silently win every min(), so treat it as a hard error.
        let secs = cubecl::future::block_on(profile.resolve())
            .duration()
            .as_secs_f64();
        fock.times.push(secs);
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
        fock,
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
        Precision::Float32 => run::<f32>(&args),
        Precision::Float64 => run::<f64>(&args),
    }
}
