//! CubeCL BabelStream benchmark with build-time CUDA or HIP backend selection
//! (`--features cuda` or `--features hip`).
//!
//! - Precision: float64 by default, float32 with `--float`, float16 with `--half`.
//! - Timing: each kernel runs under `Client::profile()`. On the CUDA and HIP
//!   runtimes of CubeCL 0.10 this uses device events; the method actually used
//!   is printed in the header. Dot timings include reading back the per-group
//!   partial sums.
//! - Dot grid: 4 × the SM/CU count queried from the device, else a warned
//!   fallback of 64.
//! - The FMAD environment value specializes two equivalent plain multiply-add
//!   branches, so actual contraction remains backend-dependent.

mod backend;
mod cli;
mod energy;
mod kernels;
mod report;
mod stream;

use backend::{BACKEND_LABEL, COMPUTE_UNIT_LABEL};
use cli::{Args, Precision};
use energy::EnergyProfiler;
use report::{ArrayCheck, Correctness, EnergyReport, Meta, Routine, RunReport};
use std::time::Instant;
use stream::{CubeStream, Elem, START_A, START_B, START_C, START_SCALAR};

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

/// Checks the block size against the device and the dot kernel's limits.
fn validate_block_size(args: &Args, max_units: u32) {
    let bs = args.block_size;
    if bs > max_units {
        fail(&format!(
            "--block-size {} exceeds the device limit of {} units per cube",
            bs, max_units
        ));
    }
    if args.array_size % bs != 0 {
        fail(&format!(
            "array size {} must be a multiple of the block size ({})",
            args.array_size, bs
        ));
    }
}

/// Dot grid = 4 × compute units. Missing hardware metadata is a hard error in
/// the pinned configuration.
fn dot_groups(compute_units: Option<u32>) -> (u32, String) {
    if let Some(n) = compute_units.filter(|&n| n > 0) {
        return (n * 4, format!("4 × {} {}", n, COMPUTE_UNIT_LABEL));
    }
    fail(&format!(
        "{} count unavailable; the pinned Dot configuration requires 4 × SM/CU groups",
        COMPUTE_UNIT_LABEL
    ))
}

fn kahan_sum<F: Elem>(v: &[F]) -> f64 {
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for x in v {
        let y = x.to_host_f64() - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Returns (checksum relative error, max element relative error).
fn metrics<F: Elem>(v: &[F], gold: F) -> (f64, f64) {
    let gold_d = gold.to_host_f64();
    let actual_sum = kahan_sum(v);
    let expected_sum = gold_d * v.len() as f64;

    let checksum_rel_err = if expected_sum != 0.0 {
        (actual_sum - expected_sum).abs() / expected_sum.abs()
    } else {
        actual_sum.abs()
    };

    let mut max_err = 0.0f64;
    for x in v {
        let abs_err = (x.to_host_f64() - gold_d).abs();
        let rel_err = if gold_d != 0.0 { abs_err / gold_d.abs() } else { abs_err };
        if rel_err > max_err {
            max_err = rel_err;
        }
    }
    (checksum_rel_err, max_err)
}

fn check_solution<F: Elem>(
    ntimes: u32,
    a: &[F],
    b: &[F],
    c: &[F],
    dot_sum: f64,
    array_size: u32,
) -> Correctness {
    let mut gold_a = F::from_f64(START_A);
    let mut gold_b = F::from_f64(START_B);
    let mut gold_c = F::from_f64(START_C);
    let scalar = F::from_f64(START_SCALAR);

    for _ in 0..ntimes {
        gold_c = gold_a;
        gold_b = scalar.h_mul(gold_c);
        gold_c = gold_a.h_add(gold_b);
        gold_a = gold_b.h_add(scalar.h_mul(gold_c));
    }

    let (chk_a, max_a) = metrics(a, gold_a);
    let (chk_b, max_b) = metrics(b, gold_b);
    let (chk_c, max_c) = metrics(c, gold_c);

    let expected_dot = gold_a.to_host_f64() * gold_b.to_host_f64() * array_size as f64;
    let actual_dot = dot_sum;
    let product_magnitude = (gold_a.to_host_f64() * gold_b.to_host_f64()).abs();
    let dot_err = if expected_dot != 0.0 {
        (actual_dot - expected_dot).abs() / expected_dot.abs()
    } else {
        actual_dot.abs()
    };

    let row = |name, expected: F, chk, max| ArrayCheck {
        name,
        expected: expected.to_host_f64(),
        checksum_rel_err: chk,
        max_elem_rel_err: Some(max),
    };
    Correctness {
        rows: vec![
            row("a", gold_a, chk_a, max_a),
            row("b", gold_b, chk_b, max_b),
            row("c", gold_c, chk_c, max_c),
            ArrayCheck {
                name: "dot",
                expected: expected_dot,
                checksum_rel_err: dot_err,
                max_elem_rel_err: None,
            },
        ],
        actual_dot,
        dot_underflow: (product_magnitude < F::tiny()).then_some(product_magnitude),
        dot_overflow: (expected_dot.abs() > F::max_finite()).then_some(expected_dot),
    }
}

fn run<F: Elem>(args: &Args, use_fma: bool) {
    let n = args.array_size as usize;
    let bpe = F::elem_size();

    let device = backend::device(args.device_index)
        .unwrap_or_else(|e| fail(&format!("device {}: {}", args.device_index, e)));
    let client = backend::client(&device);
    let props = client.properties();
    let hw = &props.hardware;
    if !props.supports_type(F::as_type_native().expect("native element type")) {
        fail(&format!("{} is not supported on this device", F::PRECISION));
    }
    validate_block_size(args, hw.max_units_per_cube);

    let (device, name_note) = backend::resolve_name(args.device_index, &client);
    let (groups, groups_detail) = dot_groups(hw.num_streaming_multiprocessors);

    let meta = Meta {
        backend: BACKEND_LABEL,
        device,
        driver: backend::resolve_driver(),
        precision: F::PRECISION,
        elem_bytes: bpe,
        array_size: n,
        block_size: args.block_size,
        dot_groups: groups,
        dot_groups_detail: groups_detail,
        num_times: args.num_times,
        use_fma,
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

    let stream = CubeStream::<F>::new(client.clone(), args.array_size, args.block_size, groups, use_fma);

    let init = Routine {
        name: "Init",
        bytes: 3 * bpe * n,
        times: vec![stream.init_arrays()],
    };

    let mut routines = [
        ("Copy", 2 * bpe * n),
        ("Mul", 2 * bpe * n),
        ("Add", 3 * bpe * n),
        ("Triad", 3 * bpe * n),
        ("Dot", 2 * bpe * n),
    ]
    .map(|(name, bytes)| Routine {
        name,
        bytes,
        times: Vec::with_capacity(args.num_times as usize),
    });

    let mut profiler = if args.energy {
        Some(EnergyProfiler::new(args.device_index).unwrap_or_else(|e| fail(&e)))
    } else {
        None
    };
    let mut measured_started: Option<Instant> = None;
    let mut dot_sum = 0.0f64;

    for iteration in 0..args.num_times {
        // BabelStream excludes iteration zero from summary statistics. Keep it as
        // an unmeasured warm-up, then sample energy continuously across all
        // measured iterations (individual runs can finish before the 5 ms sampler
        // wakes).
        if iteration == 1 {
            if let Some(e) = profiler.as_mut() {
                e.start();
            }
            measured_started = Some(Instant::now());
        }
        let copy = stream.copy();
        let mul = stream.mul();
        let add = stream.add();
        let triad = stream.triad();
        let (dot, sum) = stream.dot();
        dot_sum = sum;

        if iteration > 0 {
            for (r, t) in routines.iter_mut().zip([copy, mul, add, triad, dot]) {
                r.times.push(t);
            }
        }
    }

    let time_s = measured_started
        .map(|s| s.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let energy = profiler.as_mut().map(|e| {
        let (total_j, samples) = e.stop_and_read();
        EnergyReport {
            total_j,
            samples,
            time_s,
            iterations: args.num_times - 1,
        }
    });

    let (a, b, c) = stream.read_arrays();
    let correctness = check_solution(args.num_times, &a, &b, &c, dot_sum, args.array_size);

    let report = RunReport {
        meta,
        init,
        routines: routines.into(),
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
    let use_fma = std::env::var("FMAD").unwrap_or_else(|_| "1".to_string()) == "1";

    match args.precision {
        Precision::Float16 => run::<half::f16>(&args, use_fma),
        Precision::Float32 => run::<f32>(&args, use_fma),
        Precision::Float64 => run::<f64>(&args, use_fma),
    }
}
