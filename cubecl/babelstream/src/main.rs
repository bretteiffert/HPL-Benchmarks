<<<<<<< HEAD
//! CubeCL BabelStream benchmark with build-time CUDA or HIP backend selection
//! (`--features cuda` or `--features hip`).
//!
//! - Precision: float64 by default, float32 with `--float`, float16 with `--half`.
//! - Timing: each kernel runs under `Client::profile()`. On the CUDA and HIP
//!   runtimes of CubeCL 0.11 this uses device events; the method actually used
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

/// Shared memory in `dot_kernel` holds one partial per plane.
const DOT_MAX_PLANES: u32 = 32;
/// SM/CU count assumed when the device does not report one.
const FALLBACK_COMPUTE_UNITS: u32 = 64;

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
    for (i, device) in devices.iter().enumerate() {
        let name = device.client().properties().identity.name.trim().to_string();
        let name = if name.is_empty() { backend::placeholder_name(i) } else { name };
        println!("  {}: {}", i, name);
    }
}

/// Checks the block size against the device and the dot kernel's limits.
fn validate_block_size(args: &Args, plane_size: u32, max_units: u32) {
    let bs = args.block_size;
    if bs > max_units {
        fail(&format!(
            "--block-size {} exceeds the device limit of {} units per cube",
            bs, max_units
        ));
    }
    if plane_size > 0 && bs % plane_size != 0 {
        fail(&format!(
            "--block-size {} must be a multiple of the plane size ({})",
            bs, plane_size
        ));
    }
    if plane_size > 0 && bs / plane_size > DOT_MAX_PLANES {
        fail(&format!(
            "--block-size {} gives {} planes per cube; the dot kernel supports at most {}",
            bs,
            bs / plane_size,
            DOT_MAX_PLANES
        ));
    }
    if args.array_size % bs != 0 {
        fail(&format!(
            "array size {} must be a multiple of the block size ({})",
            args.array_size, bs
        ));
    }
}

/// Dot grid = 4 × compute units. Returns (groups, description, optional warning).
fn dot_groups(compute_units: Option<u32>) -> (u32, String, Option<String>) {
    if let Some(n) = compute_units.filter(|&n| n > 0) {
        return (n * 4, format!("4 × {} {}", n, COMPUTE_UNIT_LABEL), None);
    }
    (
        FALLBACK_COMPUTE_UNITS * 4,
        format!("4 × {} {}, fallback", FALLBACK_COMPUTE_UNITS, COMPUTE_UNIT_LABEL),
        Some(format!(
            "{} count unavailable from the device; using {}, so Dot bandwidth may be off",
            COMPUTE_UNIT_LABEL, FALLBACK_COMPUTE_UNITS
        )),
    )
=======
//! CubeCL BabelStream benchmark with build-time CUDA or HIP runtime selection and device-profiled kernel timing.
//! It defaults to float64, obtains optional display metadata from BS_GPU_NAME and BS_DRIVER, and uses BS_SM_COUNT or a warned fallback of 64 for the dot grid.
//! The FMAD environment value specializes two equivalent plain multiply-add branches, so actual contraction remains backend-dependent.

use cubecl::prelude::*;
use std::time::Duration;

const IMPLEMENTATION_STRING: &str = "CubeCL";

const START_A: f64 = 0.1;
const START_B: f64 = 0.2;
const START_C: f64 = 0.0;
const START_SCALAR: f64 = 0.4;

const TBSIZE: u32 = 1024;

const USE_FLOAT: bool = false;
#[cube]
fn mad_op<F: Float + CubeElement>(a: F, b: F, c: F, #[comptime] use_fma: bool) -> F {
    if use_fma {
        a * b + c
    } else {
        a * b + c
    }
}

#[cube(launch_unchecked)]
fn init_kernel<F: Float + CubeElement>(
    a: &mut Array<F>,
    b: &mut Array<F>,
    c: &mut Array<F>,
    init_a: F,
    init_b: F,
    init_c: F,
) {
    let i = ABSOLUTE_POS;
    a[i] = init_a;
    b[i] = init_b;
    c[i] = init_c;
}

#[cube(launch_unchecked)]
fn copy_kernel<F: Float + CubeElement>(a: &Array<F>, c: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    c[i] = a[i];
}

#[cube(launch_unchecked)]
fn mul_kernel<F: Float + CubeElement>(b: &mut Array<F>, c: &Array<F>, scalar: F) {
    let i = ABSOLUTE_POS;
    b[i] = scalar * c[i];
}

#[cube(launch_unchecked)]
fn add_kernel<F: Float + CubeElement>(a: &Array<F>, b: &Array<F>, c: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    c[i] = a[i] + b[i];
}

#[cube(launch_unchecked)]
fn triad_kernel<F: Float + CubeElement>(
    a: &mut Array<F>,
    b: &Array<F>,
    c: &Array<F>,
    scalar: F,
    #[comptime] use_fma: bool,
) {
    let i = ABSOLUTE_POS;
    a[i] = mad_op::<F>(scalar, c[i], b[i], use_fma);
}

#[cube(launch_unchecked)]
fn dot_kernel<F: Float + CubeElement>(
    a: &Array<F>,
    b: &Array<F>,
    partial: &mut Array<F>,
    array_size: usize,
    #[comptime] use_fma: bool,
) {
    let mut acc = F::new(0.0);

    let stride: usize = (CUBE_DIM as usize) * CUBE_COUNT;
    let mut i: usize = ABSOLUTE_POS;
    while i < array_size {
        acc = mad_op::<F>(a[i], b[i], acc, use_fma);
        i += stride;
    }

    let mut shared = SharedMemory::<F>::new(32usize);

    let plane_total = plane_sum(acc);
    let plane_id: u32 = UNIT_POS / PLANE_DIM;

    if UNIT_POS % PLANE_DIM == 0 {
        shared[plane_id as usize] = plane_total;
    }

    sync_cube();

    if UNIT_POS == 0 {
        let num_planes: u32 = CUBE_DIM / PLANE_DIM;
        let mut total = F::new(0.0);
        let mut p: u32 = 0u32;
        while p < num_planes {
            total += shared[p as usize];
            p += 1;
        }
        partial[CUBE_POS] = total;
    }
}

trait Elem: Float + CubeElement + Copy + PartialOrd + 'static {
    const IS_FLOAT32: bool;
    fn from_f64(v: f64) -> Self;
    fn to_host_f64(self) -> f64;
    fn h_add(self, o: Self) -> Self;
    fn h_mul(self, o: Self) -> Self;
    fn tiny() -> f64;
    fn elem_size() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl Elem for f32 {
    const IS_FLOAT32: bool = true;
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_host_f64(self) -> f64 {
        self as f64
    }
    fn h_add(self, o: Self) -> Self {
        self + o
    }
    fn h_mul(self, o: Self) -> Self {
        self * o
    }
    fn tiny() -> f64 {
        f32::MIN_POSITIVE as f64
    }
}

impl Elem for f64 {
    const IS_FLOAT32: bool = false;
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_host_f64(self) -> f64 {
        self
    }
    fn h_add(self, o: Self) -> Self {
        self + o
    }
    fn h_mul(self, o: Self) -> Self {
        self * o
    }
    fn tiny() -> f64 {
        f64::MIN_POSITIVE
    }
}

#[cfg(feature = "cuda")]
type Rt = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
type Rt = cubecl::hip::HipRuntime;

#[cfg(feature = "cuda")]
const BACKEND_LABEL: &str = "CUDA";
#[cfg(all(feature = "hip", not(feature = "cuda")))]
const BACKEND_LABEL: &str = "HIP";

fn profile_secs<R: Runtime, O: Send + 'static, Fun: FnOnce() -> O + Send>(
    client: &ComputeClient<R>,
    name: &'static str,
    f: Fun,
) -> (f64, O) {
    let (result, profile) = client.profile(f, name).expect("cubecl profile() call failed");
    let duration: Duration = cubecl::future::block_on(profile.resolve()).duration();
    (duration.as_secs_f64(), result)
}

struct CubeStream<F: Elem> {
    client: ComputeClient<Rt>,
    array_size: u32,
    dot_num_blocks: u32,
    use_fma: bool,

    d_a: cubecl::server::Handle,
    d_b: cubecl::server::Handle,
    d_c: cubecl::server::Handle,
    d_sum: cubecl::server::Handle,

    last_kernel_time: f64,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Elem> CubeStream<F> {
    fn new(array_size: u32, device_index: usize, use_fma: bool) -> Self {
        if array_size % TBSIZE != 0 {
            panic!("Array size must be a multiple of {}", TBSIZE);
        }

        let device = device_for_index(device_index);
        let client = Rt::client(&device);

        println!("Using {} device {}", BACKEND_LABEL, device_name(device_index));
        println!("Driver: {}", driver_string());

        let sm_count = sm_count(&client);
        let dot_num_blocks = sm_count * 4;
        println!(
            "Reduction kernel config: {} groups of (fixed) size {}",
            dot_num_blocks, TBSIZE
        );

        let bytes = array_size as usize * F::elem_size();
        let d_a = client.empty(bytes);
        let d_b = client.empty(bytes);
        let d_c = client.empty(bytes);
        let d_sum = client.empty(dot_num_blocks as usize * F::elem_size());

        let mut s = Self {
            client,
            array_size,
            dot_num_blocks,
            use_fma,
            d_a,
            d_b,
            d_c,
            d_sum,
            last_kernel_time: 0.0,
            _marker: std::marker::PhantomData,
        };
        s.warmup();
        s
    }

    fn elem_grid(&self) -> CubeCount {
        CubeCount::Static(self.array_size / TBSIZE, 1, 1)
    }

    fn cube_dim(&self) -> CubeDim {
        CubeDim { x: TBSIZE, y: 1, z: 1 }
    }

    fn warmup(&mut self) {
        let n = TBSIZE;
        let bytes = n as usize * F::elem_size();
        let w_a = self.client.empty(bytes);
        let w_b = self.client.empty(bytes);
        let w_c = self.client.empty(bytes);
        let w_sum = self.client.empty(self.dot_num_blocks as usize * F::elem_size());

        let one = CubeCount::Static(1, 1, 1);
        let dim = self.cube_dim();
        let n_us = n as usize;

        unsafe {
            init_kernel::launch_unchecked::<F, Rt>(
                &self.client,
                one.clone(),
                dim,
                ArrayArg::from_raw_parts(w_a.clone(), n_us),
                ArrayArg::from_raw_parts(w_b.clone(), n_us),
                ArrayArg::from_raw_parts(w_c.clone(), n_us),
                F::from_f64(START_A),
                F::from_f64(START_B),
                F::from_f64(START_C),
            );
            copy_kernel::launch_unchecked::<F, Rt>(
                &self.client,
                one.clone(),
                dim,
                ArrayArg::from_raw_parts(w_a.clone(), n_us),
                ArrayArg::from_raw_parts(w_c.clone(), n_us),
            );
            mul_kernel::launch_unchecked::<F, Rt>(
                &self.client,
                one.clone(),
                dim,
                ArrayArg::from_raw_parts(w_b.clone(), n_us),
                ArrayArg::from_raw_parts(w_c.clone(), n_us),
                F::from_f64(START_SCALAR),
            );
            add_kernel::launch_unchecked::<F, Rt>(
                &self.client,
                one.clone(),
                dim,
                ArrayArg::from_raw_parts(w_a.clone(), n_us),
                ArrayArg::from_raw_parts(w_b.clone(), n_us),
                ArrayArg::from_raw_parts(w_c.clone(), n_us),
            );
            triad_kernel::launch_unchecked::<F, Rt>(
                &self.client,
                one,
                dim,
                ArrayArg::from_raw_parts(w_a.clone(), n_us),
                ArrayArg::from_raw_parts(w_b.clone(), n_us),
                ArrayArg::from_raw_parts(w_c.clone(), n_us),
                F::from_f64(START_SCALAR),
                self.use_fma,
            );
            dot_kernel::launch_unchecked::<F, Rt>(
                &self.client,
                CubeCount::Static(self.dot_num_blocks, 1, 1),
                dim,
                ArrayArg::from_raw_parts(w_a.clone(), n_us),
                ArrayArg::from_raw_parts(w_b.clone(), n_us),
                ArrayArg::from_raw_parts(w_sum.clone(), self.dot_num_blocks as usize),
                n_us,
                self.use_fma,
            );
        }

        let _ = self
            .client
            .read_one(w_sum)
            .expect("warmup dot read_one failed");
    }

    fn init_arrays(&mut self) {
        let n = self.array_size as usize;
        let grid = self.elem_grid();
        let dim = self.cube_dim();
        let (a, b, c) = (self.d_a.clone(), self.d_b.clone(), self.d_c.clone());
        let client = self.client.clone();

        let (t, _) = profile_secs::<Rt, _, _>(&self.client, "init", move || unsafe {
            init_kernel::launch_unchecked::<F, Rt>(
                &client,
                grid,
                dim,
                ArrayArg::from_raw_parts(a.clone(), n),
                ArrayArg::from_raw_parts(b.clone(), n),
                ArrayArg::from_raw_parts(c.clone(), n),
                F::from_f64(START_A),
                F::from_f64(START_B),
                F::from_f64(START_C),
            );
        });
        self.last_kernel_time = t;
    }

    fn copy(&mut self) {
        let n = self.array_size as usize;
        let grid = self.elem_grid();
        let dim = self.cube_dim();
        let (a, c) = (self.d_a.clone(), self.d_c.clone());
        let client = self.client.clone();

        let (t, _) = profile_secs::<Rt, _, _>(&self.client, "copy", move || unsafe {
            copy_kernel::launch_unchecked::<F, Rt>(
                &client,
                grid,
                dim,
                ArrayArg::from_raw_parts(a.clone(), n),
                ArrayArg::from_raw_parts(c.clone(), n),
            );
        });
        self.last_kernel_time = t;
    }

    fn mul(&mut self) {
        let n = self.array_size as usize;
        let grid = self.elem_grid();
        let dim = self.cube_dim();
        let (b, c) = (self.d_b.clone(), self.d_c.clone());
        let client = self.client.clone();

        let (t, _) = profile_secs::<Rt, _, _>(&self.client, "mul", move || unsafe {
            mul_kernel::launch_unchecked::<F, Rt>(
                &client,
                grid,
                dim,
                ArrayArg::from_raw_parts(b.clone(), n),
                ArrayArg::from_raw_parts(c.clone(), n),
                F::from_f64(START_SCALAR),
            );
        });
        self.last_kernel_time = t;
    }

    fn add(&mut self) {
        let n = self.array_size as usize;
        let grid = self.elem_grid();
        let dim = self.cube_dim();
        let (a, b, c) = (self.d_a.clone(), self.d_b.clone(), self.d_c.clone());
        let client = self.client.clone();

        let (t, _) = profile_secs::<Rt, _, _>(&self.client, "add", move || unsafe {
            add_kernel::launch_unchecked::<F, Rt>(
                &client,
                grid,
                dim,
                ArrayArg::from_raw_parts(a.clone(), n),
                ArrayArg::from_raw_parts(b.clone(), n),
                ArrayArg::from_raw_parts(c.clone(), n),
            );
        });
        self.last_kernel_time = t;
    }

    fn triad(&mut self) {
        let n = self.array_size as usize;
        let grid = self.elem_grid();
        let dim = self.cube_dim();
        let (a, b, c) = (self.d_a.clone(), self.d_b.clone(), self.d_c.clone());
        let client = self.client.clone();
        let use_fma = self.use_fma;

        let (t, _) = profile_secs::<Rt, _, _>(&self.client, "triad", move || unsafe {
            triad_kernel::launch_unchecked::<F, Rt>(
                &client,
                grid,
                dim,
                ArrayArg::from_raw_parts(a.clone(), n),
                ArrayArg::from_raw_parts(b.clone(), n),
                ArrayArg::from_raw_parts(c.clone(), n),
                F::from_f64(START_SCALAR),
                use_fma,
            );
        });
        self.last_kernel_time = t;
    }

    fn dot(&mut self) -> F {
        let n = self.array_size as usize;
        let dim = self.cube_dim();
        let grid = CubeCount::Static(self.dot_num_blocks, 1, 1);
        let (a, b, sum) = (self.d_a.clone(), self.d_b.clone(), self.d_sum.clone());
        let client = self.client.clone();
        let use_fma = self.use_fma;
        let array_size = self.array_size as usize;
        let nb = self.dot_num_blocks as usize;

        let (t, bytes) = profile_secs::<Rt, _, _>(&self.client, "dot", move || {
            unsafe {
                dot_kernel::launch_unchecked::<F, Rt>(
                    &client,
                    grid,
                    dim,
                    ArrayArg::from_raw_parts(a.clone(), n),
                    ArrayArg::from_raw_parts(b.clone(), n),
                    ArrayArg::from_raw_parts(sum.clone(), nb),
                    array_size,
                    use_fma,
                );
            }
            client.read_one(sum).expect("dot read_one failed")
        });
        self.last_kernel_time = t;

        let partials = F::from_bytes(&bytes);
        let mut total = F::from_f64(0.0);
        for i in 0..nb {
            total = total.h_add(partials[i]);
        }
        total
    }

    fn read_arrays(&self) -> (Vec<F>, Vec<F>, Vec<F>) {
        let ba = self.client.read_one(self.d_a.clone()).expect("read_one(d_a) failed");
        let bb = self.client.read_one(self.d_b.clone()).expect("read_one(d_b) failed");
        let bc = self.client.read_one(self.d_c.clone()).expect("read_one(d_c) failed");
        (
            F::from_bytes(&ba).to_vec(),
            F::from_bytes(&bb).to_vec(),
            F::from_bytes(&bc).to_vec(),
        )
    }

    fn get_time_taken(&self) -> f64 {
        self.last_kernel_time
    }
}

fn device_for_index(index: usize) -> <Rt as Runtime>::Device {
    #[cfg(feature = "cuda")]
    {
        cubecl::cuda::CudaDevice::new(index)
    }
    #[cfg(all(feature = "hip", not(feature = "cuda")))]
    {
        cubecl::hip::AmdDevice::new(index)
    }
}

fn device_name(index: usize) -> String {
    std::env::var("BS_GPU_NAME").unwrap_or_else(|_| format!("{} device {}", BACKEND_LABEL, index))
}

fn driver_string() -> String {
    std::env::var("BS_DRIVER").unwrap_or_else(|_| "unknown".to_string())
}

fn sm_count<R: Runtime>(_client: &ComputeClient<R>) -> u32 {
    if let Ok(v) = std::env::var("BS_SM_COUNT") {
        if let Ok(n) = v.parse::<u32>() {
            return n;
        }
    }
    eprintln!(
        "warning: SM/CU count not available from CubeCL client properties; \
         set BS_SM_COUNT to match the device (dot grid = 4 * SM count)"
    );
    64
}

struct Args {
    array_size: u32,
    num_times: u32,
    device_index: usize,
    mibibytes: bool,
    csv_output: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            array_size: 33554432,
            num_times: 1000,
            device_index: 0,
            mibibytes: false,
            csv_output: false,
        }
    }
}

fn print_help(prog: &str) {
    println!();
    println!("Usage: {} [OPTIONS]", prog);
    println!();
    println!("Options:");
    println!("  -h  --help               Print the message");
    println!("      --list               List available devices");
    println!("      --device     INDEX   Select device at INDEX");
    println!("  -s  --arraysize  SIZE    Use SIZE elements in the array");
    println!("  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)");
    println!("      --float              Use floats (no-op, float is already default)");
    println!("      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)");
    println!();
}

fn parse_arguments() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut args = Args::default();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--list" => {
                println!();
                println!("Devices:");
                println!("0: {}", device_name(0));
                println!();
                std::process::exit(0);
            }
            "--device" => {
                i += 1;
                args.device_index = argv
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid device index.");
                        std::process::exit(1);
                    });
            }
            "--arraysize" | "-s" => {
                i += 1;
                let v: u32 = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("Invalid array size.");
                    std::process::exit(1);
                });
                if v == 0 {
                    eprintln!("Invalid array size.");
                    std::process::exit(1);
                }
                args.array_size = v;
            }
            "--numtimes" | "-n" => {
                i += 1;
                let v: u32 = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("Invalid number of times.");
                    std::process::exit(1);
                });
                if v < 2 {
                    eprintln!("Number of times must be 2 or more");
                    std::process::exit(1);
                }
                args.num_times = v;
            }
            "--float" => {}
            "--mibibytes" => args.mibibytes = true,
            "--csv" => args.csv_output = true,
            "--help" | "-h" => {
                print_help(&argv[0]);
                std::process::exit(0);
            }
            other => {
                eprintln!("Unrecognized argument '{}' (try '--help')", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    args
>>>>>>> main
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

<<<<<<< HEAD
/// Returns (checksum relative error, max element relative error).
=======
>>>>>>> main
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
    dot_sum: F,
    array_size: u32,
<<<<<<< HEAD
) -> Correctness {
=======
    csv_output: bool,
) {
>>>>>>> main
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
    let actual_dot = dot_sum.to_host_f64();
<<<<<<< HEAD
    let product_magnitude = (gold_a.to_host_f64() * gold_b.to_host_f64()).abs();
=======

    let product_magnitude = (gold_a.to_host_f64() * gold_b.to_host_f64()).abs();
    let dot_underflows = product_magnitude < F::tiny();

>>>>>>> main
    let dot_err = if expected_dot != 0.0 {
        (actual_dot - expected_dot).abs() / expected_dot.abs()
    } else {
        actual_dot.abs()
    };

<<<<<<< HEAD
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
=======
    let precname = if F::IS_FLOAT32 { "float32" } else { "float64" };

    let emit = |line: String| {
        if csv_output {
            eprintln!("{}", line);
        } else {
            println!("{}", line);
        }
    };

    emit(String::new());
    emit(format!("Correctness metrics ({}, {} iterations)", precname, ntimes));
    emit(format!(
        "{:<10}{:<18}{:<18}{:<18}",
        "Array", "Expected", "Checksum rel err", "Max elem rel err"
    ));

    let row = |name: &str, gold: f64, chk: f64, mx: f64| {
        format!("{:<10}{:<18.6e}{:<18.3e}{:<18.3e}", name, gold, chk, mx)
    };
    emit(row("a", gold_a.to_host_f64(), chk_a, max_a));
    emit(row("b", gold_b.to_host_f64(), chk_b, max_b));
    emit(row("c", gold_c.to_host_f64(), chk_c, max_c));

    emit(format!(
        "{:<10}{:<18.6e}{:<18.3e}{:<18}",
        "dot", expected_dot, dot_err, "-"
    ));
    emit(format!("  (actual dot   = {:.9e})", actual_dot));
    if dot_underflows {
        emit(format!(
            "  (note: expected element product {:.3e} underflows this precision at this \
             iteration count, so the dot relative error above is not meaningful)",
            product_magnitude
        ));
>>>>>>> main
    }
}

fn run<F: Elem>(args: &Args, use_fma: bool) {
<<<<<<< HEAD
    let n = args.array_size as usize;
    let bpe = F::elem_size();

    let client = backend::device(args.device_index)
        .unwrap_or_else(|e| fail(&format!("device {}: {}", args.device_index, e)))
        .client();
    let props = client.properties();
    let hw = &props.hardware;
    if !props.supports_type(F::elem_type_native()) {
        fail(&format!("{} is not supported on this device", F::PRECISION));
    }
    validate_block_size(args, hw.plane_size_min, hw.max_units_per_cube);

    let (device, name_note) = backend::resolve_name(args.device_index, &client);
    let (groups, groups_detail, groups_note) = dot_groups(hw.num_streaming_multiprocessors);

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
        notes: name_note.into_iter().chain(groups_note).collect(),
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
    let mut dot_sum = F::from_f64(0.0);

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
=======
    let bpe = F::elem_size();
    let n = args.array_size as usize;

    if !args.csv_output {
        println!("Running kernels {} times", args.num_times);
        println!("Precision: {}", if F::IS_FLOAT32 { "float" } else { "double" });
        if args.mibibytes {
            println!(
                "Array size: {:.1} MiB (={:.1} GiB)",
                (n * bpe) as f64 * 2f64.powi(-20),
                (n * bpe) as f64 * 2f64.powi(-30)
            );
            println!(
                "Total size: {:.1} MiB (={:.1} GiB)",
                3.0 * (n * bpe) as f64 * 2f64.powi(-20),
                3.0 * (n * bpe) as f64 * 2f64.powi(-30)
            );
        } else {
            println!(
                "Array size: {:.1} MB (={:.1} GB)",
                (n * bpe) as f64 * 1.0e-6,
                (n * bpe) as f64 * 1.0e-9
            );
            println!(
                "Total size: {:.1} MB (={:.1} GB)",
                3.0 * (n * bpe) as f64 * 1.0e-6,
                3.0 * (n * bpe) as f64 * 1.0e-9
            );
        }
    }

    let mut stream = CubeStream::<F>::new(args.array_size, args.device_index, use_fma);

    stream.init_arrays();
    let init_elapsed_s = stream.get_time_taken();

    let mut timings: Vec<Vec<f64>> = vec![Vec::new(); 5];
    let mut dot_sum = F::from_f64(0.0);

    for _ in 0..args.num_times {
        stream.copy();
        timings[0].push(stream.get_time_taken());
        stream.mul();
        timings[1].push(stream.get_time_taken());
        stream.add();
        timings[2].push(stream.get_time_taken());
        stream.triad();
        timings[3].push(stream.get_time_taken());
        dot_sum = stream.dot();
        timings[4].push(stream.get_time_taken());
    }

    let scale = if args.mibibytes { 2f64.powi(-20) } else { 1.0e-9 };
    let init_bw = (scale * (3 * bpe * n) as f64) / init_elapsed_s;

    if !args.csv_output {
        println!(
            "Init: {:7} s (={} {})",
            init_elapsed_s,
            init_bw,
            if args.mibibytes { "MiBytes/sec" } else { "GBytes/sec" }
        );
        println!(
            "{:<12}{:<12}{:<12}{:<12}{:<12}",
            "Function",
            if args.mibibytes { "MiBytes/sec" } else { "GBytes/sec" },
            "Min (sec)",
            "Max",
            "Average"
        );
    } else {
        println!("backend,GPU,precision,vec_size,routine,BW_GBs");
    }

    let labels = ["Copy", "Mul", "Add", "Triad", "Dot"];
    let sizes = [
        2 * bpe * n,
        2 * bpe * n,
        3 * bpe * n,
        3 * bpe * n,
        2 * bpe * n,
    ];
    let precname = if F::IS_FLOAT32 { "float32" } else { "float64" };
    let gpu = device_name(args.device_index);

    for i in 0..5 {
        let t = &timings[i][1..];
        let tmin = t.iter().cloned().fold(f64::INFINITY, f64::min);
        let tmax = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let average: f64 = t.iter().sum::<f64>() / t.len() as f64;

        if !args.csv_output {
            println!(
                "{:<12}{:<12.3}{:<12.5}{:<12.5}{:<12.5}",
                labels[i],
                scale * sizes[i] as f64 / tmin,
                tmin,
                tmax,
                average
            );
        } else {
            for tt in &timings[i] {
                println!(
                    "{},{},{},{},{},{}",
                    IMPLEMENTATION_STRING,
                    gpu,
                    precname,
                    args.array_size,
                    labels[i],
                    1.0e-9 * sizes[i] as f64 / tt
                );
>>>>>>> main
            }
        }
    }

<<<<<<< HEAD
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
=======
    let (a, b, c) = stream.read_arrays();
    check_solution(args.num_times, &a, &b, &c, dot_sum, args.array_size, args.csv_output);
}

fn main() {
    let args = parse_arguments();
    let use_fma = std::env::var("FMAD").unwrap_or_else(|_| "1".to_string()) == "1";

    if USE_FLOAT {
        run::<f32>(&args, use_fma);
    } else {
        run::<f64>(&args, use_fma);
>>>>>>> main
    }
}
