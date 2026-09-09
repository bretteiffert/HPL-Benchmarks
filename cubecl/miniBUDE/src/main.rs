//! CubeCL miniBUDE benchmark that reads reference decks, profiles kernels, and reports differential and deck metrics.
//! Exactly one of the cuda or hip Cargo features must be enabled; the run scripts use no default features and select one explicitly.
//! PRECISION_BITS defaults to 64 in source, and an fp64 differential pass is generated only for fp32 benchmark runs.

mod kernel;
mod metrics;

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use std::fs;
use std::io::Write;
use std::time::{Duration, Instant};

use kernel::fasten_main;
use metrics::{
    compare_to_deck, compute_metrics, gfmt, sci, ulp_bucket_label, Metrics, ULP_NBUCKETS,
};

#[cfg(all(feature = "cuda", feature = "hip"))]
compile_error!("enable exactly one of the `cuda` and `hip` features, not both");

#[cfg(not(any(feature = "cuda", feature = "hip")))]
compile_error!("enable exactly one of the `cuda` and `hip` features");

#[cfg(feature = "cuda")]
mod backend {
    use cubecl::cuda::{CudaDevice, CudaRuntime};
    pub type Rt = CudaRuntime;
    pub type Dev = CudaDevice;
    pub fn device(index: usize) -> Dev {
        CudaDevice::new(index)
    }
    pub fn label() -> &'static str {
        "CUDA"
    }
}

#[cfg(feature = "hip")]
mod backend {
    use cubecl::hip::{AmdDevice, HipRuntime};
    pub type Rt = HipRuntime;
    pub type Dev = AmdDevice;
    pub fn device(index: usize) -> Dev {
        AmdDevice::new(index)
    }
    pub fn label() -> &'static str {
        "HIP"
    }
}

const DEFAULT_ITERS: usize = 8;
const DEFAULT_WARMUP: usize = 2;
const DEFAULT_ENERGY_ENTRIES: usize = 8;
const DEFAULT_DATA_DIR: &str = "../../data/miniBUDE/bm1";

const PPWIS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

const PRECISION_BITS: usize = 64;

pub trait HostFloat: Float + CubeElement + Copy + 'static {
    fn from_f32(x: f32) -> Self;
    fn as_f64(self) -> f64;
    fn precision_bits() -> usize;
}

impl HostFloat for f32 {
    fn from_f32(x: f32) -> Self {
        x
    }
    fn as_f64(self) -> f64 {
        self as f64
    }
    fn precision_bits() -> usize {
        32
    }
}

impl HostFloat for f64 {
    fn from_f32(x: f32) -> Self {
        x as f64
    }
    fn as_f64(self) -> f64 {
        self
    }
    fn precision_bits() -> usize {
        64
    }
}

#[derive(Clone, Copy)]
struct DiskAtom {
    x: f32,
    y: f32,
    z: f32,
    ty: i32,
}

#[derive(Clone, Copy)]
struct DiskFFParams {
    hbtype: i32,
    radius: f32,
    hphb: f32,
    elsc: f32,
}

fn le_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn le_i32(b: &[u8]) -> i32 {
    i32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn read_bytes(path: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| panic!("Bad file: {} ({})", path, e))
}

fn read_atoms(path: &str) -> Vec<DiskAtom> {
    read_bytes(path)
        .chunks_exact(16)
        .map(|c| DiskAtom {
            x: le_f32(&c[0..4]),
            y: le_f32(&c[4..8]),
            z: le_f32(&c[8..12]),
            ty: le_i32(&c[12..16]),
        })
        .collect()
}

fn read_ffparams(path: &str) -> Vec<DiskFFParams> {
    read_bytes(path)
        .chunks_exact(16)
        .map(|c| DiskFFParams {
            hbtype: le_i32(&c[0..4]),
            radius: le_f32(&c[4..8]),
            hphb: le_f32(&c[8..12]),
            elsc: le_f32(&c[12..16]),
        })
        .collect()
}

fn read_f32s(path: &str) -> Vec<f32> {
    read_bytes(path)
        .chunks_exact(4)
        .map(|c| le_f32(c))
        .collect()
}

struct Params {
    protein: Vec<DiskAtom>,
    ligand: Vec<DiskAtom>,
    forcefield: Vec<DiskFFParams>,
    poses: Vec<Vec<f32>>,
    ref_energies: Vec<f32>,
    iterations: usize,
    warmup_iterations: usize,
    out_rows: usize,
    deck_dir: String,
    output: String,
    device_selector: String,
    csv: bool,
    list: bool,
}

impl Params {
    fn new() -> Self {
        Params {
            protein: vec![],
            ligand: vec![],
            forcefield: vec![],
            poses: vec![vec![]; 6],
            ref_energies: vec![],
            iterations: DEFAULT_ITERS,
            warmup_iterations: DEFAULT_WARMUP,
            out_rows: DEFAULT_ENERGY_ENTRIES,
            deck_dir: DEFAULT_DATA_DIR.to_string(),
            output: String::new(),
            device_selector: String::new(),
            csv: false,
            list: false,
        }
    }
    fn total_iterations(&self) -> usize {
        self.iterations + self.warmup_iterations
    }
    fn natpro(&self) -> usize {
        self.protein.len()
    }
    fn natlig(&self) -> usize {
        self.ligand.len()
    }
    fn nposes(&self) -> usize {
        self.poses[0].len()
    }
}

struct Sample {
    ppwi: usize,
    wgsize: usize,
    precision_bits: usize,
    energies: Vec<f64>,
    kernel_millis: Vec<f64>,
    wall_millis: Vec<f64>,
    context_millis: f64,
}

type Client = ComputeClient<backend::Rt>;

fn profile_ms(client: &Client, name: &'static str, f: impl FnOnce() + Send) -> f64 {
    let (_, profile) = client
        .profile(f, name)
        .expect("device profiling failed");
    let ticks = cubecl::future::block_on(profile.resolve());
    let duration: Duration = ticks.duration();
    duration.as_secs_f64() * 1e3
}

fn upload<F: HostFloat>(client: &Client, xs: &[F]) -> cubecl::server::Handle {
    client.create(Bytes::from_bytes_vec(F::as_bytes(xs).to_vec()))
}

fn upload_u32(client: &Client, xs: &[u32]) -> cubecl::server::Handle {
    client.create(Bytes::from_bytes_vec(u32::as_bytes(xs).to_vec()))
}

fn fasten<F: HostFloat>(client: &Client, p: &Params, wgsize: usize, ppwi: usize) -> Sample {
    let n = p.nposes();

    let prot_x: Vec<F> = p.protein.iter().map(|a| F::from_f32(a.x)).collect();
    let prot_y: Vec<F> = p.protein.iter().map(|a| F::from_f32(a.y)).collect();
    let prot_z: Vec<F> = p.protein.iter().map(|a| F::from_f32(a.z)).collect();
    let prot_t: Vec<u32> = p.protein.iter().map(|a| a.ty as u32).collect();
    let lig_x: Vec<F> = p.ligand.iter().map(|a| F::from_f32(a.x)).collect();
    let lig_y: Vec<F> = p.ligand.iter().map(|a| F::from_f32(a.y)).collect();
    let lig_z: Vec<F> = p.ligand.iter().map(|a| F::from_f32(a.z)).collect();
    let lig_t: Vec<u32> = p.ligand.iter().map(|a| a.ty as u32).collect();
    let ff_hb: Vec<u32> = p.forcefield.iter().map(|f| f.hbtype as u32).collect();
    let ff_rad: Vec<F> = p.forcefield.iter().map(|f| F::from_f32(f.radius)).collect();
    let ff_hphb: Vec<F> = p.forcefield.iter().map(|f| F::from_f32(f.hphb)).collect();
    let ff_elsc: Vec<F> = p.forcefield.iter().map(|f| F::from_f32(f.elsc)).collect();

    let context_start = Instant::now();
    let h_prot_x = upload(client, &prot_x);
    let h_prot_y = upload(client, &prot_y);
    let h_prot_z = upload(client, &prot_z);
    let h_prot_t = upload_u32(client, &prot_t);
    let h_lig_x = upload(client, &lig_x);
    let h_lig_y = upload(client, &lig_y);
    let h_lig_z = upload(client, &lig_z);
    let h_lig_t = upload_u32(client, &lig_t);
    let h_ff_hb = upload_u32(client, &ff_hb);
    let h_ff_rad = upload(client, &ff_rad);
    let h_ff_hphb = upload(client, &ff_hphb);
    let h_ff_elsc = upload(client, &ff_elsc);
    let h_t: Vec<_> = (0..6)
        .map(|i| {
            let v: Vec<F> = p.poses[i].iter().map(|&x| F::from_f32(x)).collect();
            upload(client, &v)
        })
        .collect();
    let h_out = client.empty(n * std::mem::size_of::<F>());
    cubecl::future::block_on(client.sync());
    let context_millis = context_start.elapsed().as_secs_f64() * 1e3;

    let blocks = ((n as f64 / ppwi as f64).ceil() / wgsize as f64).ceil() as u32;
    let cube_count = CubeCount::Static(blocks, 1, 1);
    let cube_dim = CubeDim::new_1d(wgsize as u32);

    let mut sample = Sample {
        ppwi,
        wgsize,
        precision_bits: F::precision_bits(),
        energies: vec![0.0; n],
        kernel_millis: vec![],
        wall_millis: vec![],
        context_millis,
    };

    for _ in 0..p.total_iterations() {
        let wall_start = Instant::now();
        let ms = profile_ms(client, "fasten_main", || unsafe {
            fasten_main::launch_unchecked::<F, backend::Rt>(
                client,
                cube_count.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(h_out.clone(), n),
                ArrayArg::from_raw_parts(h_prot_x.clone(), prot_x.len()),
                ArrayArg::from_raw_parts(h_prot_y.clone(), prot_y.len()),
                ArrayArg::from_raw_parts(h_prot_z.clone(), prot_z.len()),
                ArrayArg::from_raw_parts(h_prot_t.clone(), prot_t.len()),
                ArrayArg::from_raw_parts(h_lig_x.clone(), lig_x.len()),
                ArrayArg::from_raw_parts(h_lig_y.clone(), lig_y.len()),
                ArrayArg::from_raw_parts(h_lig_z.clone(), lig_z.len()),
                ArrayArg::from_raw_parts(h_lig_t.clone(), lig_t.len()),
                ArrayArg::from_raw_parts(h_t[0].clone(), n),
                ArrayArg::from_raw_parts(h_t[1].clone(), n),
                ArrayArg::from_raw_parts(h_t[2].clone(), n),
                ArrayArg::from_raw_parts(h_t[3].clone(), n),
                ArrayArg::from_raw_parts(h_t[4].clone(), n),
                ArrayArg::from_raw_parts(h_t[5].clone(), n),
                ArrayArg::from_raw_parts(h_ff_hb.clone(), ff_hb.len()),
                ArrayArg::from_raw_parts(h_ff_rad.clone(), ff_rad.len()),
                ArrayArg::from_raw_parts(h_ff_hphb.clone(), ff_hphb.len()),
                ArrayArg::from_raw_parts(h_ff_elsc.clone(), ff_elsc.len()),
                p.natlig() as u32,
                p.natpro() as u32,
                n as u32,
                ppwi,
            )
        });
        let wall_ms = wall_start.elapsed().as_secs_f64() * 1e3;

        sample.kernel_millis.push(ms);
        sample.wall_millis.push(wall_ms);
    }

    let bytes = client
        .read_one(h_out.clone())
        .expect("reading device results failed");
    let raw = F::from_bytes(&bytes);
    for (i, v) in raw.iter().take(n).enumerate() {
        sample.energies[i] = v.as_f64();
    }

    sample
}

struct SummaryStats {
    min: f64,
    max: f64,
    sum: f64,
    mean: f64,
    std_dev: f64,
}

impl SummaryStats {
    fn new(ys: &[f64]) -> Self {
        let sum: f64 = ys.iter().sum();
        let mean = sum / ys.len() as f64;
        let variance = ys.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / ys.len() as f64;
        SummaryStats {
            min: ys.iter().cloned().fold(f64::INFINITY, f64::min),
            max: ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            sum,
            mean,
            std_dev: variance.sqrt(),
        }
    }
}

struct Result {
    sample: Sample,
    ms: SummaryStats,
    gflops: f64,
    ginsts: f64,
    interactions_per_sec: f64,
    launch_overhead_ms: f64,
    vs_reference: Metrics,
    has_differential: bool,
    vs_deck: Metrics,
}

fn evaluate(p: &Params, s: Sample, reference: Option<&[f64]>) -> Result {
    assert!(
        !s.kernel_millis.is_empty(),
        "Sample size is 0, this implementation did not measure kernel runtime!"
    );

    let ms_without_warmup: Vec<f64> = s.kernel_millis[p.warmup_iterations..].to_vec();
    let elapsed: f64 = ms_without_warmup.iter().sum();
    let ms = elapsed / p.iterations as f64;
    let runtime = ms * 1e-3;

    let ops_per_wg = s.ppwi * 27
        + p.natlig() * (2 + s.ppwi * 18 + p.natpro() * (10 + s.ppwi * 30))
        + s.ppwi;
    let total_ops = ops_per_wg as f64 * (p.nposes() as f64 / s.ppwi as f64);
    let gflops = (total_ops / runtime) / 1e9;

    let total_finsts = 25 * p.natpro() * p.natlig() * p.nposes();
    let ginsts = (total_finsts as f64 / runtime) / 1e9;

    let interactions = p.nposes() * p.natlig() * p.natpro();
    let interactions_per_sec = interactions as f64 / runtime;

    let has_differential = reference.is_some();
    let vs_reference = match reference {
        Some(r) => compute_metrics(&s.energies, r, s.precision_bits),
        None => Metrics::default(),
    };
    let vs_deck = compare_to_deck(&s.energies, &p.ref_energies);

    let mut launch_overhead_ms = 0.0;
    if s.wall_millis.len() == s.kernel_millis.len()
        && s.kernel_millis.len() > p.warmup_iterations
    {
        let acc: f64 = (p.warmup_iterations..s.kernel_millis.len())
            .map(|i| s.wall_millis[i] - s.kernel_millis[i])
            .sum();
        launch_overhead_ms = acc / (s.kernel_millis.len() - p.warmup_iterations) as f64;
    }

    let stats = SummaryStats::new(&ms_without_warmup);
    Result {
        sample: s,
        ms: stats,
        gflops,
        ginsts,
        interactions_per_sec,
        launch_overhead_ms,
        vs_reference,
        has_differential,
        vs_deck,
    }
}

fn dump_results(p: &Params, r: &Result) {
    if p.output.is_empty() {
        return;
    }
    let mut f = fs::File::create(&p.output).expect("cannot open output file");
    for e in &r.sample.energies {
        writeln!(f, "{:.17e}", e).unwrap();
    }
}

fn show_human_readable(p: &Params, r: &Result) {
    let prefix = " ";
    let raw: Vec<String> = r.sample.kernel_millis.iter().map(|&x| gfmt(x)).collect();

    println!("{} - precision_bits:      {}", prefix, r.sample.precision_bits);
    println!(
        "{}   param:               {{ ppwi: {}, wgsize: {} }}",
        prefix, r.sample.ppwi, r.sample.wgsize
    );
    println!("{}   raw_iterations:      [{}]", prefix, raw.join(","));
    println!("{}   context_ms:          {:.6}", prefix, r.sample.context_millis);
    println!("{}   sum_ms:              {:.3}", prefix, r.ms.sum);
    println!("{}   avg_ms:              {:.3}", prefix, r.ms.mean);
    println!("{}   min_ms:              {:.3}", prefix, r.ms.min);
    println!("{}   max_ms:              {:.3}", prefix, r.ms.max);
    println!("{}   stddev_ms:           {:.3}", prefix, r.ms.std_dev);
    println!(
        "{}   giga_interactions/s: {:.3}",
        prefix,
        r.interactions_per_sec / 1e9
    );
    println!("{}   gflop/s:             {:.3}", prefix, r.gflops);
    println!("{}   gfinst/s:            {:.3}", prefix, r.ginsts);
    println!("{}   launch_overhead_ms:  {:.3}", prefix, r.launch_overhead_ms);
    println!("{}   energies:            ", prefix);
    for i in 0..p.out_rows.min(r.sample.energies.len()) {
        println!("{}     - {:.6}", prefix, r.sample.energies[i]);
    }

    if r.has_differential {
        let m = &r.vs_reference;
        println!("{}   vs_fp64_reference:", prefix);
        println!("{}     l2_rel_norm:       {}", prefix, sci(m.l2_rel_norm, 6));
        println!("{}     rms_abs:           {}", prefix, sci(m.rms_abs, 6));
        println!("{}     mean_signed:       {}", prefix, sci(m.mean_signed, 6));
        println!(
            "{}     max_abs:           {{ value: {}, index: {} }}",
            prefix,
            sci(m.max_abs, 6),
            m.max_abs_idx
        );
        println!(
            "{}     max_rel:           {{ value: {}, index: {} }}",
            prefix,
            sci(m.max_rel, 6),
            m.max_rel_idx
        );
        println!(
            "{}     max_ulp:           {{ value: {}, index: {} }}",
            prefix, m.max_ulp, m.max_ulp_idx
        );
        println!("{}     compared:          {}", prefix, m.compared);
        println!("{}     non_finite:        {}", prefix, m.non_finite);
        println!("{}     ulp_histogram:", prefix);
        for i in 0..ULP_NBUCKETS {
            if m.ulp_histogram[i] > 0 {
                println!(
                    "{}       \"{}\": {}",
                    prefix,
                    ulp_bucket_label(i),
                    m.ulp_histogram[i]
                );
            }
        }
    } else {
        println!(
            "{}   vs_fp64_reference:   ~ # benchmarked precision is the reference",
            prefix
        );
    }

    let m = &r.vs_deck;
    println!(
        "{}   vs_deck_reference:   # informational; ref_energies.out is low precision",
        prefix
    );
    println!("{}     l2_rel_norm:       {}", prefix, sci(m.l2_rel_norm, 6));
    println!(
        "{}     max_rel:           {{ value: {}, index: {} }}",
        prefix,
        sci(m.max_rel, 6),
        m.max_rel_idx
    );
    println!(
        "{}     beyond_tolerance:  {}/{} # tolerance = {}%",
        prefix, m.beyond_tolerance, m.compared, metrics::REPORT_TOLERANCE_PCT
    );
}

fn show_csv(r: &Result, header: bool) {
    if header {
        println!(
            "precision_bits,ppwi,wgsize,sum_ms,avg_ms,min_ms,max_ms,stddev_ms,\
interactions/s,gflops/s,gfinst/s,launch_overhead_ms,\
l2_rel_norm,rms_abs,mean_signed,max_abs,max_rel,max_ulp,deck_l2_rel_norm"
        );
    }
    print!(
        "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        r.sample.precision_bits,
        r.sample.ppwi,
        r.sample.wgsize,
        r.ms.sum,
        r.ms.mean,
        r.ms.min,
        r.ms.max,
        r.ms.std_dev,
        r.interactions_per_sec,
        r.gflops,
        r.ginsts,
        r.launch_overhead_ms
    );
    if r.has_differential {
        let m = &r.vs_reference;
        print!(
            ",{},{},{},{},{},{}",
            sci(m.l2_rel_norm, 6),
            sci(m.rms_abs, 6),
            sci(m.mean_signed, 6),
            sci(m.max_abs, 6),
            sci(m.max_rel, 6),
            m.max_ulp
        );
    } else {
        print!(",,,,,");
    }
    println!(",{}", sci(r.vs_deck.l2_rel_norm, 6));
}

fn print_help() {
    println!();
    println!("Usage: bude_cubecl [COMMAND|OPTIONS]\n");
    println!("Commands:");
    println!("  help -h --help       Print this message");
    println!("  list -l --list       List available devices");
    println!("Options:");
    println!("  -d --device  INDEX   Select device at INDEX from output of --list");
    println!("                       [optional] default=0");
    println!("  -i --iter    I       Repeat kernel I times");
    println!("                       [optional] default={}", DEFAULT_ITERS);
    println!("  -n --poses   N       Compute energies for only N poses, use 0 for deck max");
    println!("                       [optional] default=0 ");
    println!("  -p --ppwi    PPWI    A CSV list of poses per work-item for the kernel, use `all` for everything");
    println!(
        "                       [optional] default={}; available={:?}",
        PPWIS[0], PPWIS
    );
    println!("  -w --wgsize  WGSIZE  A CSV list of work-group sizes");
    println!("                       [optional] default=1");
    println!("     --deck    DIR     Use the DIR directory as input deck");
    println!("                       [optional] default=`{}`", DEFAULT_DATA_DIR);
    println!("  -o --out     PATH    Save resulting energies to PATH");
    println!("                       [optional]");
    println!("  -r --rows    N       Output first N row(s) of energy values");
    println!(
        "                       [optional] default={}",
        DEFAULT_ENERGY_ENTRIES
    );
    println!("     --csv             Output results in CSV format");
    println!("                       [optional] default=false");
    println!();
    println!("Precision is not a flag: edit the PRECISION_BITS constant in src/main.rs,");
    println!("as with the reference's -DPRECISION_BITS.");
    println!();
}

fn parse_ints(s: &str, name: &str) -> Vec<usize> {
    s.split(',')
        .map(|t| {
            t.trim().parse::<usize>().unwrap_or_else(|_| {
                eprintln!("malformed value, positive integer required for <{}>: `{}`", name, t);
                std::process::exit(1);
            })
        })
        .collect()
}

fn parse_params(args: &[String]) -> (Params, Vec<usize>, Vec<usize>, usize) {
    let mut p = Params::new();
    let mut wgsizes: Vec<usize> = vec![];
    let mut ppwis: Vec<usize> = vec![];
    let mut nposes_arg: usize = 0;
    let precision_bits: usize = PRECISION_BITS;

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let value = |i: &mut usize| -> String {
            if *i + 1 >= args.len() {
                eprintln!("[{}] specified but no value was given", a);
                std::process::exit(1);
            }
            *i += 1;
            args[*i].clone()
        };
        match a {
            "--iter" | "-i" => p.iterations = parse_ints(&value(&mut i), "iter")[0],
            "--poses" | "-n" => nposes_arg = parse_ints(&value(&mut i), "poses")[0],
            "--device" | "-d" => p.device_selector = value(&mut i),
            "--deck" => p.deck_dir = value(&mut i),
            "--out" | "-o" => p.output = value(&mut i),
            "--rows" | "-r" => p.out_rows = parse_ints(&value(&mut i), "rows")[0],
            "--wgsize" | "-w" => wgsizes = parse_ints(&value(&mut i), "wgsize"),
            "--ppwi" | "-p" => {
                let s = value(&mut i);
                if s == "all" {
                    ppwis = PPWIS.to_vec();
                } else {
                    ppwis = parse_ints(&s, "ppwi");
                    for v in &ppwis {
                        if !PPWIS.contains(v) {
                            eprintln!(
                                "PPWI {} is not a supported value, should be one of `{:?}`",
                                v, PPWIS
                            );
                            std::process::exit(1);
                        }
                    }
                }
            }
            "--csv" => p.csv = true,
            "list" | "--list" | "-l" => p.list = true,
            "help" | "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                println!("Unrecognized argument '{}' (try '--help')", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if ppwis.is_empty() {
        ppwis = vec![PPWIS[0]];
    }
    if wgsizes.is_empty() {
        wgsizes = vec![1];
    }
    assert!(
        precision_bits == 32 || precision_bits == 64,
        "precision must be 32 or 64, got {}",
        precision_bits
    );

    if p.list {
        return (p, wgsizes, ppwis, precision_bits);
    }

    p.ligand = read_atoms(&format!("{}/ligand.in", p.deck_dir));
    p.protein = read_atoms(&format!("{}/protein.in", p.deck_dir));
    p.forcefield = read_ffparams(&format!("{}/forcefield.in", p.deck_dir));

    let poses = read_f32s(&format!("{}/poses.in", p.deck_dir));
    assert!(
        poses.len() % 6 == 0,
        "Pose size ({}) not divisible my 6!",
        poses.len()
    );
    let max_poses = poses.len() / 6;
    let n = if nposes_arg == 0 { max_poses } else { nposes_arg };
    assert!(
        n <= max_poses,
        "Requested poses ({}) exceeded max poses ({}) for deck",
        n,
        max_poses
    );
    for k in 0..6 {
        p.poses[k] = poses[k * max_poses..k * max_poses + n].to_vec();
    }

    let refs = fs::read_to_string(&format!("{}/ref_energies.out", p.deck_dir))
        .expect("cannot read ref_energies.out");
    p.ref_energies = refs
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<f32>().expect("bad ref energy"))
        .collect();
    assert!(
        p.nposes() <= p.ref_energies.len(),
        "Size of reference energies ({}) is less than poses ({})",
        p.ref_energies.len(),
        p.nposes()
    );

    (p, wgsizes, ppwis, precision_bits)
}

fn enumerate_devices() -> Vec<(usize, String)> {
    vec![(0, format!("{} device 0", backend::label()))]
}

fn run(p: &Params, wgsizes: &[usize], ppwis: &[usize], precision_bits: usize) -> bool {
    let devices = enumerate_devices();
    if devices.is_empty() {
        eprintln!(" # (no devices available)");
    }

    if p.list {
        println!("{}", if p.csv { "index,name" } else { "devices:" });
        for (i, name) in &devices {
            if p.csv {
                println!("{},{}", i, name);
            } else {
                println!("  {}: \"{}\"", i, name);
            }
        }
        return true;
    }

    let index: usize = p.device_selector.parse().unwrap_or(0);
    let device = backend::device(index);
    let client = <backend::Rt as Runtime>::client(&device);

    if !p.csv {
        println!(
            "device: {{ index: {},  name: \"{} device {}\" }}",
            index,
            backend::label(),
            index
        );
    }

    let want_differential = precision_bits != 64;
    let mut golden: Option<Vec<f64>> = None;

    let mut dumped = false;
    let mut results: Vec<Result> = vec![];

    for &ppwi in ppwis {
        for &wgsize in wgsizes {
            let n = p.nposes();
            if n < ppwi * wgsize {
                println!(
                    " # WARNING: pose count {} <= ({} (wgsize) * {} (ppwi)), skipping",
                    n, wgsize, ppwi
                );
                continue;
            }
            if n % (ppwi * wgsize) != 0 {
                println!(
                    " # WARNING: pose count {} % ({} (wgsize) * {} (ppwi)) != 0, skipping",
                    n, wgsize, ppwi
                );
                continue;
            }

            if want_differential && golden.is_none() {
                println!(
                    "# computing fp64 reference pass (ppwi={}, wgsize={})",
                    ppwi, wgsize
                );
                golden = Some(fasten::<f64>(&client, p, wgsize, ppwi).energies);
            }

            let sample = if precision_bits == 32 {
                fasten::<f32>(&client, p, wgsize, ppwi)
            } else {
                fasten::<f64>(&client, p, wgsize, ppwi)
            };

            let result = evaluate(p, sample, golden.as_deref());
            if !dumped {
                dumped = true;
                dump_results(p, &result);
            }
            results.push(result);
        }
    }

    if results.is_empty() {
        eprintln!("# no runnable configurations");
        return false;
    }

    if !p.csv {
        println!("results:");
    }
    for (i, r) in results.iter().enumerate() {
        if p.csv {
            show_csv(r, i == 0);
        } else {
            show_human_readable(p, r);
        }
    }

    let best = results
        .iter()
        .min_by(|a, b| a.ms.sum.partial_cmp(&b.ms.sum).unwrap())
        .unwrap();
    println!(
        "{}best: {{ min_ms: {:.3}, max_ms: {:.3}, sum_ms: {:.3}, avg_ms: {:.3}, ppwi: {}, wgsize: {} }}",
        if p.csv { "# " } else { "" },
        best.ms.min,
        best.ms.max,
        best.ms.sum,
        best.ms.mean,
        best.sample.ppwi,
        best.sample.wgsize
    );
    true
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (p, wgsizes, ppwis, precision_bits) = parse_params(&args);
    if !p.csv {
        println!("miniBUDE:  cubecl 0.10");
        println!("backend:   {}", backend::label());
        println!("precision: {} # PRECISION_BITS constant in src/main.rs", precision_bits);
        println!("ppwi:      {:?}", ppwis);
        println!("wgsize:    {:?}", wgsizes);
    }
    std::process::exit(if run(&p, &wgsizes, &ppwis, precision_bits) {
        0
    } else {
        1
    });
}
