//! Command-line parsing. Flag names follow upstream miniBUDE (`-i`, `-n` for
//! poses, `-p`, `-w`, `--deck`, ...); precision, CSV and energy flags match the
//! other benchmarks.

use crate::report::OutputMode;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Float32,
    Float64,
}

pub const DEFAULT_ITERS: usize = 8;
pub const DEFAULT_WARMUP: usize = 2;
pub const DEFAULT_ROWS: usize = 8;
pub const DEFAULT_DECK: &str = "../../data/miniBUDE/bm1";
pub const PPWIS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
pub const PINNED_PPWI: usize = 1;
pub const PINNED_WGSIZE: usize = 256;

pub struct Args {
    pub list: bool,
    pub deck: String,
    /// 0 = every pose in the deck.
    pub poses: usize,
    pub iterations: usize,
    pub warmup: usize,
    pub ppwis: Vec<usize>,
    pub wgsizes: Vec<usize>,
    pub rows: usize,
    pub out: Option<String>,
    pub device_index: usize,
    pub precision: Precision,
    pub output: OutputMode,
    pub energy: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            list: false,
            deck: DEFAULT_DECK.to_string(),
            poses: 0,
            iterations: DEFAULT_ITERS,
            warmup: DEFAULT_WARMUP,
            ppwis: vec![PINNED_PPWI],
            wgsizes: vec![PINNED_WGSIZE],
            rows: DEFAULT_ROWS,
            out: None,
            device_index: 0,
            precision: Precision::Float32,
            output: OutputMode::Text,
            energy: true,
        }
    }
}

fn print_help(prog: &str) {
    println!("Usage: {} [OPTIONS]", prog);
    println!();
    println!("Options:");
    println!("  -h  --help               Print this message");
    println!("  -l  --list               List available devices");
    println!("  -d  --device     INDEX   Select device at INDEX (default 0)");
    println!("      --deck       DIR     Input deck directory (default {})", DEFAULT_DECK);
    println!("  -n  --poses      N       Compute energies for the first N poses (default 0 = all)");
    println!("  -i  --iter       I       Timed iterations per configuration (default {})", DEFAULT_ITERS);
    println!("      --warmup     W       Untimed iterations before them (default {})", DEFAULT_WARMUP);
    println!("  -p  --ppwi       N       Pinned poses per work-item; N must be {}", PINNED_PPWI);
    println!("  -w  --wgsize     N       Pinned work-group size; N must be {}", PINNED_WGSIZE);
    println!("      --float              Use float32 (default); also runs an fp64 reference pass");
    println!("      --double             Use optional float64 mode");
    println!("  -r  --rows       N       Show the first N pose energies (default {})", DEFAULT_ROWS);
    println!("  -o  --out        PATH    Save all pose energies of the first configuration to PATH");
    println!("      --csv                CSV to stdout: one summary row per configuration");
    println!("      --csv-raw            CSV to stdout: one row per configuration per timed iteration");
    println!("      --no-energy          Skip energy measurement");
    println!();
    println!("The pose count must be a multiple of {}.", PINNED_PPWI * PINNED_WGSIZE);
    println!("In the CSV modes, the header, energy and correctness sections go to stderr.");
    println!();
    println!("Environment:");
    println!("  GPU_NAME=S       Override the reported device name");
    println!("  GPU_DRIVER=S     Override the reported driver string");
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {} (try '--help')", msg);
    std::process::exit(1);
}

fn raw<'a>(argv: &'a [String], i: &mut usize, flag: &str) -> &'a str {
    *i += 1;
    argv.get(*i)
        .map(String::as_str)
        .unwrap_or_else(|| fail(&format!("missing value for {}", flag)))
}

fn value<T: std::str::FromStr>(argv: &[String], i: &mut usize, flag: &str) -> T {
    let s = raw(argv, i, flag);
    s.parse()
        .unwrap_or_else(|_| fail(&format!("invalid value '{}' for {}", s, flag)))
}

fn list(s: &str, flag: &str) -> Vec<usize> {
    s.split(',')
        .map(|t| {
            t.trim()
                .parse::<usize>()
                .unwrap_or_else(|_| fail(&format!("invalid value '{}' in {}", t, flag)))
        })
        .collect()
}

pub fn parse_arguments() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut args = Args::default();
    let mut csv = false;
    let mut csv_raw = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--list" | "-l" | "list" => args.list = true,
            "--device" | "-d" => args.device_index = value(&argv, &mut i, "--device"),
            "--deck" => args.deck = raw(&argv, &mut i, "--deck").to_string(),
            "--poses" | "-n" => args.poses = value(&argv, &mut i, "--poses"),
            "--iter" | "-i" => {
                args.iterations = value(&argv, &mut i, "--iter");
                if args.iterations == 0 {
                    fail("--iter must be at least 1");
                }
            }
            "--warmup" => args.warmup = value(&argv, &mut i, "--warmup"),
            "--ppwi" | "-p" => {
                let s = raw(&argv, &mut i, "--ppwi");
                args.ppwis = if s == "all" { PPWIS.to_vec() } else { list(s, "--ppwi") };
                if let Some(v) = args.ppwis.iter().find(|v| !PPWIS.contains(v)) {
                    fail(&format!("ppwi {} is not supported; use one of {:?}", v, PPWIS));
                }
            }
            "--wgsize" | "-w" => {
                args.wgsizes = list(raw(&argv, &mut i, "--wgsize"), "--wgsize");
                if args.wgsizes.contains(&0) {
                    fail("work-group sizes must be greater than 0");
                }
            }
            "--float" => args.precision = Precision::Float32,
            "--double" => args.precision = Precision::Float64,
            "--rows" | "-r" => args.rows = value(&argv, &mut i, "--rows"),
            "--out" | "-o" => args.out = Some(raw(&argv, &mut i, "--out").to_string()),
            "--csv" => csv = true,
            "--csv-raw" => csv_raw = true,
            "--no-energy" => args.energy = false,
            "--help" | "-h" | "help" => {
                print_help(&argv[0]);
                std::process::exit(0);
            }
            other => fail(&format!("unrecognized argument '{}'", other)),
        }
        i += 1;
    }
    args.output = match (csv, csv_raw) {
        (true, true) => fail("--csv and --csv-raw are mutually exclusive"),
        (true, false) => OutputMode::Csv,
        (false, true) => OutputMode::CsvRaw,
        (false, false) => OutputMode::Text,
    };
    if args.ppwis.len() != 1 || args.ppwis[0] != PINNED_PPWI {
        fail("the pinned configuration requires --ppwi 1");
    }
    if args.wgsizes.len() != 1 || args.wgsizes[0] != PINNED_WGSIZE {
        fail("the pinned configuration requires --wgsize 256");
    }
    args
}
