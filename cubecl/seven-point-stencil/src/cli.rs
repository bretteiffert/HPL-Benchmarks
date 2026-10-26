//! Command-line parsing.

use crate::report::OutputMode;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Float16,
    Float32,
    Float64,
}

pub const DEFAULT_SIZE: u32 = 1024;
pub const DEFAULT_BLOCK: (u32, u32, u32) = (64, 4, 1);
pub const DEFAULT_NUM_TIMES: u32 = 100;

pub struct Args {
    pub list: bool,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub num_times: u32,
    pub device_index: usize,
    pub precision: Precision,
    pub mibibytes: bool,
    pub output: OutputMode,
    pub energy: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            list: false,
            grid: (DEFAULT_SIZE, DEFAULT_SIZE, DEFAULT_SIZE),
            block: DEFAULT_BLOCK,
            num_times: DEFAULT_NUM_TIMES,
            device_index: 0,
            precision: Precision::Float64,
            mibibytes: false,
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
    println!("      --list               List available devices");
    println!("      --device     INDEX   Select device at INDEX (default 0)");
    println!("  -s  --size       L       Use an L x L x L grid (default {}), L >= 3", DEFAULT_SIZE);
    println!("      --grid       X,Y,Z   Use an X x Y x Z grid instead, each >= 3");
    println!("      --block-size X,Y,Z");
    println!(
        "                           Pinned units per cube; must be {},{},{}",
        DEFAULT_BLOCK.0, DEFAULT_BLOCK.1, DEFAULT_BLOCK.2
    );
    println!("  -n  --numtimes   NUM     Timed iterations, NUM >= 1 (default {});", DEFAULT_NUM_TIMES);
    println!("                           one untimed warm-up run comes first");
    println!("      --half               Use float16");
    println!("      --float              Use float32");
    println!("      --double             Use float64 (default)");
    println!("      --mibibytes          Report sizes and bandwidth in MiB (2^20) instead of MB (10^6);");
    println!("                           CSV bandwidth is always GB/s");
    println!("      --csv                CSV to stdout: one summary row");
    println!("      --csv-raw            CSV to stdout: one row per timed iteration");
    println!("      --no-energy          Skip energy measurement");
    println!();
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

/// Parses "X", "X,Y" or "X,Y,Z" (also accepts 'x' as a separator), filling
/// missing trailing dimensions with `fill`.
fn dims(s: &str, flag: &str, fill: u32) -> (u32, u32, u32) {
    let parts: Vec<&str> = s.split([',', 'x']).map(str::trim).collect();
    if parts.is_empty() || parts.len() > 3 {
        fail(&format!("{} takes 1 to 3 comma-separated values, got '{}'", flag, s));
    }
    let mut out = [fill; 3];
    for (slot, p) in out.iter_mut().zip(&parts) {
        *slot = p
            .parse()
            .unwrap_or_else(|_| fail(&format!("invalid value '{}' for {}", s, flag)));
    }
    (out[0], out[1], out[2])
}

pub fn parse_arguments() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut args = Args::default();
    let mut csv = false;
    let mut csv_raw = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--list" => args.list = true,
            "--device" => args.device_index = value(&argv, &mut i, "--device"),
            "--size" | "-s" => {
                let l: u32 = value(&argv, &mut i, "--size");
                args.grid = (l, l, l);
            }
            "--grid" => {
                let s = raw(&argv, &mut i, "--grid");
                let g = dims(s, "--grid", 0);
                if s.split([',', 'x']).count() != 3 {
                    fail("--grid takes three values: X,Y,Z");
                }
                args.grid = g;
            }
            "--block-size" => {
                let s = raw(&argv, &mut i, "--block-size");
                args.block = dims(s, "--block-size", 1);
            }
            "--numtimes" | "-n" => {
                let v: u32 = value(&argv, &mut i, "--numtimes");
                if v < 1 {
                    fail("number of times must be 1 or more");
                }
                args.num_times = v;
            }
            "--half" => args.precision = Precision::Float16,
            "--float" => args.precision = Precision::Float32,
            "--double" => args.precision = Precision::Float64,
            "--mibibytes" => args.mibibytes = true,
            "--csv" => csv = true,
            "--csv-raw" => csv_raw = true,
            "--no-energy" => args.energy = false,
            "--help" | "-h" => {
                print_help(&argv[0]);
                std::process::exit(0);
            }
            other => fail(&format!("unrecognized argument '{}'", other)),
        }
        i += 1;
    }

    let (x, y, z) = args.grid;
    if x < 3 || y < 3 || z < 3 {
        fail("every grid dimension must be at least 3 (the stencil needs an interior)");
    }
    let (bx, by, bz) = args.block;
    if bx == 0 || by == 0 || bz == 0 {
        fail("block dimensions must be greater than 0");
    }
    if args.block != DEFAULT_BLOCK {
        fail("the pinned configuration requires --block-size 64,4,1");
    }

    args.output = match (csv, csv_raw) {
        (true, true) => fail("--csv and --csv-raw are mutually exclusive"),
        (true, false) => OutputMode::Csv,
        (false, true) => OutputMode::CsvRaw,
        (false, false) => OutputMode::Text,
    };
    args
}
