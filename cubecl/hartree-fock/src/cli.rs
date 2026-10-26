//! Command-line parsing.

use crate::report::OutputMode;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Float32,
    Float64,
}

pub const DEFAULT_BLOCK_SIZE: u32 = 256;
pub const DEFAULT_NUM_TIMES: u32 = 10;
pub const DEFAULT_MAX_CHECK_ATOMS: usize = 128;

/// When to run the fp64 CPU reference check.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Check {
    /// Up to `--max-check-atoms` atoms (the reference is O(natoms⁴)).
    Auto,
    Always,
    Never,
}

pub struct Args {
    pub list: bool,
    pub input: Option<String>,
    pub num_times: u32,
    pub block_size: u32,
    pub device_index: usize,
    pub precision: Precision,
    pub legacy_sqrt: bool,
    pub check: Check,
    pub max_check_atoms: usize,
    pub output: OutputMode,
    pub energy: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            list: false,
            input: None,
            num_times: DEFAULT_NUM_TIMES,
            block_size: DEFAULT_BLOCK_SIZE,
            device_index: 0,
            precision: Precision::Float64,
            legacy_sqrt: false,
            check: Check::Auto,
            max_check_atoms: DEFAULT_MAX_CHECK_ATOMS,
            output: OutputMode::Text,
            energy: true,
        }
    }
}

fn print_help(prog: &str) {
    println!("Usage: {} INPUT [OPTIONS]", prog);
    println!();
    println!("  INPUT                    Input file, e.g. ../../data/hartree-fock/he64");
    println!();
    println!("Options:");
    println!("  -h  --help               Print this message");
    println!("      --list               List available devices");
    println!("      --device     INDEX   Select device at INDEX (default 0)");
    println!("  -n  --numtimes   NUM     Timed iterations, NUM >= 1 (default {});", DEFAULT_NUM_TIMES);
    println!("                           one untimed run comes first");
    println!("      --block-size SIZE    Pinned units per cube; SIZE must be {}", DEFAULT_BLOCK_SIZE);
    println!("      --float              Use float32");
    println!("      --double             Use float64 (default)");
    println!("      --check              Always run the fp64 CPU reference check");
    println!("      --no-check           Never run it");
    println!(
        "      --max-check-atoms N  Run it automatically up to N atoms (default {})",
        DEFAULT_MAX_CHECK_ATOMS
    );
    println!("      --csv                CSV to stdout: one summary row");
    println!("      --csv-raw            CSV to stdout: one row per timed iteration");
    println!("      --no-energy          Skip energy measurement");
    println!();
    println!("The pinned configuration uses native-precision square roots.");
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

fn value<T: std::str::FromStr>(argv: &[String], i: &mut usize, flag: &str) -> T {
    *i += 1;
    let s = argv
        .get(*i)
        .unwrap_or_else(|| fail(&format!("missing value for {}", flag)));
    s.parse()
        .unwrap_or_else(|_| fail(&format!("invalid value '{}' for {}", s, flag)))
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
            "--numtimes" | "-n" => {
                let v: u32 = value(&argv, &mut i, "--numtimes");
                if v < 1 {
                    fail("number of times must be 1 or more");
                }
                args.num_times = v;
            }
            "--block-size" => {
                let v: u32 = value(&argv, &mut i, "--block-size");
                if v == 0 {
                    fail("block size must be greater than 0");
                }
                args.block_size = v;
            }
            "--float" => args.precision = Precision::Float32,
            "--double" => args.precision = Precision::Float64,
            "--legacy-sqrt" => fail(
                "the pinned configuration requires native square roots; --legacy-sqrt is not supported",
            ),
            "--check" => args.check = Check::Always,
            "--no-check" => args.check = Check::Never,
            "--max-check-atoms" => args.max_check_atoms = value(&argv, &mut i, "--max-check-atoms"),
            "--csv" => csv = true,
            "--csv-raw" => csv_raw = true,
            "--no-energy" => args.energy = false,
            "--help" | "-h" => {
                print_help(&argv[0]);
                std::process::exit(0);
            }
            other if other.starts_with('-') => fail(&format!("unrecognized argument '{}'", other)),
            path => {
                if args.input.is_some() {
                    fail(&format!("unexpected extra argument '{}'", path));
                }
                args.input = Some(path.to_string());
            }
        }
        i += 1;
    }
    if args.input.is_none() && !args.list {
        fail("no input file given");
    }
    if args.block_size != DEFAULT_BLOCK_SIZE {
        fail("the pinned configuration requires --block-size 256");
    }
    args.output = match (csv, csv_raw) {
        (true, true) => fail("--csv and --csv-raw are mutually exclusive"),
        (true, false) => OutputMode::Csv,
        (false, true) => OutputMode::CsvRaw,
        (false, false) => OutputMode::Text,
    };
    args
}
