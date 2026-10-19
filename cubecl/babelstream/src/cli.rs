//! Command-line parsing.

use crate::report::OutputMode;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Float16,
    Float32,
    Float64,
}

pub const DEFAULT_BLOCK_SIZE: u32 = 1024;

pub struct Args {
    pub list: bool,
    pub array_size: u32,
    pub num_times: u32,
    pub device_index: usize,
    pub block_size: u32,
    pub precision: Precision,
    pub mibibytes: bool,
    pub output: OutputMode,
    pub energy: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            list: false,
            array_size: 33_554_432,
            num_times: 1000,
            device_index: 0,
            block_size: DEFAULT_BLOCK_SIZE,
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
    println!("  -s  --arraysize  SIZE    Use SIZE elements per array (default 33554432);");
    println!("                           must be a multiple of the block size");
    println!("  -n  --numtimes   NUM     Run the test NUM times, NUM >= 2 (default 1000);");
    println!("                           the first run is excluded from statistics");
    println!("      --block-size SIZE    Units per cube for every kernel (default {});", DEFAULT_BLOCK_SIZE);
    println!("                           a multiple of the plane size, at most 32 planes");
    println!("      --half               Use float16");
    println!("      --float              Use float32");
    println!("      --double             Use float64 (default)");
    println!("      --mibibytes          Report sizes and bandwidth in MiB (2^20) instead of MB (10^6);");
    println!("                           CSV bandwidth is always GB/s");
    println!("      --csv                CSV to stdout: one summary row per routine");
    println!("      --csv-raw            CSV to stdout: one row per routine per measured iteration");
    println!("      --no-energy          Skip energy measurement");
    println!();
    println!("In the CSV modes, the header, energy and correctness sections go to stderr.");
    println!();
    println!("Environment:");
    println!("  FMAD=0|1         Select the multiply-add code path (default 1)");
    println!("  GPU_NAME=S       Override the reported device name");
    println!("  GPU_DRIVER=S     Override the reported driver string");
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {} (try '--help')", msg);
    std::process::exit(1);
}

fn value<T: std::str::FromStr>(argv: &[String], i: &mut usize, flag: &str) -> T {
    *i += 1;
    argv.get(*i)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| fail(&format!("invalid or missing value for {}", flag)))
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
            "--arraysize" | "-s" => {
                let v: u32 = value(&argv, &mut i, "--arraysize");
                if v == 0 {
                    fail("array size must be greater than 0");
                }
                args.array_size = v;
            }
            "--numtimes" | "-n" => {
                let v: u32 = value(&argv, &mut i, "--numtimes");
                if v < 2 {
                    fail("number of times must be 2 or more");
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
    args.output = match (csv, csv_raw) {
        (true, true) => fail("--csv and --csv-raw are mutually exclusive"),
        (true, false) => OutputMode::Csv,
        (false, true) => OutputMode::CsvRaw,
        (false, false) => OutputMode::Text,
    };
    args
}
