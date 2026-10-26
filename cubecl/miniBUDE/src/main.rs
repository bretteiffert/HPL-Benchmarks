//! CubeCL miniBUDE benchmark with build-time CUDA or HIP backend selection
//! (`--features cuda` or `--features hip`).
//!
//! - Reads an input deck, then runs every (ppwi, wgsize) configuration given on
//!   the command line; configurations the pose count doesn't divide are skipped.
//! - Precision: float64 by default, float32 with `--float`. A float32 run first
//!   computes the energies once in float64 and compares against them.
//! - Correctness: per configuration, against that fp64 pass (float32 runs) and,
//!   informationally, against the deck's low-precision reference energies.
//! - Timing: each run is profiled with `Client::profile()`; on CUDA and HIP in
//!   CubeCL 0.11 this uses device events. The method used is in the header.

mod backend;
mod bude;
mod cli;
mod deck;
mod energy;
mod kernels;
#[allow(dead_code)]
mod metrics;
mod report;

use backend::BACKEND_LABEL;
use bude::{fasten, rates, Elem};
use cli::{Args, Precision};
use cubecl::client::Client;
use cubecl::prelude::*;
use deck::Deck;
use metrics::{compare_to_deck, compute_metrics};
use report::{ConfigResult, EnergyReport, Meta, RunReport};
use std::io::Write;

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

/// Keeps the (ppwi, wgsize) pairs that can run; returns them plus a note for
/// each one skipped.
fn runnable_configs(args: &Args, nposes: usize, client: &Client) -> (Vec<(usize, usize)>, Vec<String>) {
    let hw = &client.properties().hardware;
    let mut configs = Vec::new();
    let mut notes = Vec::new();
    for &ppwi in &args.ppwis {
        for &wgsize in &args.wgsizes {
            let per_cube = ppwi * wgsize;
            let skip = if wgsize > hw.max_units_per_cube as usize {
                Some(format!("wgsize exceeds the device limit of {}", hw.max_units_per_cube))
            } else if nposes < per_cube {
                Some(format!("{} poses are fewer than ppwi x wgsize = {}", nposes, per_cube))
            } else if nposes % per_cube != 0 {
                Some(format!("{} poses are not a multiple of ppwi x wgsize = {}", nposes, per_cube))
            } else if (nposes / per_cube) as u64 > hw.max_cube_count.0 as u64 {
                Some(format!("needs more cubes than the device limit of {}", hw.max_cube_count.0))
            } else {
                None
            };
            match skip {
                Some(why) => notes.push(format!("skipping ppwi {}, wgsize {}: {}", ppwi, wgsize, why)),
                None => configs.push((ppwi, wgsize)),
            }
        }
    }
    (configs, notes)
}

fn write_energies(path: &str, energies: &[f64]) {
    let write = || -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        for e in energies {
            writeln!(f, "{:.17e}", e)?;
        }
        f.flush()
    };
    write().unwrap_or_else(|e| fail(&format!("could not write {}: {}", path, e)));
}

fn run<F: Elem>(args: &Args, deck: &Deck) {
    let client = backend::device(args.device_index)
        .unwrap_or_else(|e| fail(&format!("device {}: {}", args.device_index, e)))
        .client();
    let props = client.properties();
    if !props.supports_type(F::elem_type_native()) {
        fail(&format!("{} is not supported on this device", F::PRECISION));
    }

    let (configs, mut notes) = runnable_configs(args, deck.nposes(), &client);
    if configs.is_empty() {
        fail(&format!("no runnable configurations:\n  {}", notes.join("\n  ")));
    }

    // A float32 run is compared against the same energies computed in float64.
    let mut differential = F::BITS != 64;
    if differential && !props.supports_type(f64::elem_type_native()) {
        differential = false;
        notes.push("float64 is not supported on this device, so there is no fp64 reference pass".into());
    }

    let (device, name_note) = backend::resolve_name(args.device_index, &client);
    notes.extend(name_note);
    let meta = Meta {
        backend: BACKEND_LABEL,
        device,
        driver: backend::resolve_driver(),
        precision: F::PRECISION,
        differential,
        deck: std::path::Path::new(&deck.dir)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| deck.dir.clone()),
        natlig: deck.natlig(),
        natpro: deck.natpro(),
        nposes: deck.nposes(),
        max_poses: deck.max_poses,
        ppwis: args.ppwis.clone(),
        wgsizes: args.wgsizes.clone(),
        iterations: args.iterations,
        warmup: args.warmup,
        timing: match props.timing_method {
            cubecl::profile::TimingMethod::Device => "device timestamps".to_string(),
            cubecl::profile::TimingMethod::System => {
                "host clock, device synced before and after each kernel".to_string()
            }
        },
        energy_enabled: args.energy,
        notes,
    };
    report::emit_header(&meta, args.output);

    // One untimed fp64 pass with the first runnable configuration.
    let golden = differential.then(|| {
        let (ppwi, wgsize) = configs[0];
        fasten::<f64>(&client, deck, ppwi, wgsize, 1, 0, args.device_index, false)
            .unwrap_or_else(|e| fail(&format!("fp64 reference pass: {}", e)))
            .energies
    });

    let mut results = Vec::with_capacity(configs.len());
    let mut energies_head = Vec::new();
    for (k, &(ppwi, wgsize)) in configs.iter().enumerate() {
        let s = fasten::<F>(
            &client,
            deck,
            ppwi,
            wgsize,
            args.iterations,
            args.warmup,
            args.device_index,
            args.energy,
        )
        .unwrap_or_else(|e| fail(&format!("ppwi {}, wgsize {}: {}", ppwi, wgsize, e)));

        if k == 0 {
            if let Some(path) = &args.out {
                write_energies(path, &s.energies);
            }
            energies_head = s.energies.iter().take(args.rows).copied().collect();
        }

        let r = rates(deck, &s);
        results.push(ConfigResult {
            ppwi,
            wgsize,
            gflops: r.gflops,
            gfinsts: r.gfinsts,
            interactions_per_s: r.interactions_per_s,
            launch_overhead_s: r.launch_overhead_s,
            context_s: s.context_s,
            energy: s.energy.as_ref().map(|e| EnergyReport {
                total_j: e.total_j,
                samples: e.samples,
                time_s: e.time_s,
                iterations: args.iterations,
            }),
            vs_reference: golden.as_deref().map(|g| compute_metrics(&s.energies, g, F::BITS)),
            vs_deck: compare_to_deck(&s.energies, &deck.ref_energies),
            times: s.kernel_s,
        });
    }

    let report = RunReport {
        meta,
        configs: results,
        energies_head,
    };
    report::emit_results(&report, args.output);
}

fn main() {
    let args = cli::parse_arguments();
    if args.list {
        list_devices();
        return;
    }
    let deck = deck::read_deck(&args.deck, args.poses).unwrap_or_else(|e| fail(&e));
    match args.precision {
        Precision::Float32 => run::<f32>(&args, &deck),
        Precision::Float64 => run::<f64>(&args, &deck),
    }
}
