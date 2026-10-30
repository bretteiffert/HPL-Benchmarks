//! Device upload, timed runs of one (ppwi, wgsize) configuration, and the
//! derived rates. The op and instruction counts are the original formulas.

use crate::deck::Deck;
use crate::energy::EnergyProfiler;
use crate::kernels::fasten_main;
use crate::backend::{BackendRuntime, Client};
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::time::Instant;

pub trait Elem: Float + CubeElement + Copy + Send + 'static {
    const PRECISION: &'static str;
    /// 32 or 64; selects f32 or f64 ULP distances in the metrics.
    const BITS: usize;
    fn from_f32(x: f32) -> Self;
    fn as_f64(self) -> f64;
}

impl Elem for f32 {
    const PRECISION: &'static str = "float32";
    const BITS: usize = 32;
    fn from_f32(x: f32) -> Self {
        x
    }
    fn as_f64(self) -> f64 {
        self as f64
    }
}

impl Elem for f64 {
    const PRECISION: &'static str = "float64";
    const BITS: usize = 64;
    fn from_f32(x: f32) -> Self {
        x as f64
    }
    fn as_f64(self) -> f64 {
        self
    }
}

/// Device copies of the deck in precision `F`, plus the output buffer.
#[derive(Clone)]
struct Buffers {
    out: Handle,
    prot: [Handle; 4],
    lig: [Handle; 4],
    poses: [Handle; 6],
    ff: [Handle; 4],
    natpro: usize,
    natlig: usize,
    nff: usize,
    n: usize,
}

impl Buffers {
    fn upload<F: Elem>(client: &Client, deck: &Deck) -> Self {
        let up = |xs: Vec<F>| client.create_from_slice(F::as_bytes(&xs));
        let up_u32 = |xs: Vec<u32>| client.create_from_slice(u32::as_bytes(&xs));
        let atoms = |a: &[crate::deck::Atom]| {
            [
                up(a.iter().map(|a| F::from_f32(a.x)).collect()),
                up(a.iter().map(|a| F::from_f32(a.y)).collect()),
                up(a.iter().map(|a| F::from_f32(a.z)).collect()),
                up_u32(a.iter().map(|a| a.ty as u32).collect()),
            ]
        };
        let ff = &deck.forcefield;
        let pose = |k: usize| up(deck.poses[k].iter().map(|&x| F::from_f32(x)).collect());
        Self {
            out: client.empty(deck.nposes() * std::mem::size_of::<F>()),
            prot: atoms(&deck.protein),
            lig: atoms(&deck.ligand),
            poses: [pose(0), pose(1), pose(2), pose(3), pose(4), pose(5)],
            ff: [
                up_u32(ff.iter().map(|f| f.hbtype as u32).collect()),
                up(ff.iter().map(|f| F::from_f32(f.radius)).collect()),
                up(ff.iter().map(|f| F::from_f32(f.hphb)).collect()),
                up(ff.iter().map(|f| F::from_f32(f.elsc)).collect()),
            ],
            natpro: deck.natpro(),
            natlig: deck.natlig(),
            nff: ff.len(),
            n: deck.nposes(),
        }
    }

    fn launch<F: Elem>(&self, client: &Client, ppwi: usize, wgsize: usize) {
        let blocks = self.n.div_ceil(ppwi).div_ceil(wgsize) as u32;
        let (np, nl, nf, n) = (self.natpro, self.natlig, self.nff, self.n);
        let arg = |h: &Handle, len: usize| unsafe { ArrayArg::<BackendRuntime>::from_raw_parts(h.clone(), len) };
        unsafe {
            fasten_main::launch_unchecked::<F, BackendRuntime>(
                client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim::new_1d(wgsize as u32),
                arg(&self.out, n),
                arg(&self.prot[0], np),
                arg(&self.prot[1], np),
                arg(&self.prot[2], np),
                arg(&self.prot[3], np),
                arg(&self.lig[0], nl),
                arg(&self.lig[1], nl),
                arg(&self.lig[2], nl),
                arg(&self.lig[3], nl),
                arg(&self.poses[0], n),
                arg(&self.poses[1], n),
                arg(&self.poses[2], n),
                arg(&self.poses[3], n),
                arg(&self.poses[4], n),
                arg(&self.poses[5], n),
                arg(&self.ff[0], nf),
                arg(&self.ff[1], nf),
                arg(&self.ff[2], nf),
                arg(&self.ff[3], nf),
                nl as u32,
                np as u32,
                n as u32,
                ppwi,
            );
        }
    }
}

/// Board energy over the measured (post-warm-up) iterations.
pub struct EnergySample {
    pub total_j: f64,
    pub samples: u32,
    pub time_s: f64,
}

pub struct Sample {
    pub ppwi: usize,
    pub wgsize: usize,
    pub energies: Vec<f64>,
    /// Kernel time per measured iteration (warm-up dropped), seconds.
    pub kernel_s: Vec<f64>,
    /// Host wall time per measured iteration, seconds.
    pub wall_s: Vec<f64>,
    /// Time to upload the deck, seconds.
    pub context_s: f64,
    pub energy: Option<EnergySample>,
}

/// Uploads the deck and runs `warmup + iterations` launches of one
/// configuration, timing each with `client.profile()`.
#[allow(clippy::too_many_arguments)]
pub fn fasten<F: Elem>(
    client: &Client,
    deck: &Deck,
    ppwi: usize,
    wgsize: usize,
    iterations: usize,
    warmup: usize,
    device_index: usize,
    measure_energy: bool,
) -> Result<Sample, String> {
    let context_start = Instant::now();
    let bufs = Buffers::upload::<F>(client, deck);
    cubecl::future::block_on(client.sync()).map_err(|e| format!("deck upload failed: {:?}", e))?;
    let context_s = context_start.elapsed().as_secs_f64();

    let mut profiler = if measure_energy {
        Some(EnergyProfiler::new(device_index)?)
    } else {
        None
    };
    let mut measured_started = None;
    let mut kernel_s = Vec::with_capacity(iterations);
    let mut wall_s = Vec::with_capacity(iterations);

    for iteration in 0..warmup + iterations {
        if iteration == warmup {
            if let Some(p) = profiler.as_mut() {
                p.start();
            }
            measured_started = Some(Instant::now());
        }
        let wall_start = Instant::now();
        let (inner, b) = (client.clone(), bufs.clone());
        let (_, profile) = client
            .profile(move || b.launch::<F>(&inner, ppwi, wgsize), "fasten_main")
            .map_err(|e| format!("cubecl profile() call failed: {:?}", e))?;
        // `None` means the device recorded no measurement for this window; a
        // zero would silently win every min(), so treat it as a hard error.
        let secs = cubecl::future::block_on(profile.resolve())
            .duration()
            .as_secs_f64();
        let wall = wall_start.elapsed().as_secs_f64();
        if iteration >= warmup {
            kernel_s.push(secs);
            wall_s.push(wall);
        }
    }

    let energy = profiler.as_mut().map(|p| {
        let (total_j, samples) = p.stop_and_read();
        EnergySample {
            total_j,
            samples,
            time_s: measured_started.map_or(0.0, |s: Instant| s.elapsed().as_secs_f64()),
        }
    });

    let bytes = client
        .read_one(bufs.out.clone())
        .map_err(|e| format!("reading device results failed: {:?}", e))?;
    let energies = F::from_bytes(&bytes)
        .iter()
        .take(deck.nposes())
        .map(|v| v.as_f64())
        .collect();

    Ok(Sample {
        ppwi,
        wgsize,
        energies,
        kernel_s,
        wall_s,
        context_s,
        energy,
    })
}

/// Throughput derived from the mean measured kernel time.
pub struct Rates {
    pub gflops: f64,
    pub gfinsts: f64,
    pub interactions_per_s: f64,
    /// Mean host wall time minus kernel time, seconds.
    pub launch_overhead_s: f64,
}

pub fn rates(deck: &Deck, s: &Sample) -> Rates {
    let runtime = s.kernel_s.iter().sum::<f64>() / s.kernel_s.len() as f64;
    let (natlig, natpro, nposes, ppwi) = (deck.natlig(), deck.natpro(), deck.nposes(), s.ppwi);

    let ops_per_wg = ppwi * 27 + natlig * (2 + ppwi * 18 + natpro * (10 + ppwi * 30)) + ppwi;
    let total_ops = ops_per_wg as f64 * (nposes as f64 / ppwi as f64);
    let total_finsts = (25 * natpro * natlig * nposes) as f64;
    let interactions = (nposes * natlig * natpro) as f64;
    let overhead = s
        .wall_s
        .iter()
        .zip(&s.kernel_s)
        .map(|(w, k)| w - k)
        .sum::<f64>()
        / s.kernel_s.len() as f64;

    Rates {
        gflops: total_ops / runtime / 1e9,
        gfinsts: total_finsts / runtime / 1e9,
        interactions_per_s: interactions / runtime,
        launch_overhead_s: overhead,
    }
}
