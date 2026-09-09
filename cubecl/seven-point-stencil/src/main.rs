//! CubeCL seven-point Laplacian benchmark with configurable cube dimensions, device-profiled timing, and analytical correctness metrics.
//! It defaults to float64 and selects exactly one CUDA or HIP runtime through Cargo features.
//! GPU_NAME supplies the reported device label when set and otherwise falls back to unknown-gpu.

use cubecl::prelude::*;
use cubecl::server::Handle;

#[cfg(all(feature = "cuda", not(feature = "hip")))]
type Rt = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "cuda", not(feature = "hip")))]
const BACKEND_LABEL: &str = "CUDA";

#[cfg(all(feature = "hip", not(feature = "cuda")))]
type Rt = cubecl::hip::HipRuntime;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
const BACKEND_LABEL: &str = "HIP";

#[cfg(all(feature = "cuda", feature = "hip"))]
compile_error!("enable exactly one of the `cuda` or `hip` features, not both");
#[cfg(not(any(feature = "cuda", feature = "hip")))]
compile_error!("enable exactly one of the `cuda` or `hip` features");

type Precision = f64;
const PRECISION_STR: &str = "double";

const TB_SIZE: u32 = 1024;
const L: u32 = 1024;
const NUM_ITER: usize = 100;

#[cube(launch_unchecked)]
fn test_function_kernel<F: Float + CubeElement>(
    u: &mut Array<F>,
    nx: u32,
    ny: u32,
    nz: u32,
    hx: F,
    hy: F,
    hz: F,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    let k = ABSOLUTE_POS_Z;

    if i < nx && j < ny && k < nz {
        let pos = (i + nx * (j + ny * k)) as usize;

        let c = F::new(0.5);
        let x = F::cast_from(i) * hx;
        let y = F::cast_from(j) * hy;
        let z = F::cast_from(k) * hz;
        let lx = F::cast_from(nx) * hx;
        let ly = F::cast_from(ny) * hy;
        let lz = F::cast_from(nz) * hz;
        u[pos] = c * x * (x - lx) + c * y * (y - ly) + c * z * (z - lz);
    }
}

#[cube(launch_unchecked)]
fn laplacian_kernel<F: Float + CubeElement>(
    f: &mut Array<F>,
    u: &Array<F>,
    nx: u32,
    ny: u32,
    nz: u32,
    invhx2: F,
    invhy2: F,
    invhz2: F,
    invhxyz2: F,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    let k = ABSOLUTE_POS_Z;

    if i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1 {
        let nx = nx as usize;
        let ny = ny as usize;
        let i = i as usize;
        let j = j as usize;
        let k = k as usize;
        let slice = nx * ny;
        let pos = i + nx * j + slice * k;

        f[pos] = u[pos] * invhxyz2
            + (u[pos - 1] + u[pos + 1]) * invhx2
            + (u[pos - nx] + u[pos + nx]) * invhy2
            + (u[pos - slice] + u[pos + slice]) * invhz2;
    }
}

type Client = ComputeClient<Rt>;

fn test_function(
    client: &Client,
    d_u: &Handle,
    n: usize,
    nx: u32,
    ny: u32,
    nz: u32,
    hx: Precision,
    hy: Precision,
    hz: Precision,
) {
    let gx = (nx - 1) / TB_SIZE + 1;

    unsafe {
        test_function_kernel::launch_unchecked::<Precision, Rt>(
            client,
            CubeCount::Static(gx, ny, nz),
            CubeDim::new_1d(TB_SIZE),
            ArrayArg::from_raw_parts(d_u.clone(), n),
            nx,
            ny,
            nz,
            hx,
            hy,
            hz,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn laplacian(
    client: &Client,
    d_f: &Handle,
    d_u: &Handle,
    n: usize,
    nx: u32,
    ny: u32,
    nz: u32,
    blk_x: u32,
    blk_y: u32,
    blk_z: u32,
    hx: Precision,
    hy: Precision,
    hz: Precision,
) {
    let gx = (nx - 1) / blk_x + 1;
    let gy = (ny - 1) / blk_y + 1;
    let gz = (nz - 1) / blk_z + 1;

    let invhx2 = 1.0 / hx / hx;
    let invhy2 = 1.0 / hy / hy;
    let invhz2 = 1.0 / hz / hz;
    let invhxyz2 = -2.0 * (invhx2 + invhy2 + invhz2);

    unsafe {
        laplacian_kernel::launch_unchecked::<Precision, Rt>(
            client,
            CubeCount::Static(gx, gy, gz),
            CubeDim::new_3d(blk_x, blk_y, blk_z),
            ArrayArg::from_raw_parts(d_f.clone(), n),
            ArrayArg::from_raw_parts(d_u.clone(), n),
            nx,
            ny,
            nz,
            invhx2,
            invhy2,
            invhz2,
            invhxyz2,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn time_one_iteration(
    client: &Client,
    d_f: &Handle,
    d_u: &Handle,
    n: usize,
    nx: u32,
    ny: u32,
    nz: u32,
    blk_x: u32,
    blk_y: u32,
    blk_z: u32,
    hx: Precision,
    hy: Precision,
    hz: Precision,
) -> f64 {
    cubecl::future::block_on(client.sync());

    let (_, profile_duration) = client
        .profile(
            || {
                laplacian(
                    client, d_f, d_u, n, nx, ny, nz, blk_x, blk_y, blk_z, hx, hy, hz,
                );
            },
            "laplacian",
        )
        .expect("client.profile failed");

    let ticks = cubecl::future::block_on(profile_duration.resolve());
    ticks.duration().as_secs_f64() * 1e3
}

struct CorrectnessResult {
    max_abs_err: f64,
    l2_rel_err: f64,
    mean_abs_err: f64,
    num_checked: usize,
}

const ELEM_SIZE: usize = std::mem::size_of::<Precision>();

#[inline(always)]
fn elem_at(bytes: &[u8], idx: usize) -> f64 {
    let mut buf = [0u8; ELEM_SIZE];
    buf.copy_from_slice(&bytes[idx * ELEM_SIZE..(idx + 1) * ELEM_SIZE]);
    Precision::from_ne_bytes(buf) as f64
}

fn check_correctness(bytes: &[u8], nx: usize, ny: usize, nz: usize) -> CorrectnessResult {
    let expected = 3.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq_err = 0.0_f64;
    let mut sum_sq_exact = 0.0_f64;
    let mut max_err = 0.0_f64;
    let mut count = 0usize;

    for k in 1..nz.saturating_sub(1) {
        for j in 1..ny.saturating_sub(1) {
            for i in 1..nx.saturating_sub(1) {
                let pos = (i + nx * (j + ny * k)) as usize;
                let val = elem_at(bytes, pos);
                let err = (val - expected).abs();
                sum_abs += err;
                sum_sq_err += err * err;
                sum_sq_exact += expected * expected;
                if err > max_err {
                    max_err = err;
                }
                count += 1;
            }
        }
    }

    if count > 0 {
        CorrectnessResult {
            max_abs_err: max_err,
            mean_abs_err: sum_abs / count as f64,
            l2_rel_err: if sum_sq_exact > 0.0 {
                (sum_sq_err / sum_sq_exact).sqrt()
            } else {
                sum_sq_err.sqrt()
            },
            num_checked: count,
        }
    } else {
        CorrectnessResult {
            max_abs_err: 0.0,
            l2_rel_err: 0.0,
            mean_abs_err: 0.0,
            num_checked: 0,
        }
    }
}

fn g(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let exp = x.abs().log10().floor() as i32;
    if exp < -4 || exp >= 6 {
        let mantissa = x / 10f64.powi(exp);
        return format!(
            "{:.5}e{}{:02}",
            mantissa,
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        );
    } else {
        let decimals = (5 - exp).max(0) as usize;
        let s = format!("{:.*}", decimals, x);
        let s = if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        };
        s
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();

    let mut blk_x = TB_SIZE;
    let mut blk_y = 1u32;
    let mut blk_z = 1u32;

    let mut nx = L;
    let mut ny = L;
    let mut nz = L;
    let mut csv_output = false;

    if argv.len() > 1 {
        nx = argv[1].parse().unwrap();
    }
    if argv.len() > 2 {
        ny = argv[2].parse().unwrap();
    }
    if argv.len() > 3 {
        nz = argv[3].parse().unwrap();
    }
    if argv.len() > 4 {
        blk_x = argv[4].parse().unwrap();
    }
    if argv.len() > 5 {
        blk_y = argv[5].parse().unwrap();
    }
    if argv.len() > 6 {
        blk_z = argv[6].parse().unwrap();
    }
    if argv.len() > 7 && argv[7] == "--csv" {
        csv_output = true;
    }

    let (nxu, nyu, nzu) = (nx as usize, ny as usize, nz as usize);

    let theoretical_fetch_size = (nxu * nyu * nzu
        - 8
        - 4 * (nxu - 2)
        - 4 * (nyu - 2)
        - 4 * (nzu - 2))
        * ELEM_SIZE;
    let theoretical_write_size = (nxu - 2) * (nyu - 2) * (nzu - 2) * ELEM_SIZE;
    let datasize = (theoretical_fetch_size + theoretical_write_size) as f64;

    let gpu_name = std::env::var("GPU_NAME").unwrap_or_else(|_| "unknown-gpu".to_string());
    let client = Rt::client(&Default::default());

    if csv_output {
        println!("backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs");
    } else {
        println!("{}", gpu_name);
        println!("Precision: {}", PRECISION_STR);
        println!("nx,ny,nz = {}, {}, {}", nx, ny, nz);
        println!("block sizes = {}, {}, {}", blk_x, blk_y, blk_z);
        println!(
            "Theoretical fetch size (GB): {}",
            g(theoretical_fetch_size as f64 * 1e-9)
        );
        println!(
            "Theoretical write size (GB): {}",
            g(theoretical_write_size as f64 * 1e-9)
        );
    }

    let n = nxu * nyu * nzu;
    let numbytes = n * ELEM_SIZE;

    let d_u = client.empty(numbytes);
    let d_f = client.empty(numbytes);

    let hx = 1.0 / (nx - 1) as Precision;
    let hy = 1.0 / (ny - 1) as Precision;
    let hz = 1.0 / (nz - 1) as Precision;

    test_function(&client, &d_u, n, nx, ny, nz, hx, hy, hz);

    laplacian(
        &client, &d_f, &d_u, n, nx, ny, nz, blk_x, blk_y, blk_z, hx, hy, hz,
    );
    cubecl::future::block_on(client.sync());

    let bytes = client
        .read_one(d_f.clone())
        .expect("failed to read Laplacian output");
    let corr = check_correctness(bytes.as_ref(), nxu, nyu, nzu);
    drop(bytes);

    if csv_output {
        eprintln!("correctness,precision,max_abs_err,l2_rel_err,mean_abs_err,points_checked");
        eprintln!(
            "correctness,{},{},{},{},{}",
            PRECISION_STR,
            g(corr.max_abs_err),
            g(corr.l2_rel_err),
            g(corr.mean_abs_err),
            corr.num_checked
        );
    } else {
        println!("--- Correctness metrics (precision = {}) ---", PRECISION_STR);
        println!("Interior points checked : {}", corr.num_checked);
        println!("Max abs error           : {}", g(corr.max_abs_err));
        println!("Mean abs error          : {}", g(corr.mean_abs_err));
        println!("Relative L2 error       : {}\n", g(corr.l2_rel_err));
    }

    let mut total_elapsed = 0.0_f64;

    for _ in 0..NUM_ITER {
        let elapsed = time_one_iteration(
            &client, &d_f, &d_u, n, nx, ny, nz, blk_x, blk_y, blk_z, hx, hy, hz,
        );
        if csv_output {
            println!(
                "CubeCL-{},{},{},{},{},{},{},{}",
                BACKEND_LABEL,
                gpu_name,
                PRECISION_STR,
                nx,
                blk_x,
                blk_y,
                blk_z,
                g(datasize / elapsed / 1e6)
            );
        }
        total_elapsed += elapsed;
    }

    if !csv_output {
        println!(
            "Laplacian kernel took: {} ms, effective memory bandwidth: {} GB/s \n",
            g(total_elapsed / NUM_ITER as f64),
            g(datasize * NUM_ITER as f64 / total_elapsed / 1e6)
        );
    }
}
