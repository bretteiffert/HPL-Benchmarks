//! Precision trait, grid geometry, device buffers and timed launches.

use crate::kernels::*;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// `LaunchArg<RuntimeArg = Self>` lets generic code pass an `F` value as a
/// kernel scalar; f16, f32 and f64 all implement it that way.
pub trait Elem: Float + CubeElement + LaunchArg<RuntimeArg = Self> + Copy + 'static {
    const PRECISION: &'static str;
    fn from_f64(v: f64) -> Self;
    fn to_host_f64(self) -> f64;
    /// Largest finite value, for checking that the stencil coefficients fit.
    fn max_finite() -> f64;
    fn elem_size() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl Elem for half::f16 {
    const PRECISION: &'static str = "float16";
    fn from_f64(v: f64) -> Self {
        half::f16::from_f64(v)
    }
    fn to_host_f64(self) -> f64 {
        self.to_f64()
    }
    fn max_finite() -> f64 {
        half::f16::MAX.to_f64()
    }
}

impl Elem for f32 {
    const PRECISION: &'static str = "float32";
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_host_f64(self) -> f64 {
        self as f64
    }
    fn max_finite() -> f64 {
        f32::MAX as f64
    }
}

impl Elem for f64 {
    const PRECISION: &'static str = "float64";
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_host_f64(self) -> f64 {
        self
    }
    fn max_finite() -> f64 {
        f64::MAX
    }
}

/// Grid size and the cube shape used for the Laplacian.
#[derive(Clone, Copy)]
pub struct Geometry {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub block: (u32, u32, u32),
}

impl Geometry {
    pub fn points(&self) -> usize {
        self.nx as usize * self.ny as usize * self.nz as usize
    }

    pub fn interior_points(&self) -> usize {
        (self.nx as usize - 2) * (self.ny as usize - 2) * (self.nz as usize - 2)
    }

    /// Cubes needed to cover the grid with `block`, rounding up per axis.
    pub fn cubes(&self) -> (u32, u32, u32) {
        let (bx, by, bz) = self.block;
        (self.nx.div_ceil(bx), self.ny.div_ceil(by), self.nz.div_ceil(bz))
    }

    /// Bytes the Laplacian must read: every point except the 8 corners and the
    /// 12 edges, which no interior stencil touches.
    pub fn fetch_bytes(&self, elem: usize) -> usize {
        let (nx, ny, nz) = (self.nx as usize, self.ny as usize, self.nz as usize);
        (self.points() - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)) * elem
    }

    /// Bytes the Laplacian writes: the interior points.
    pub fn write_bytes(&self, elem: usize) -> usize {
        self.interior_points() * elem
    }

    /// Grid spacings h = 1 / (n − 1).
    pub fn spacing(&self) -> (f64, f64, f64) {
        (
            1.0 / (self.nx - 1) as f64,
            1.0 / (self.ny - 1) as f64,
            1.0 / (self.nz - 1) as f64,
        )
    }
}

/// Stencil coefficients 1/h² per axis and the centre weight −2·Σ 1/h².
pub fn coefficients(h: (f64, f64, f64)) -> (f64, f64, f64, f64) {
    let (ix, iy, iz) = (1.0 / (h.0 * h.0), 1.0 / (h.1 * h.1), 1.0 / (h.2 * h.2));
    (ix, iy, iz, -2.0 * (ix + iy + iz))
}

/// # Safety
/// `handle` must hold at least `n` elements of the kernel's element type.
unsafe fn arr(handle: Handle, n: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(handle, n) }
}

pub struct Stencil<F: Elem> {
    client: Client,
    geo: Geometry,
    /// Units per cube for the (1D) initialization kernel.
    init_block: u32,
    u: Handle,
    f: Handle,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Elem> Stencil<F> {
    pub fn new(client: Client, geo: Geometry, init_block: u32) -> Self {
        let bytes = geo.points() * F::elem_size();
        let u = client.empty(bytes);
        let f = client.empty(bytes);
        Self {
            client,
            geo,
            init_block,
            u,
            f,
            _marker: std::marker::PhantomData,
        }
    }

    /// Fills `u` with the test function. Not timed.
    pub fn init(&self) {
        let g = self.geo;
        let (hx, hy, hz) = g.spacing();
        let n = g.points();
        unsafe {
            test_function_kernel::launch_unchecked::<F>(
                &self.client,
                CubeCount::Static(g.nx.div_ceil(self.init_block), g.ny, g.nz),
                CubeDim::new_1d(self.init_block),
                arr(self.u.clone(), n),
                g.nx,
                g.ny,
                g.nz,
                F::from_f64(hx),
                F::from_f64(hy),
                F::from_f64(hz),
            );
        }
    }

    fn launch_laplacian(client: &Client, geo: Geometry, f: Handle, u: Handle) {
        let n = geo.points();
        let (cx, cy, cz) = geo.cubes();
        let (bx, by, bz) = geo.block;
        let (ix, iy, iz, ic) = coefficients(geo.spacing());
        unsafe {
            laplacian_kernel::launch_unchecked::<F>(
                client,
                CubeCount::Static(cx, cy, cz),
                CubeDim::new_3d(bx, by, bz),
                arr(f, n),
                arr(u, n),
                geo.nx,
                geo.ny,
                geo.nz,
                F::from_f64(ix),
                F::from_f64(iy),
                F::from_f64(iz),
                F::from_f64(ic),
            );
        }
    }

    /// Runs the Laplacian once without timing (warm-up and correctness run).
    pub fn laplacian(&self) {
        Self::launch_laplacian(&self.client, self.geo, self.f.clone(), self.u.clone());
    }

    /// Runs the Laplacian under `client.profile()` and returns seconds.
    pub fn timed_laplacian(&self) -> f64 {
        let (geo, f, u) = (self.geo, self.f.clone(), self.u.clone());
        let client = self.client.clone();
        let (_, profile) = self
            .client
            .profile(move || Self::launch_laplacian(&client, geo, f, u), "laplacian")
            .expect("cubecl profile() call failed");
        // `None` means the device recorded no measurement for this window; a
        // zero would silently win every min(), so treat it as a hard error.
        cubecl::future::block_on(profile.resolve())
            .unwrap_or_else(|| panic!("laplacian kernel produced no timing measurement"))
            .duration()
            .as_secs_f64()
    }

    /// Reads `f` back and hands the elements to `check` in place, without
    /// copying them into a second host array (the grid can be many GB).
    pub fn with_result<R>(&self, check: impl FnOnce(&[F]) -> R) -> R {
        let bytes = self
            .client
            .read_one(self.f.clone())
            .expect("read_one(f) failed");
        check(F::from_bytes(&bytes))
    }
}
