//! Device buffers and timed kernel launches.

use crate::kernels::*;
use crate::backend::{BackendRuntime, Client};
use cubecl::prelude::*;
use cubecl::server::Handle;

pub const START_A: f64 = 0.1;
pub const START_B: f64 = 0.2;
pub const START_C: f64 = 0.0;
pub const START_SCALAR: f64 = 0.4;

/// Device element type used by the benchmark kernels and scalar launch args.
pub trait Elem: Float + CubeElement + ScalarArgSettings + Copy + PartialOrd + 'static {
    const PRECISION: &'static str;
    fn from_f64(v: f64) -> Self;
    fn to_host_f64(self) -> f64;
    fn h_add(self, o: Self) -> Self;
    fn h_mul(self, o: Self) -> Self;
    fn tiny() -> f64;
    /// Largest finite value; the dot result overflows above this.
    fn max_finite() -> f64;
    fn elem_size() -> usize {
        std::mem::size_of::<Self>()
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
    fn h_add(self, o: Self) -> Self {
        self + o
    }
    fn h_mul(self, o: Self) -> Self {
        self * o
    }
    fn tiny() -> f64 {
        f32::MIN_POSITIVE as f64
    }
    fn max_finite() -> f64 {
        f32::MAX as f64
    }
}

/// Host-side gold values use f16 arithmetic (each op rounds to f16), matching
/// how the other precisions compute their expected results.
impl Elem for half::f16 {
    const PRECISION: &'static str = "float16";
    fn from_f64(v: f64) -> Self {
        half::f16::from_f64(v)
    }
    fn to_host_f64(self) -> f64 {
        self.to_f64()
    }
    fn h_add(self, o: Self) -> Self {
        self + o
    }
    fn h_mul(self, o: Self) -> Self {
        self * o
    }
    fn tiny() -> f64 {
        half::f16::MIN_POSITIVE.to_f64()
    }
    fn max_finite() -> f64 {
        half::f16::MAX.to_f64()
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
    fn h_add(self, o: Self) -> Self {
        self + o
    }
    fn h_mul(self, o: Self) -> Self {
        self * o
    }
    fn tiny() -> f64 {
        f64::MIN_POSITIVE
    }
    fn max_finite() -> f64 {
        f64::MAX
    }
}

/// One set of BabelStream arrays plus the grid that covers them exactly.
struct Buffers {
    a: Handle,
    b: Handle,
    c: Handle,
    sum: Handle,
    n: usize,
    grid: CubeCount,
}

/// # Safety
/// `handle` must hold at least `n` elements of the kernel's element type.
unsafe fn arr(handle: Handle, n: usize) -> ArrayArg<BackendRuntime> {
    unsafe { ArrayArg::from_raw_parts(handle, n) }
}

pub struct CubeStream<F: Elem> {
    client: Client,
    block_size: u32,
    dot_groups: u32,
    use_fma: bool,
    main: Buffers,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Elem> CubeStream<F> {
    /// `array_size` must be a multiple of `block_size`; the caller validates this.
    pub fn new(
        client: Client,
        array_size: u32,
        block_size: u32,
        dot_groups: u32,
        use_fma: bool,
    ) -> Self {
        let main = Self::alloc(&client, array_size as usize, block_size, dot_groups);
        let s = Self {
            client,
            block_size,
            dot_groups,
            use_fma,
            main,
            _marker: std::marker::PhantomData,
        };
        s.warmup();
        s
    }

    fn alloc(client: &Client, n: usize, block_size: u32, dot_groups: u32) -> Buffers {
        let bytes = n * F::elem_size();
        Buffers {
            a: client.empty(bytes),
            b: client.empty(bytes),
            c: client.empty(bytes),
            sum: client.empty(dot_groups as usize * F::elem_size()),
            n,
            grid: CubeCount::Static(n as u32 / block_size, 1, 1),
        }
    }

    fn cube_dim(&self) -> CubeDim {
        CubeDim::new_1d(self.block_size)
    }

    /// Compiles every kernel by running each once on a single-cube problem.
    fn warmup(&self) {
        let w = Self::alloc(&self.client, self.block_size as usize, self.block_size, self.dot_groups);
        self.launch_init(&w);
        self.launch_copy(&w);
        self.launch_mul(&w);
        self.launch_add(&w);
        self.launch_triad(&w);
        self.launch_dot(&w);
    }

    /// Runs `f` under `client.profile()` and returns (seconds, output).
    fn timed<O: Send + 'static>(
        &self,
        name: &'static str,
        f: impl FnOnce(&Client) -> O + Send,
    ) -> (f64, O) {
        let client = self.client.clone();
        let (out, profile) = self
            .client
            .profile(move || f(&client), name)
            .expect("cubecl profile() call failed");
        // `None` means the device recorded no measurement for this window; a
        // zero would silently win every min(), so treat it as a hard error.
        let secs = cubecl::future::block_on(profile.resolve())
            .duration()
            .as_secs_f64();
        (secs, out)
    }

    fn launch_init(&self, buf: &Buffers) -> f64 {
        let (a, b, c, n) = (buf.a.clone(), buf.b.clone(), buf.c.clone(), buf.n);
        let (grid, dim) = (buf.grid.clone(), self.cube_dim());
        self.timed("init", move |client| unsafe {
            init_kernel::launch_unchecked::<F, BackendRuntime>(
                client,
                grid,
                dim,
                arr(a, n),
                arr(b, n),
                arr(c, n),
                F::from_f64(START_A),
                F::from_f64(START_B),
                F::from_f64(START_C),
            );
        })
        .0
    }

    fn launch_copy(&self, buf: &Buffers) -> f64 {
        let (a, c, n) = (buf.a.clone(), buf.c.clone(), buf.n);
        let (grid, dim) = (buf.grid.clone(), self.cube_dim());
        self.timed("copy", move |client| unsafe {
            copy_kernel::launch_unchecked::<F, BackendRuntime>(client, grid, dim, arr(a, n), arr(c, n));
        })
        .0
    }

    fn launch_mul(&self, buf: &Buffers) -> f64 {
        let (b, c, n) = (buf.b.clone(), buf.c.clone(), buf.n);
        let (grid, dim) = (buf.grid.clone(), self.cube_dim());
        self.timed("mul", move |client| unsafe {
            mul_kernel::launch_unchecked::<F, BackendRuntime>(
                client,
                grid,
                dim,
                arr(b, n),
                arr(c, n),
                F::from_f64(START_SCALAR),
            );
        })
        .0
    }

    fn launch_add(&self, buf: &Buffers) -> f64 {
        let (a, b, c, n) = (buf.a.clone(), buf.b.clone(), buf.c.clone(), buf.n);
        let (grid, dim) = (buf.grid.clone(), self.cube_dim());
        self.timed("add", move |client| unsafe {
            add_kernel::launch_unchecked::<F, BackendRuntime>(client, grid, dim, arr(a, n), arr(b, n), arr(c, n));
        })
        .0
    }

    fn launch_triad(&self, buf: &Buffers) -> f64 {
        let (a, b, c, n) = (buf.a.clone(), buf.b.clone(), buf.c.clone(), buf.n);
        let (grid, dim, use_fma) = (buf.grid.clone(), self.cube_dim(), self.use_fma);
        self.timed("triad", move |client| unsafe {
            triad_kernel::launch_unchecked::<F, BackendRuntime>(
                client,
                grid,
                dim,
                arr(a, n),
                arr(b, n),
                arr(c, n),
                F::from_f64(START_SCALAR),
                use_fma,
            );
        })
        .0
    }

    /// Times the dot kernel plus the read-back of the per-group partial sums,
    /// then finishes the reduction on the host. Returns (seconds, dot product).
    fn launch_dot(&self, buf: &Buffers) -> (f64, f64) {
        let (a, b, sum, n) = (buf.a.clone(), buf.b.clone(), buf.sum.clone(), buf.n);
        let (dim, use_fma) = (self.cube_dim(), self.use_fma);
        let groups = self.dot_groups;
        let (t, bytes) = self.timed("dot", move |client| {
            unsafe {
                dot_kernel::launch_unchecked::<F, BackendRuntime>(
                    client,
                    CubeCount::Static(groups, 1, 1),
                    dim,
                    arr(a, n),
                    arr(b, n),
                    arr(sum.clone(), groups as usize),
                    n,
                    use_fma,
                );
            }
            client.read_one(sum).expect("dot read_one failed")
        });
        let partials = F::from_bytes(&bytes);
        let total = partials
            .iter()
            .take(groups as usize)
            .map(|&p| p.to_host_f64())
            .sum::<f64>();
        (t, total)
    }

    pub fn init_arrays(&self) -> f64 {
        self.launch_init(&self.main)
    }
    pub fn copy(&self) -> f64 {
        self.launch_copy(&self.main)
    }
    pub fn mul(&self) -> f64 {
        self.launch_mul(&self.main)
    }
    pub fn add(&self) -> f64 {
        self.launch_add(&self.main)
    }
    pub fn triad(&self) -> f64 {
        self.launch_triad(&self.main)
    }
    pub fn dot(&self) -> (f64, f64) {
        self.launch_dot(&self.main)
    }

    pub fn read_arrays(&self) -> (Vec<F>, Vec<F>, Vec<F>) {
        let read = |h: &Handle, what: &str| {
            let bytes = self
                .client
                .read_one(h.clone())
                .unwrap_or_else(|e| panic!("read_one({}) failed: {:?}", what, e));
            F::from_bytes(&bytes).to_vec()
        };
        (
            read(&self.main.a, "a"),
            read(&self.main.b, "b"),
            read(&self.main.c, "c"),
        )
    }
}
