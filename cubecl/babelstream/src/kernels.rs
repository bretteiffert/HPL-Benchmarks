//! BabelStream kernels. All kernels index with `ABSOLUTE_POS` and have no bounds
//! check, so the launch grid must cover the arrays exactly
//! (array size = cube count × block size).
//!
//! Kernels that take a scalar `F` argument also need `F: LaunchArg` (CubeCL
//! 0.11 implements it per float type rather than as a blanket impl).

use cubecl::prelude::*;

/// Specializes two identical multiply-add branches on the FMAD setting; whether the
/// backend contracts them into an FMA is up to its compiler.
#[cube]
pub fn mad_op<F: Float>(a: F, b: F, c: F, #[comptime] use_fma: bool) -> F {
    if use_fma {
        a * b + c
    } else {
        a * b + c
    }
}

#[cube(launch_unchecked)]
pub fn init_kernel<F: Float + LaunchArg>(
    a: &mut [F],
    b: &mut [F],
    c: &mut [F],
    init_a: F,
    init_b: F,
    init_c: F,
) {
    let i = ABSOLUTE_POS;
    a[i] = init_a;
    b[i] = init_b;
    c[i] = init_c;
}

#[cube(launch_unchecked)]
pub fn copy_kernel<F: Float>(a: &[F], c: &mut [F]) {
    let i = ABSOLUTE_POS;
    c[i] = a[i];
}

#[cube(launch_unchecked)]
pub fn mul_kernel<F: Float + LaunchArg>(b: &mut [F], c: &[F], scalar: F) {
    let i = ABSOLUTE_POS;
    b[i] = scalar * c[i];
}

#[cube(launch_unchecked)]
pub fn add_kernel<F: Float>(a: &[F], b: &[F], c: &mut [F]) {
    let i = ABSOLUTE_POS;
    c[i] = a[i] + b[i];
}

#[cube(launch_unchecked)]
pub fn triad_kernel<F: Float + LaunchArg>(
    a: &mut [F],
    b: &[F],
    c: &[F],
    scalar: F,
    #[comptime] use_fma: bool,
) {
    let i = ABSOLUTE_POS;
    a[i] = mad_op::<F>(scalar, c[i], b[i], use_fma);
}

/// Pinned 256-unit grid-stride dot product with a shared-memory tree reduction.
#[cube(launch_unchecked)]
pub fn dot_kernel<F: Float>(
    a: &[F],
    b: &[F],
    partial: &mut [F],
    array_size: usize,
    #[comptime] use_fma: bool,
) {
    let mut acc = F::new(0.0f32);

    // No type annotations on locals: the 0.11 macro rejects them on `let mut`.
    let stride = (CUBE_DIM as usize) * CUBE_COUNT;
    let mut i = ABSOLUTE_POS;
    while i < array_size {
        acc = mad_op::<F>(a[i], b[i], acc, use_fma);
        i += stride;
    }

    let mut shared = Shared::<[F]>::new_slice(256usize);
    shared[UNIT_POS as usize] = acc;
    sync_cube();

    let mut offset = 128u32;
    while offset > 0 {
        if UNIT_POS < offset {
            shared[UNIT_POS as usize] =
                shared[UNIT_POS as usize] + shared[(UNIT_POS + offset) as usize];
        }
        sync_cube();
        offset /= 2;
    }

    if UNIT_POS == 0 {
        partial[CUBE_POS] = shared[0];
    }
}
