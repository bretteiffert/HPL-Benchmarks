//! Seven-point Laplacian kernels on an `nx × ny × nz` grid stored x-fastest
//! (`pos = i + nx * (j + ny * k)`).
//!
//! Kernels use the CubeCL 0.10 `Array` container API.

use cubecl::prelude::*;

/// Fills `u` with `0.5·x(x−Lx) + 0.5·y(y−Ly) + 0.5·z(z−Lz)`, whose Laplacian is
/// exactly 3 everywhere, so the second-difference stencil reproduces it exactly
/// up to rounding.
#[cube(launch_unchecked)]
pub fn test_function_kernel<F: Float + ScalarArgSettings>(
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

        let c = F::new(0.5f32);
        let x = F::cast_from(i) * hx;
        let y = F::cast_from(j) * hy;
        let z = F::cast_from(k) * hz;
        let lx = F::cast_from(nx) * hx;
        let ly = F::cast_from(ny) * hy;
        let lz = F::cast_from(nz) * hz;
        u[pos] = c * x * (x - lx) + c * y * (y - ly) + c * z * (z - lz);
    }
}

/// `f = ∇²u` on interior points; boundary points of `f` are left untouched.
#[cube(launch_unchecked)]
pub fn laplacian_kernel<F: Float + ScalarArgSettings>(
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
