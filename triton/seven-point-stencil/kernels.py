# Seven-point stencil device code (Triton). Both kernels are unchanged from
# the original laplacian.py; the tile shape (BLK_X, BLK_Y, BLK_Z) is constexpr.

import triton
import triton.language as tl


@triton.jit
def test_function_kernel(u_ptr, h_ptr, nx, ny, nz, TB: tl.constexpr):
    i = tl.program_id(0) * TB + tl.arange(0, TB)
    j = tl.program_id(1)
    k = tl.program_id(2)

    mask = i < nx

    hx = tl.load(h_ptr + 0)
    hy = tl.load(h_ptr + 1)
    hz = tl.load(h_ptr + 2)

    pos = (i + nx * (j + ny * k)).to(tl.int64)

    c = 0.5
    x = i.to(hx.dtype) * hx
    y = j.to(hy.dtype) * hy
    z = k.to(hz.dtype) * hz
    Lx = nx.to(hx.dtype) * hx
    Ly = ny.to(hy.dtype) * hy
    Lz = nz.to(hz.dtype) * hz

    val = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz)
    tl.store(u_ptr + pos, val, mask=mask)


@triton.jit
def laplacian_kernel(f_ptr, u_ptr, p_ptr, nx, ny, nz,
                     BLK_X: tl.constexpr, BLK_Y: tl.constexpr, BLK_Z: tl.constexpr):
    i = tl.program_id(0) * BLK_X + tl.arange(0, BLK_X)[:, None, None]
    j = tl.program_id(1) * BLK_Y + tl.arange(0, BLK_Y)[None, :, None]
    k = tl.program_id(2) * BLK_Z + tl.arange(0, BLK_Z)[None, None, :]

    interior = ((i > 0) & (i < nx - 1) &
                (j > 0) & (j < ny - 1) &
                (k > 0) & (k < nz - 1))

    invhx2 = tl.load(p_ptr + 0)
    invhy2 = tl.load(p_ptr + 1)
    invhz2 = tl.load(p_ptr + 2)
    invhxyz2 = tl.load(p_ptr + 3)

    slc = nx * ny
    pos = (i + nx * j + slc * k).to(tl.int64)

    cc = tl.load(u_ptr + pos, mask=interior, other=0.0)
    xm = tl.load(u_ptr + pos - 1, mask=interior, other=0.0)
    xp = tl.load(u_ptr + pos + 1, mask=interior, other=0.0)
    ym = tl.load(u_ptr + pos - nx, mask=interior, other=0.0)
    yp = tl.load(u_ptr + pos + nx, mask=interior, other=0.0)
    zm = tl.load(u_ptr + pos - slc, mask=interior, other=0.0)
    zp = tl.load(u_ptr + pos + slc, mask=interior, other=0.0)

    val = (cc * invhxyz2
           + (xm + xp) * invhx2
           + (ym + yp) * invhy2
           + (zm + zp) * invhz2)

    tl.store(f_ptr + pos, val, mask=interior)
