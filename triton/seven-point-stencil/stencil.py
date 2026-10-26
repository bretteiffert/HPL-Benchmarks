"""Seven-point stencil buffers, timed launches and the correctness check.

Nothing here prints; main.py fills report.Report. Host setup (spacing,
coefficients) is unchanged from the original laplacian.py.
"""

import numpy as np
import torch
import triton

from kernels import laplacian_kernel, test_function_kernel

TEST_FUNCTION_TB = 1024    # tile of the untimed initialisation kernel (original TBSize)
EXPECTED = 3.0             # Laplacian of u = 0.5 x(x-Lx) + 0.5 y(y-Ly) + 0.5 z(z-Lz)

_PRECISIONS = {
    "float16": (np.float16, torch.float16),
    "float32": (np.float32, torch.float32),
    "float64": (np.float64, torch.float64),
}


class TimerError(Exception):
    pass


def traffic_bytes(nx, ny, nz, itemsize):
    """Theoretical traffic: every point except the 8 corners and the 12 edges is
    read once; every interior point is written once."""
    fetch = (nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)) * itemsize
    write = ((nx - 2) * (ny - 2) * (nz - 2)) * itemsize
    return fetch, write


def launch_grid(nx, ny, nz, bx, by, bz):
    return (triton.cdiv(nx, bx), triton.cdiv(ny, by), triton.cdiv(nz, bz))


def coefficients(nx, ny, nz, npdt):
    """(hx, hy, hz) and (1/hx², 1/hy², 1/hz², -2·Σ1/h²) in the run precision."""
    hx = npdt(1.0 / (nx - 1))
    hy = npdt(1.0 / (ny - 1))
    hz = npdt(1.0 / (nz - 1))
    invhx2 = npdt(1) / hx / hx
    invhy2 = npdt(1) / hy / hy
    invhz2 = npdt(1) / hz / hz
    invhxyz2 = npdt(-2) * (invhx2 + invhy2 + invhz2)
    return (hx, hy, hz), (invhx2, invhy2, invhz2, invhxyz2)


class Stencil:
    def __init__(self, nx, ny, nz, block, precision, device_index, num_warps):
        npdt, tdt = _PRECISIONS[precision]
        self.nx, self.ny, self.nz = nx, ny, nz
        self.bx, self.by, self.bz = block
        self.num_warps = num_warps
        torch.cuda.set_device(device_index)
        self.device = torch.device("cuda", device_index)
        n = nx * ny * nz
        self.d_u = torch.empty(n, dtype=tdt, device=self.device)
        self.d_f = torch.empty(n, dtype=tdt, device=self.device)
        h, p = coefficients(nx, ny, nz, npdt)
        with np.errstate(over="ignore"):
            self.d_h = torch.tensor(list(h), dtype=tdt, device=self.device)
            self.d_p = torch.tensor(list(p), dtype=tdt, device=self.device)
        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.stop_ev = torch.cuda.Event(enable_timing=True)

    @property
    def stream_ptr(self):
        return torch.cuda.current_stream(self.device).cuda_stream

    def initialise(self):
        grid = (triton.cdiv(self.nx, TEST_FUNCTION_TB), self.ny, self.nz)
        test_function_kernel[grid](self.d_u, self.d_h, self.nx, self.ny, self.nz,
                                   TB=TEST_FUNCTION_TB, num_warps=self.num_warps)

    def launch(self):
        grid = launch_grid(self.nx, self.ny, self.nz, self.bx, self.by, self.bz)
        laplacian_kernel[grid](self.d_f, self.d_u, self.d_p, self.nx, self.ny, self.nz,
                               BLK_X=self.bx, BLK_Y=self.by, BLK_Z=self.bz,
                               num_warps=self.num_warps)

    def untimed_run(self):
        """Initialise u, then one untimed Laplacian (also compiles the kernel)."""
        self.initialise()
        self.launch()
        torch.cuda.synchronize()

    def timed_run(self, iteration):
        torch.cuda.synchronize()
        self.start_ev.record()
        self.launch()
        self.stop_ev.record()
        self.stop_ev.synchronize()
        ms = self.start_ev.elapsed_time(self.stop_ev)
        if not ms or ms <= 0.0:
            raise TimerError("device timer returned no measurement for timed iteration %d"
                             % iteration)
        return ms * 1.0e-3


class Correctness:
    def __init__(self):
        self.max_abs_err = 0.0
        self.mean_abs_err = 0.0
        self.l2_rel_err = 0.0
        self.points = 0
        self.notes = []


def check(st, chunk_bytes=64 << 20):
    """Compare the interior of f with the exact value 3 on the device, a few
    z-slabs at a time, so the grid is never copied to the host."""
    nx, ny, nz = st.nx, st.ny, st.nz
    view = st.d_f.reshape(nz, ny, nx)
    slabs = max(1, chunk_bytes // max(1, (nx - 2) * (ny - 2) * 8))
    sum_abs = sum_sq = max_err = 0.0
    count = 0
    nonfinite = 0
    for k0 in range(1, nz - 1, slabs):
        k1 = min(nz - 1, k0 + slabs)
        err = torch.abs(view[k0:k1, 1:ny - 1, 1:nx - 1].to(torch.float64) - EXPECTED)
        sum_abs += float(err.sum().item())
        sum_sq += float((err * err).sum().item())
        max_err = max(max_err, float(err.max().item()))
        nonfinite += int((~torch.isfinite(err)).sum().item())
        count += int(err.numel())
    c = Correctness()
    c.points = count
    if count:
        c.max_abs_err = max_err
        c.mean_abs_err = sum_abs / count
        c.l2_rel_err = (sum_sq / (EXPECTED * EXPECTED * count)) ** 0.5
    if nonfinite:
        c.notes.append("%d interior values are not finite, so the errors above are not "
                       "meaningful" % nonfinite)
    return c
