"""BabelStream buffers, timed launches and the CPU reference check.

Nothing here prints; main.py collects the numbers into report.Report.
"""

import numpy as np
import torch

from kernels import (add_kernel, copy_kernel, dot_kernel, init_kernel,
                     mul_kernel, triad_kernel)
from report import sci

START_A = 0.1
START_B = 0.2
START_C = 0.0
START_SCALAR = 0.4

class TimerError(Exception):
    pass


class TritonStream:
    """Three device arrays plus the dot partial sums; each launch is timed with
    device events on the current stream."""

    def __init__(self, array_size, block_size, dot_groups, device_index, dtype):
        self.array_size = array_size
        self.block_size = block_size
        self.dot_groups = dot_groups
        self.dtype = dtype
        torch.cuda.set_device(device_index)
        self.device = torch.device("cuda", device_index)

        self.d_a = torch.empty(array_size, dtype=dtype, device=self.device)
        self.d_b = torch.empty(array_size, dtype=dtype, device=self.device)
        self.d_c = torch.empty(array_size, dtype=dtype, device=self.device)
        self.d_sum = torch.empty(dot_groups, dtype=dtype, device=self.device)
        self.sums = torch.empty(dot_groups, dtype=dtype, device="cpu")

        self.d_scalar = torch.tensor([START_SCALAR], dtype=dtype, device=self.device)
        self.d_init = torch.tensor([START_A, START_B, START_C], dtype=dtype, device=self.device)

        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.stop_ev = torch.cuda.Event(enable_timing=True)

        self.warmup()

    @property
    def stream_ptr(self):
        return torch.cuda.current_stream(self.device).cuda_stream

    def _start(self):
        self.start_ev.record()

    def _stop(self, routine):
        self.stop_ev.record()
        self.stop_ev.synchronize()
        ms = self.start_ev.elapsed_time(self.stop_ev)
        if not ms or ms <= 0.0:
            raise TimerError("device timer returned no measurement for %s" % routine)
        return ms * 1.0e-3

    def warmup(self):
        """Untimed launch of every kernel on one block, so JIT compilation stays
        out of the timed runs."""
        n = self.block_size
        w_a = torch.empty(n, dtype=self.dtype, device=self.device)
        w_b = torch.empty(n, dtype=self.dtype, device=self.device)
        w_c = torch.empty(n, dtype=self.dtype, device=self.device)
        w_sum = torch.empty(self.dot_groups, dtype=self.dtype, device=self.device)
        bs = self.block_size

        init_kernel[(1,)](w_a, w_b, w_c, self.d_init, BLOCK_SIZE=bs)
        copy_kernel[(1,)](w_a, w_c, BLOCK_SIZE=bs)
        mul_kernel[(1,)](w_b, w_c, self.d_scalar, BLOCK_SIZE=bs)
        add_kernel[(1,)](w_a, w_b, w_c, BLOCK_SIZE=bs)
        triad_kernel[(1,)](w_a, w_b, w_c, self.d_scalar, BLOCK_SIZE=bs)
        dot_kernel[(self.dot_groups,)](w_a, w_b, w_sum, n, self.dot_groups, BLOCK_SIZE=bs)
        self.sums.copy_(w_sum)
        torch.cuda.synchronize()

    def _grid(self):
        return (self.array_size // self.block_size,)

    # Each timed routine returns its time in seconds.
    def init_arrays(self):
        self._start()
        init_kernel[self._grid()](self.d_a, self.d_b, self.d_c, self.d_init,
                                  BLOCK_SIZE=self.block_size)
        return self._stop("Init")

    def copy(self):
        self._start()
        copy_kernel[self._grid()](self.d_a, self.d_c, BLOCK_SIZE=self.block_size)
        return self._stop("Copy")

    def mul(self):
        self._start()
        mul_kernel[self._grid()](self.d_b, self.d_c, self.d_scalar,
                                 BLOCK_SIZE=self.block_size)
        return self._stop("Mul")

    def add(self):
        self._start()
        add_kernel[self._grid()](self.d_a, self.d_b, self.d_c,
                                 BLOCK_SIZE=self.block_size)
        return self._stop("Add")

    def triad(self):
        self._start()
        triad_kernel[self._grid()](self.d_a, self.d_b, self.d_c, self.d_scalar,
                                   BLOCK_SIZE=self.block_size)
        return self._stop("Triad")

    def dot(self):
        """Returns (seconds, dot). The timed window includes copying the
        partial sums to the host, as in the original."""
        self._start()
        dot_kernel[(self.dot_groups,)](self.d_a, self.d_b, self.d_sum,
                                       self.array_size, self.dot_groups,
                                       BLOCK_SIZE=self.block_size)
        self.sums.copy_(self.d_sum)
        seconds = self._stop("Dot")

        partials = self.sums.numpy()
        total = partials.dtype.type(0.0)
        # Overflow to inf is reported as a correctness note, not a warning.
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(self.dot_groups):
                total = total + partials[i]
        return seconds, total

    def read_arrays(self):
        return (self.d_a.cpu().numpy(), self.d_b.cpu().numpy(), self.d_c.cpu().numpy())


# -- CPU reference -------------------------------------------------------------

class Correctness:
    def __init__(self):
        self.rows = []        # (array, expected, checksum_rel_err, max_elem_rel_err)
        self.dot_row = None   # ("dot", expected, rel_err, None)
        self.actual_dot = 0.0
        self.notes = []


def _kahan_sum(v):
    s = 0.0
    comp = 0.0
    for x in v:
        y = float(x) - comp
        t = s + y
        comp = (t - s) - y
        s = t
    return s


def check_solution(ntimes, a, b, c, dot_sum, array_size):
    """Compare the arrays and dot against gold values computed in the array's
    precision after ``ntimes`` iterations (warm-up included: it updates the arrays)."""
    T = a.dtype.type
    finfo = np.finfo(T)
    goldA = T(START_A)
    goldB = T(START_B)
    goldC = T(START_C)
    scalar = T(START_SCALAR)

    for _ in range(ntimes):
        goldC = goldA
        goldB = scalar * goldC
        goldC = goldA + goldB
        goldA = goldB + scalar * goldC

    def metrics(v, gold):
        gold_d = float(gold)
        actual_sum = _kahan_sum(v)
        expected_sum = gold_d * len(v)
        checksum_rel_err = (abs(actual_sum - expected_sum) / abs(expected_sum)
                            if expected_sum != 0.0 else abs(actual_sum))
        vd = v.astype(np.float64)
        abs_err = np.abs(vd - gold_d)
        rel = abs_err / abs(gold_d) if gold_d != 0.0 else abs_err
        return checksum_rel_err, float(rel.max())

    out = Correctness()
    for name, v, gold in (("a", a, goldA), ("b", b, goldB), ("c", c, goldC)):
        chk, mx = metrics(v, gold)
        out.rows.append((name, float(gold), chk, mx))

    expected_dot = float(goldA) * float(goldB) * float(array_size)
    actual_dot = float(dot_sum)
    dot_err = (abs(actual_dot - expected_dot) / abs(expected_dot)
               if expected_dot != 0.0 else abs(actual_dot))
    out.dot_row = ("dot", expected_dot, dot_err, None)
    out.actual_dot = actual_dot

    tiny = float(finfo.tiny)
    small = [name for name, gold, _, _ in out.rows if abs(gold) < tiny]
    if small:
        out.notes.append(
            "expected %s below the smallest normal %s (%s) after %s iterations, "
            "so their relative errors are not meaningful"
            % (", ".join(small), _precname(T), sci(tiny, 3), "{:,}".format(ntimes)))
    product = abs(float(goldA) * float(goldB))
    if product < tiny:
        out.notes.append(
            "expected element product %s underflows %s at this iteration count, "
            "so the dot relative error is not meaningful" % (sci(product, 3), _precname(T)))
    if abs(expected_dot) > float(finfo.max):
        out.notes.append(
            "expected dot %s exceeds the %s maximum %s, so the dot relative error "
            "is not meaningful" % (sci(expected_dot, 3), _precname(T),
                                   "{:,.0f}".format(float(finfo.max))))
    return out


def _precname(T):
    return {2: "float16", 4: "float32", 8: "float64"}[np.dtype(T).itemsize]
