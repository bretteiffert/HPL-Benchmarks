"""Hartree-Fock problem: input parsing, buffers, timed launches, CPU reference.

Nothing here prints; main.py collects the numbers into report.Report. The host
math (input scaling, density, Schwarz screening, CPU reference) is unchanged
from the original hartree-fock.py; only the precision is now chosen at run time.
"""

import math
import os

import numpy as np
import torch
import triton.language as tl

from kernels import hartree_fock_kernel

SQRTPI2_DEV = 1.12837916709551255856
REF_SQRTPI2 = 1.12837916709551255856
REF_DTOL = 1.0e-12
REF_RCUT = 1.0e-12

_PRECISIONS = {
    "float32": (np.float32, torch.float32, tl.float32),
    "float64": (np.float64, torch.float64, tl.float64),
}


class InputError(Exception):
    pass


class TimerError(Exception):
    pass


# -- input ---------------------------------------------------------------------

def read_input(path):
    """Parse ngauss, natoms, ngauss (exponent, coefficient) pairs, then natoms
    x/y/z coordinates. Returns float64 arrays; geometry is stored x[], y[], z[]."""
    try:
        with open(path) as f:
            tok = f.read().split()
    except OSError as e:
        raise InputError("cannot read input file '%s': %s" % (path, e.strerror or e))

    pos = [0]

    def take(what, kind):
        if pos[0] >= len(tok):
            raise InputError("input file '%s' ends early: expected %s (value %d)"
                             % (path, what, pos[0] + 1))
        t = tok[pos[0]]
        pos[0] += 1
        try:
            return kind(t)
        except ValueError:
            raise InputError("input file '%s': expected %s for %s, got '%s'"
                             % (path, "an integer" if kind is int else "a number", what, t))

    ngauss = take("ngauss", int)
    natoms = take("natoms", int)
    if ngauss < 1:
        raise InputError("input file '%s': ngauss must be at least 1, got %d" % (path, ngauss))
    if natoms < 1:
        raise InputError("input file '%s': natoms must be at least 1, got %d" % (path, natoms))
    xpnt = np.empty(ngauss, dtype=np.float64)
    coef = np.empty(ngauss, dtype=np.float64)
    for i in range(ngauss):
        xpnt[i] = take("exponent %d" % (i + 1), float)
        coef[i] = take("coefficient %d" % (i + 1), float)
    geom = np.empty(3 * natoms, dtype=np.float64)
    for i in range(natoms):
        for j in range(3):
            geom[j * natoms + i] = take("coordinate %s of atom %d" % ("xyz"[j], i + 1), float)
    return ngauss, natoms, xpnt, coef, geom


# -- host math (unchanged) ------------------------------------------------------

_erf_f64 = np.vectorize(math.erf, otypes=[np.float64])


def _erfv(x, dt):
    return _erf_f64(np.asarray(x, dtype=np.float64)).astype(dt)


def _sq(x):
    return x * x


def ssss_host(i, j, k, l, ngauss, xpnt, coef, geom, natoms, dt, sqrtpi2, dtol, rcut,
              legacy):
    one = dt(1.0)
    eri = np.zeros(np.shape(i), dtype=dt)
    gx_i, gx_j = geom[i], geom[j]
    gy_i, gy_j = geom[natoms + i], geom[natoms + j]
    gz_i, gz_j = geom[2 * natoms + i], geom[2 * natoms + j]
    gx_k, gx_l = geom[k], geom[l]
    gy_k, gy_l = geom[natoms + k], geom[natoms + l]
    gz_k, gz_l = geom[2 * natoms + k], geom[2 * natoms + l]
    for ib in range(ngauss):
        for jb in range(ngauss):
            aij = one / (xpnt[ib] + xpnt[jb])
            dij = coef[ib] * coef[jb] * np.exp(
                -xpnt[ib] * xpnt[jb] * aij *
                (_sq(gx_i - gx_j) + _sq(gy_i - gy_j) + _sq(gz_i - gz_j))
            ) * np.power(aij, dt(1.5))
            ok_ij = np.abs(dij) > dtol
            xij = aij * (xpnt[ib] * gx_i + xpnt[jb] * gx_j)
            yij = aij * (xpnt[ib] * gy_i + xpnt[jb] * gy_j)
            zij = aij * (xpnt[ib] * gz_i + xpnt[jb] * gz_j)
            for kb in range(ngauss):
                for lb in range(ngauss):
                    akl = one / (xpnt[kb] + xpnt[lb])
                    dkl = dij * coef[kb] * coef[lb] * np.exp(
                        -xpnt[kb] * xpnt[lb] * akl *
                        (_sq(gx_k - gx_l) + _sq(gy_k - gy_l) + _sq(gz_k - gz_l))
                    ) * np.power(akl, dt(1.5))
                    ok = ok_ij & (np.abs(dkl) > dtol)
                    aijkl = ((xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb]) /
                             (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb]))
                    tt = aijkl * (
                        _sq(xij - akl * (xpnt[kb] * gx_k + xpnt[lb] * gx_l)) +
                        _sq(yij - akl * (xpnt[kb] * gy_k + xpnt[lb] * gy_l)) +
                        _sq(zij - akl * (xpnt[kb] * gz_k + xpnt[lb] * gz_l)))
                    if legacy:
                        s_tt = np.sqrt(tt.astype(np.float32)).astype(dt)
                        s_aijkl = dt(np.sqrt(np.float32(aijkl)))
                    else:
                        s_tt = np.sqrt(tt)
                        s_aijkl = np.sqrt(aijkl)
                    tt_s = np.where(tt > rcut, tt, dt(1.0)).astype(dt)
                    f0t = np.where(tt > rcut,
                                   (np.power(tt_s, dt(-0.5)) * _erfv(s_tt, dt)).astype(dt),
                                   dt(sqrtpi2)).astype(dt)
                    eri = eri + np.where(ok, dkl * f0t * s_aijkl, dt(0.0)).astype(dt)
    return eri


def build_schwarz(ngauss, natoms, xpnt, coef, geom, dt, sqrtpi2, dtol, rcut, legacy):
    nn = (natoms * natoms + natoms) // 2
    schwarz = np.zeros(nn + 1, dtype=dt)
    ii, jj = np.tril_indices(natoms)
    vals = ssss_host(ii, jj, ii, jj, ngauss, xpnt, coef, geom, natoms,
                     dt, sqrtpi2, dtol, rcut, legacy)
    schwarz[1:] = np.sqrt(np.abs(vals)).astype(dt)
    return schwarz, nn


def fock_build_cpu(ngauss, natoms, geom, xpnt, coef, dens):
    dt = np.float64
    fock = np.zeros(natoms * natoms, dtype=dt)
    nn = (natoms * natoms + natoms) // 2
    schwarz = np.zeros(nn + 1, dtype=dt)
    ii, jj = np.tril_indices(natoms)
    schwarz[1:] = np.sqrt(np.abs(ssss_host(ii, jj, ii, jj, ngauss, xpnt, coef, geom,
                                           natoms, dt, REF_SQRTPI2, REF_DTOL,
                                           REF_RCUT, False)))

    npairs = len(ii)
    CHUNK = max(1, 1 << 22)
    p = 0
    while p < npairs:
        q_end = p
        total = 0
        while q_end < npairs and (total + q_end + 1) <= CHUNK:
            total += q_end + 1
            q_end += 1
        if q_end == p:
            q_end = p + 1
            total = p + 1
        counts = np.arange(p, q_end) + 1
        I = np.repeat(ii[p:q_end], counts)
        J = np.repeat(jj[p:q_end], counts)
        flat = np.concatenate([np.arange(c) for c in counts])
        K = ii[flat]
        L = jj[flat]
        p = q_end

        IJ = (I * (I + 1)) // 2 + J + 1
        KL = (K * (K + 1)) // 2 + L + 1
        sel = schwarz[IJ] * schwarz[KL] > REF_DTOL
        I, J, K, L = I[sel], J[sel], K[sel], L[sel]
        if I.size == 0:
            continue

        eri = ssss_host(I, J, K, L, ngauss, xpnt, coef, geom, natoms,
                        dt, REF_SQRTPI2, REF_DTOL, REF_RCUT, False)
        eri = np.where(I == J, eri * 0.5, eri)
        eri = np.where(K == L, eri * 0.5, eri)
        eri = np.where((I == K) & (J == L), eri * 0.5, eri)

        np.add.at(fock, I * natoms + J, dens[K * natoms + L] * eri * 4.0)
        np.add.at(fock, K * natoms + L, dens[I * natoms + J] * eri * 4.0)
        np.add.at(fock, I * natoms + K, -dens[J * natoms + L] * eri)
        np.add.at(fock, I * natoms + L, -dens[J * natoms + K] * eri)
        np.add.at(fock, J * natoms + K, -dens[I * natoms + L] * eri)
        np.add.at(fock, J * natoms + L, -dens[I * natoms + K] * eri)
    return fock


def energy_from_fock(fock, dens, natoms):
    e = 0.0
    for t in range(natoms * natoms):
        e += float(fock[t]) * float(dens[t])
    return e * 0.5


def energy_symmetrized(fock, dens, natoms):
    F = fock.reshape(natoms, natoms)
    D = dens.reshape(natoms, natoms)
    S = F + F.T
    np.fill_diagonal(S, np.diag(F))
    return float((S * D).sum()) * 0.5


# -- problem setup ---------------------------------------------------------------

class Problem:
    """Host arrays in the build precision plus float64 copies for the reference."""

    def __init__(self, path, precision, legacy_sqrt):
        REAL = _PRECISIONS[precision][0]
        self.path = path
        self.input_name = os.path.basename(os.path.normpath(path))
        self.precision = precision
        self.legacy = bool(legacy_sqrt)
        self.real = REAL

        ngauss, natoms, xpnt_d, coef_d, geom_d = read_input(path)
        self.ngauss, self.natoms = ngauss, natoms

        h_xpnt = xpnt_d.astype(REAL)
        h_coef = coef_d.astype(REAL)
        h_geom = geom_d.astype(REAL)

        nsq = natoms * natoms
        h_dens = np.full(nsq, REAL(0.1), dtype=REAL)
        dens_d = np.full(nsq, 0.1, dtype=np.float64)
        for i in range(natoms):
            h_dens[i * natoms + i] = REAL(1.0)
            dens_d[i * natoms + i] = 1.0

        h_coef = (h_coef * np.power(REAL(2.0) * h_xpnt, REAL(0.75))).astype(REAL)
        coef_d = coef_d * np.power(2.0 * xpnt_d, 0.75)
        h_geom = (h_geom * REAL(1.889725987722)).astype(REAL)
        geom_d = geom_d * 1.889725987722

        self.sqrtpi2_host = REAL(math.pow(3.1415926535897931, -0.5) * 2.0)
        self.dtol = REAL(1.0e-12)
        self.rcut = REAL(1.0e-12)

        self.h_xpnt, self.h_coef, self.h_geom, self.h_dens = h_xpnt, h_coef, h_geom, h_dens
        self.xpnt_d, self.coef_d, self.geom_d, self.dens_d = xpnt_d, coef_d, geom_d, dens_d
        self.h_schwarz, self.nn = build_schwarz(ngauss, natoms, h_xpnt, h_coef, h_geom,
                                                REAL, self.sqrtpi2_host, self.dtol,
                                                self.rcut, self.legacy)
        self.nnnn = (self.nn * self.nn + self.nn) // 2

    def grid(self, block_size):
        return (self.nnnn + 1 + block_size - 1) // block_size


# -- device run ------------------------------------------------------------------

class FockBuild:
    def __init__(self, problem, block_size, device_index, num_warps):
        p = problem
        self.p = p
        self.block_size = block_size
        self.num_warps = num_warps
        _, torch_real, tl_real = _PRECISIONS[p.precision]
        self.tl_real = tl_real
        torch.cuda.set_device(device_index)
        dev = torch.device("cuda", device_index)
        self.device = dev
        self.d_xpnt = torch.as_tensor(p.h_xpnt, dtype=torch_real, device=dev)
        self.d_coef = torch.as_tensor(p.h_coef, dtype=torch_real, device=dev)
        self.d_geom = torch.as_tensor(p.h_geom, dtype=torch_real, device=dev)
        self.d_dens = torch.as_tensor(p.h_dens, dtype=torch_real, device=dev)
        self.d_fock = torch.as_tensor(np.zeros(p.natoms * p.natoms, dtype=p.real),
                                      dtype=torch_real, device=dev)
        self.d_schwarz = torch.as_tensor(p.h_schwarz, dtype=torch_real, device=dev)
        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.stop_ev = torch.cuda.Event(enable_timing=True)

    @property
    def stream_ptr(self):
        return torch.cuda.current_stream(self.device).cuda_stream

    def launch(self):
        p = self.p
        hartree_fock_kernel[(p.grid(self.block_size),)](
            p.nnnn, p.ngauss, p.natoms,
            self.d_geom, self.d_xpnt, self.d_coef, self.d_dens, self.d_schwarz, self.d_fock,
            float(p.dtol), float(p.rcut), float(SQRTPI2_DEV),
            DT=self.tl_real, BLOCK=self.block_size, LEGACY=p.legacy,
            num_warps=self.num_warps,
        )

    def untimed_run(self):
        """First launch (also compiles the kernel). Returns its Fock matrix,
        the only one that is checked: later launches keep accumulating."""
        self.launch()
        torch.cuda.synchronize()
        return self.d_fock.cpu().numpy().copy()

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


# -- correctness -------------------------------------------------------------------

class Correctness:
    def __init__(self):
        self.e2_energy = 0.0          # build precision, build-precision reduction
        self.checked = False
        self.e_gpu_fp64 = None        # build-precision Fock, fp64 reduction
        self.e_ref = None             # fp64 CPU reference
        self.rel_err_fock = None      # Fock build only
        self.rel_err_ref = None       # including the build-precision reduction
        self.max_abs = None
        self.max_rel = None
        self.rms = None
        self.asym = None
        self.e_symmetrized = None
        self.notes = []


def in_pipe_energy(problem, h_fock):
    """2e- energy reduced in the build precision, as the original printed it."""
    REAL = problem.real
    n = problem.natoms
    erep = REAL(0.0)
    for i in range(n):
        for j in range(n):
            erep = REAL(erep + h_fock[i * n + j] * problem.h_dens[i * n + j])
    return float(erep) * 0.5


def check(problem, h_fock, run_check):
    p = problem
    c = Correctness()
    c.e2_energy = in_pipe_energy(p, h_fock)
    if not run_check:
        return c
    natoms, nsq = p.natoms, p.natoms * p.natoms
    fock_gpu = h_fock.astype(np.float64)
    fock_ref = fock_build_cpu(p.ngauss, natoms, p.geom_d, p.xpnt_d, p.coef_d, p.dens_d)
    e_gpu = energy_from_fock(fock_gpu, p.dens_d, natoms)
    e_ref = energy_from_fock(fock_ref, p.dens_d, natoms)
    denom = abs(e_ref) if abs(e_ref) > 0.0 else 1.0
    d = np.abs(fock_gpu - fock_ref)
    fscale = float(np.abs(fock_ref).max()) or 1.0
    F = fock_gpu.reshape(natoms, natoms)

    c.checked = True
    c.e_gpu_fp64 = e_gpu
    c.e_ref = e_ref
    c.rel_err_fock = abs(e_gpu - e_ref) / denom
    c.rel_err_ref = abs(c.e2_energy - e_ref) / denom
    c.max_abs = float(d.max())
    c.max_rel = c.max_abs / fscale
    c.rms = float(np.sqrt((d * d).sum() / nsq))
    c.asym = float(np.abs(F - F.T).max())
    c.e_symmetrized = energy_symmetrized(fock_gpu, p.dens_d, natoms)
    return c
