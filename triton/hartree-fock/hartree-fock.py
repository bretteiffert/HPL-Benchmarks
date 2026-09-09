#!/usr/bin/env python3
# Triton Hartree-Fock two-electron Fock-build benchmark with device-event timing and optional CPU correctness checks.
# Precision and legacy square-root behavior are selected by the in-source PRECISION and LEGACY_SQRTF constants, which default to fp64 and native square root.
# The kernel resolves the available Triton pow and erf functions at import time and uses the active CUDA or ROCm Torch backend without environment-based backend selection.

import sys
import math
import numpy as np
import torch
import triton
import triton.language as tl


def _resolve_math():
    import triton.language as _tl
    cands = []
    try:
        from triton.language.extra import libdevice as _ld
        cands.append(_ld)
    except ImportError:
        pass
    try:
        from triton.language.extra.cuda import libdevice as _ldc
        cands.append(_ldc)
    except ImportError:
        pass
    cands.append(_tl.math)
    cands.append(_tl)

    out = {}
    for name in ("pow", "erf"):
        for mod in cands:
            fn = getattr(mod, name, None)
            if fn is not None:
                out[name] = fn
                break
        else:
            raise ImportError(
                f"could not resolve a Triton device {name}(). Tried: "
                + ", ".join(getattr(m, "__name__", str(m)) for m in cands)
                + ". Add the correct module for your Triton version here."
            )
    return out["pow"], out["erf"]


_pow, _erf = _resolve_math()


PRECISION = 64
if PRECISION == 32:
    REAL = np.float32
    TORCH_REAL = torch.float32
    TL_REAL = tl.float32
    REAL_NAME = "fp32"
elif PRECISION == 64:
    REAL = np.float64
    TORCH_REAL = torch.float64
    TL_REAL = tl.float64
    REAL_NAME = "fp64"
else:
    raise SystemExit("PRECISION must be 32 or 64")

LEGACY_SQRTF = 0
SQRT_MODE = "legacy-sqrtf" if LEGACY_SQRTF else "native"

SQRTPI2_HOST = REAL(math.pow(3.1415926535897931, -0.5) * 2.0)
SQRTPI2_DEV = 1.12837916709551255856
DTOL = REAL(1.0e-12)
RCUT = REAL(1.0e-12)
TOBOHRS = REAL(1.889725987722)

BLK_SIZE = 256

_erf_f64 = np.vectorize(math.erf, otypes=[np.float64])


def _erfv(x, dt):
    return _erf_f64(np.asarray(x, dtype=np.float64)).astype(dt)


REF_SQRTPI2 = 1.12837916709551255856
REF_DTOL = 1.0e-12
REF_RCUT = 1.0e-12


@triton.jit
def hartree_fock_kernel(
    nnnn, ngauss, natoms,
    geom_ptr, xpnt_ptr, coef_ptr, dens_ptr, schwarz_ptr, fock_ptr,
    dtol, rcut, sqrtpi2,
    DT: tl.constexpr, BLOCK: tl.constexpr, LEGACY: tl.constexpr,
):
    pid = tl.program_id(0)
    ijkl = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)

    active = (ijkl >= 1) & (ijkl <= nnnn)

    ij = tl.sqrt((2 * ijkl).to(tl.float32)).to(tl.int64)
    n = (ij * ij + ij) // 2
    while tl.max(((n < ijkl) & active).to(tl.int32)) > 0:
        step = (n < ijkl) & active
        ij = tl.where(step, ij + 1, ij)
        n = (ij * ij + ij) // 2
    kl = ijkl - (ij * ij - ij) // 2

    ij_s = tl.where(active, ij, 0)
    kl_s = tl.where(active, kl, 0)
    s_ij = tl.load(schwarz_ptr + ij_s, mask=active, other=0.0)
    s_kl = tl.load(schwarz_ptr + kl_s, mask=active, other=0.0)
    keep = active & (s_ij * s_kl > dtol)

    i = tl.sqrt((2 * ij).to(tl.float32)).to(tl.int64) - 1
    n = (i * i + i) // 2
    while tl.max(((n < ij) & keep).to(tl.int32)) > 0:
        step = (n < ij) & keep
        i = tl.where(step, i + 1, i)
        n = (i * i + i) // 2
    j = ij - (i * i - i) // 2

    k = tl.sqrt((2 * kl).to(tl.float32)).to(tl.int64)
    n = (k * k + k) // 2
    while tl.max(((n < kl) & keep).to(tl.int32)) > 0:
        step = (n < kl) & keep
        k = tl.where(step, k + 1, k)
        n = (k * k + k) // 2
    l = kl - (k * k - k) // 2

    i = i - 1
    j = j - 1
    k = k - 1
    l = l - 1

    i = tl.where(keep, i, 0)
    j = tl.where(keep, j, 0)
    k = tl.where(keep, k, 0)
    l = tl.where(keep, l, 0)

    natoms64 = natoms.to(tl.int64)
    one = tl.full((), 1.0, DT)

    eri = tl.zeros([BLOCK], dtype=DT)
    for ib in range(0, ngauss):
        xa = tl.load(xpnt_ptr + ib)
        ca = tl.load(coef_ptr + ib)
        for jb in range(0, ngauss):
            xb = tl.load(xpnt_ptr + jb)
            cb = tl.load(coef_ptr + jb)
            aij = one / (xa + xb)
            gxi = tl.load(geom_ptr + i, mask=keep, other=0.0)
            gxj = tl.load(geom_ptr + j, mask=keep, other=0.0)
            gyi = tl.load(geom_ptr + natoms64 + i, mask=keep, other=0.0)
            gyj = tl.load(geom_ptr + natoms64 + j, mask=keep, other=0.0)
            gzi = tl.load(geom_ptr + 2 * natoms64 + i, mask=keep, other=0.0)
            gzj = tl.load(geom_ptr + 2 * natoms64 + j, mask=keep, other=0.0)
            dx = gxi - gxj
            dy = gyi - gyj
            dz = gzi - gzj
            dij = ca * cb * tl.exp(-xa * xb * aij * (dx * dx + dy * dy + dz * dz)) \
                * _pow(aij, tl.full((), 1.5, DT))
            ok_ij = keep & (tl.abs(dij) > dtol)

            xij = aij * (xa * gxi + xb * gxj)
            yij = aij * (xa * gyi + xb * gyj)
            zij = aij * (xa * gzi + xb * gzj)

            for kb in range(0, ngauss):
                xc = tl.load(xpnt_ptr + kb)
                cc = tl.load(coef_ptr + kb)
                for lb in range(0, ngauss):
                    xd = tl.load(xpnt_ptr + lb)
                    cd = tl.load(coef_ptr + lb)
                    akl = one / (xc + xd)
                    gxk = tl.load(geom_ptr + k, mask=keep, other=0.0)
                    gxl = tl.load(geom_ptr + l, mask=keep, other=0.0)
                    gyk = tl.load(geom_ptr + natoms64 + k, mask=keep, other=0.0)
                    gyl = tl.load(geom_ptr + natoms64 + l, mask=keep, other=0.0)
                    gzk = tl.load(geom_ptr + 2 * natoms64 + k, mask=keep, other=0.0)
                    gzl = tl.load(geom_ptr + 2 * natoms64 + l, mask=keep, other=0.0)
                    ex = gxk - gxl
                    ey = gyk - gyl
                    ez = gzk - gzl
                    dkl = dij * cc * cd \
                        * tl.exp(-xc * xd * akl * (ex * ex + ey * ey + ez * ez)) \
                        * _pow(akl, tl.full((), 1.5, DT))
                    ok_kl = ok_ij & (tl.abs(dkl) > dtol)

                    aijkl = (xa + xb) * (xc + xd) / (xa + xb + xc + xd)
                    px = xij - akl * (xc * gxk + xd * gxl)
                    py = yij - akl * (xc * gyk + xd * gyl)
                    pz = zij - akl * (xc * gzk + xd * gzl)
                    tt = aijkl * (px * px + py * py + pz * pz)

                    if LEGACY:
                        s_tt = tl.sqrt(tt.to(tl.float32)).to(DT)
                        s_aijkl = tl.sqrt(aijkl.to(tl.float32)).to(DT)
                    else:
                        s_tt = tl.sqrt(tt)
                        s_aijkl = tl.sqrt(aijkl)

                    f0t = tl.where(tt > rcut,
                                   _pow(tt, tl.full((), -0.5, DT)) * _erf(s_tt),
                                   sqrtpi2)
                    eri += tl.where(ok_kl, dkl * f0t * s_aijkl, tl.zeros([BLOCK], dtype=DT))

    half = tl.full((), 0.5, DT)
    eri = tl.where(i == j, eri * half, eri)
    eri = tl.where(k == l, eri * half, eri)
    eri = tl.where((i == k) & (j == l), eri * half, eri)

    d_ij = tl.load(dens_ptr + i * natoms64 + j, mask=keep, other=0.0)
    d_kl = tl.load(dens_ptr + k * natoms64 + l, mask=keep, other=0.0)
    d_jl = tl.load(dens_ptr + j * natoms64 + l, mask=keep, other=0.0)
    d_jk = tl.load(dens_ptr + j * natoms64 + k, mask=keep, other=0.0)
    d_il = tl.load(dens_ptr + i * natoms64 + l, mask=keep, other=0.0)
    d_ik = tl.load(dens_ptr + i * natoms64 + k, mask=keep, other=0.0)
    four = tl.full((), 4.0, DT)

    tl.atomic_add(fock_ptr + i * natoms64 + j, d_kl * eri * four, mask=keep)
    tl.atomic_add(fock_ptr + k * natoms64 + l, d_ij * eri * four, mask=keep)
    tl.atomic_add(fock_ptr + i * natoms64 + k, -d_jl * eri, mask=keep)
    tl.atomic_add(fock_ptr + i * natoms64 + l, -d_jk * eri, mask=keep)
    tl.atomic_add(fock_ptr + j * natoms64 + k, -d_il * eri, mask=keep)
    tl.atomic_add(fock_ptr + j * natoms64 + l, -d_ik * eri, mask=keep)


def read_file(filename):
    with open(filename) as f:
        tok = f.read().split()
    p = 0
    ngauss = int(tok[p]); p += 1
    natoms = int(tok[p]); p += 1
    xpnt_d = np.empty(ngauss, dtype=np.float64)
    coef_d = np.empty(ngauss, dtype=np.float64)
    for i in range(ngauss):
        xpnt_d[i] = float(tok[p]); p += 1
        coef_d[i] = float(tok[p]); p += 1
    geom_d = np.empty(3 * natoms, dtype=np.float64)
    for i in range(natoms):
        for j in range(3):
            geom_d[j * natoms + i] = float(tok[p]); p += 1
    return (ngauss, natoms,
            xpnt_d.astype(REAL), coef_d.astype(REAL), geom_d.astype(REAL),
            xpnt_d, coef_d, geom_d)


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


def main():
    if len(sys.argv) < 2:
        print("ERROR: Please provide an input file")
        return 1
    csv_output = False
    run_check = True
    force_check = False
    max_check_atoms = 128
    num_iter = 10
    for s in sys.argv[2:]:
        if s == "--csv":
            csv_output = True
        elif s == "--check":
            run_check = True; force_check = True
        elif s == "--no-check":
            run_check = False
        elif s.startswith("--max-check-atoms="):
            max_check_atoms = int(s.split("=", 1)[1])
        elif s.startswith("--iters="):
            num_iter = int(s.split("=", 1)[1])

    (ngauss, natoms, h_xpnt, h_coef, h_geom,
     xpnt_d, coef_d, geom_d) = read_file(sys.argv[1])

    nsq = natoms * natoms
    h_dens = np.full(nsq, REAL(0.1), dtype=REAL)
    dens_d = np.full(nsq, 0.1, dtype=np.float64)
    for i in range(natoms):
        h_dens[i * natoms + i] = REAL(1.0)
        dens_d[i * natoms + i] = 1.0

    h_coef = (h_coef * np.power(REAL(2.0) * h_xpnt, REAL(0.75))).astype(REAL)
    coef_d = coef_d * np.power(2.0 * xpnt_d, 0.75)
    h_geom = (h_geom * TOBOHRS).astype(REAL)
    geom_d = geom_d * 1.889725987722

    h_fock = np.zeros(nsq, dtype=REAL)
    h_schwarz, nn = build_schwarz(ngauss, natoms, h_xpnt, h_coef, h_geom,
                                  REAL, SQRTPI2_HOST, DTOL, RCUT, bool(LEGACY_SQRTF))

    nnnn = (nn * nn + nn) // 2
    n_blks = (nnnn + 1 + BLK_SIZE - 1) // BLK_SIZE

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no GPU device available to torch")
    dev = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(dev)

    d_xpnt = torch.as_tensor(h_xpnt, dtype=TORCH_REAL, device=dev)
    d_coef = torch.as_tensor(h_coef, dtype=TORCH_REAL, device=dev)
    d_geom = torch.as_tensor(h_geom, dtype=TORCH_REAL, device=dev)
    d_dens = torch.as_tensor(h_dens, dtype=TORCH_REAL, device=dev)
    d_fock = torch.as_tensor(h_fock, dtype=TORCH_REAL, device=dev)
    d_schwarz = torch.as_tensor(h_schwarz, dtype=TORCH_REAL, device=dev)

    grid = (n_blks,)

    def launch():
        hartree_fock_kernel[grid](
            nnnn, ngauss, natoms,
            d_geom, d_xpnt, d_coef, d_dens, d_schwarz, d_fock,
            float(DTOL), float(RCUT), float(SQRTPI2_DEV),
            DT=TL_REAL, BLOCK=BLK_SIZE, LEGACY=bool(LEGACY_SQRTF),
        )

    launch()
    torch.cuda.synchronize()

    h_fock = d_fock.cpu().numpy()
    erep = REAL(0.0)
    for i in range(natoms):
        for j in range(natoms):
            erep = REAL(erep + h_fock[i * natoms + j] * h_dens[i * natoms + j])

    if not csv_output:
        print("2e- energy=%.16f" % (float(erep) * 0.5))
        if run_check and (force_check or natoms <= max_check_atoms):
            fock_gpu = h_fock.astype(np.float64)
            fock_ref = fock_build_cpu(ngauss, natoms, geom_d, xpnt_d, coef_d, dens_d)
            e_gpu = energy_from_fock(fock_gpu, dens_d, natoms)
            e_ref = energy_from_fock(fock_ref, dens_d, natoms)
            denom = abs(e_ref) if abs(e_ref) > 0.0 else 1.0
            rel_loss = abs(e_gpu - e_ref) / denom
            e_inpipe = float(erep) * 0.5
            rel_inpipe = abs(e_inpipe - e_ref) / denom
            d = np.abs(fock_gpu - fock_ref)
            maxd = float(d.max()); rmsd = float(np.sqrt((d * d).sum() / nsq))
            fscale = float(np.abs(fock_ref).max()) or 1.0
            F = fock_gpu.reshape(natoms, natoms)
            asym = float(np.abs(F - F.T).max())
            print()
            print("--- precision check (build: %s, sqrt: %s) ---" % (REAL_NAME, SQRT_MODE))
            print("2e- energy (%s, %s reduction)      = %.16f" % (REAL_NAME, REAL_NAME, e_inpipe))
            print("2e- energy (%s Fock, fp64 reduction) = %.16f" % (REAL_NAME, e_gpu))
            print("2e- energy (fp64 CPU reference)       = %.16f" % e_ref)
            print("precision loss, Fock build only       = %.6e" % rel_loss)
            print("precision loss, incl. %s reduction  = %.6e" % (REAL_NAME, rel_inpipe))
            print("max |F - F_ref| (elementwise)         = %.6e  (rel %.6e)" % (maxd, maxd / fscale))
            print("rms |F - F_ref| (elementwise)         = %.6e" % rmsd)
            print("max |F(i,j) - F(j,i)|                 = %.6e" % asym)
            print("2e- energy from symmetrized F=A+A^T   = %.16f   (diagnostic)"
                  % energy_symmetrized(fock_gpu, dens_d, natoms))
            print("---------------------------------------------------------------")
        elif run_check:
            print("(correctness check skipped: natoms=%d > %d; use --check to force)"
                  % (natoms, max_check_atoms))
    else:
        print("backend,GPU,precision,sqrt_mode,natoms,ngauss,exec_time_ms")
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(num_iter):
            torch.cuda.synchronize()
            start.record()
            launch()
            stop.record()
            stop.synchronize()
            elapsed = start.elapsed_time(stop)
            print("Triton,%s,%s,%s,%d,%d,%f"
                  % (gpu_name, REAL_NAME, SQRT_MODE, natoms, ngauss, elapsed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
