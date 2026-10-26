# Hartree-Fock device code (Triton). The kernel is unchanged from the original
# hartree-fock.py; pow and erf are resolved from the installed Triton at import
# time (triton.language.extra.libdevice in Triton 3.7.1).

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
