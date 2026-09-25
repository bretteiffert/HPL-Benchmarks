# Mojo 1.0.0 / MAX 26.5 port of the Hartree-Fock two-electron Fock-matrix benchmark.
#
# Preserves the original benchmark behavior:
#   * source-selectable fp32/fp64 arithmetic
#   * Schwarz screening and GPU atomic Fock accumulation
#   * optional fp64 CPU reference validation
#   * native/legacy Boys-function sqrt modes
#   * human-readable and per-launch CSV output
#
# Mojo 1.0 migration points:
#   * GPU host APIs moved to max.gpu.host
#   * `def` replaces the old `fn` spelling
#   * raw pointers use Pointer and unsafe_offset=
#   * legacy raw allocation uses unsafe_alloc / unsafe_free explicitly
#   * Atomic moved to std.atomic
#   * enqueue_function takes the kernel function parameter once
#   * GPU ABI integer arguments use fixed-width integers, not Int
#   * execution_time uses an explicit runtime closure capture list

from std.python import PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import erf, exp, sqrt
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.atomic import Atomic
from std.sys import argv
from std.sys.info import is_nvidia_gpu
from std.sys.intrinsics import llvm_intrinsic

comptime PRECISION = 64
comptime dtype = DType.float32 if PRECISION == 32 else DType.float64
comptime real_t = Scalar[dtype]
comptime REAL_NAME = "fp32" if PRECISION == 32 else "fp64"

comptime LEGACY_SQRTF = False
comptime SQRT_MODE = "legacy-sqrtf" if LEGACY_SQRTF else "native"

comptime SQRTPI2_HOST = real_t(Float64(3.1415926535897931) ** Float64(-0.5) * 2.0)
comptime SQRTPI2_DEV = real_t(1.12837916709551255856)
comptime DTOL = real_t(1.0e-12)
comptime RCUT = real_t(1.0e-12)
comptime TOBOHRS = real_t(1.889725987722)

comptime BLK_SIZE = 256

comptime REF_SQRTPI2 = Float64(1.12837916709551255856)
comptime REF_DTOL = Float64(1.0e-12)
comptime REF_RCUT = Float64(1.0e-12)


def cfmt_f64(spec: String, v: Float64) raises -> String:
    var py_spec = PythonObject(spec)
    var py_v = PythonObject(v)
    var result = py_spec % py_v
    return String(result)


@always_inline
def sq[T: DType](x: Scalar[T]) -> Scalar[T]:
    return x * x


@always_inline
def r_exp(x: real_t) -> real_t:
    comptime if dtype == DType.float64:
        return real_t(exp(Float64(x)))
    else:
        return real_t(exp(Float32(x)))


@always_inline
def r_erf(x: real_t) -> real_t:
    comptime if dtype == DType.float64:
        return real_t(erf(Float64(x)))
    else:
        return real_t(erf(Float32(x)))


@always_inline
def r_pow(a: real_t, b: real_t) -> real_t:
    comptime if dtype == DType.float64:
        return real_t(Float64(a) ** Float64(b))
    else:
        return real_t(Float32(a) ** Float32(b))


@always_inline
def r_sqrt(x: real_t) -> real_t:
    comptime if dtype == DType.float64:
        comptime if is_nvidia_gpu():
            var d: Float64 = Float64(x)
            var r: Float64 = llvm_intrinsic["llvm.nvvm.sqrt.rn.d", Float64](d)
            return real_t(r)
        else:
            return real_t(sqrt(Float64(x)))
    else:
        return real_t(sqrt(Float32(x)))


@always_inline
def idx_sqrt(x: UInt64) -> UInt64:
    return UInt64(sqrt(Float32(x)))


@always_inline
def sqrt_boys(x: real_t) -> real_t:
    comptime if LEGACY_SQRTF:
        return real_t(sqrt(Float32(x)))
    else:
        return r_sqrt(x)


@always_inline
def atomic_add_real(ptr: Pointer[real_t, MutAnyOrigin], v: real_t):
    _ = Atomic[dtype].fetch_add(ptr, v)


def hartree_fock_kernel(
    nnnn: UInt64,
    ngauss_arg: Int64,
    natoms_arg: Int64,
    geom: Pointer[real_t, MutAnyOrigin],
    xpnt: Pointer[real_t, MutAnyOrigin],
    coef: Pointer[real_t, MutAnyOrigin],
    dens: Pointer[real_t, MutAnyOrigin],
    schwarz: Pointer[real_t, MutAnyOrigin],
    fock: Pointer[real_t, MutAnyOrigin],
):
    # Int is fine for device-local indexing; only host/device ABI arguments
    # need fixed-width integer types in Mojo 1.0.
    var ngauss = Int(ngauss_arg)
    var natoms = Int(natoms_arg)

    var ijkl = UInt64(block_idx.x) * UInt64(block_dim.x) + UInt64(thread_idx.x)
    if ijkl == 0 or ijkl > nnnn:
        return

    var nat = UInt64(natoms)

    var ij = idx_sqrt(2 * ijkl)
    var n = (ij * ij + ij) // 2
    while n < ijkl:
        ij += 1
        n = (ij * ij + ij) // 2
    var kl = ijkl - (ij * ij - ij) // 2

    if not (schwarz[unsafe_offset=Int(ij)] * schwarz[unsafe_offset=Int(kl)] > DTOL):
        return

    var i = idx_sqrt(2 * ij) - 1
    n = (i * i + i) // 2
    while n < ij:
        i += 1
        n = (i * i + i) // 2
    var j = ij - (i * i - i) // 2

    var k = idx_sqrt(2 * kl)
    n = (k * k + k) // 2
    while n < kl:
        k += 1
        n = (k * k + k) // 2
    var l = kl - (k * k - k) // 2
    i -= 1
    j -= 1
    k -= 1
    l -= 1

    var eri = real_t(0.0)
    var one_t = real_t(1.0)
    var p15 = real_t(1.5)
    var pm05 = real_t(-0.5)

    for ib in range(ngauss):
        for jb in range(ngauss):
            var aij = one_t / (xpnt[unsafe_offset=ib] + xpnt[unsafe_offset=jb])
            var dij = (
                coef[unsafe_offset=ib]
                * coef[unsafe_offset=jb]
                * r_exp(
                    -xpnt[unsafe_offset=ib]
                    * xpnt[unsafe_offset=jb]
                    * aij
                    * (
                        sq(geom[unsafe_offset=Int(i)] - geom[unsafe_offset=Int(j)])
                        + sq(geom[unsafe_offset=Int(nat + i)] - geom[unsafe_offset=Int(nat + j)])
                        + sq(geom[unsafe_offset=Int(2 * nat + i)] - geom[unsafe_offset=Int(2 * nat + j)])
                    )
                )
                * r_pow(aij, p15)
            )
            if abs(dij) > DTOL:
                var xij = aij * (xpnt[unsafe_offset=ib] * geom[unsafe_offset=Int(i)] + xpnt[unsafe_offset=jb] * geom[unsafe_offset=Int(j)])
                var yij = aij * (
                    xpnt[unsafe_offset=ib] * geom[unsafe_offset=Int(nat + i)] + xpnt[unsafe_offset=jb] * geom[unsafe_offset=Int(nat + j)]
                )
                var zij = aij * (
                    xpnt[unsafe_offset=ib] * geom[unsafe_offset=Int(2 * nat + i)]
                    + xpnt[unsafe_offset=jb] * geom[unsafe_offset=Int(2 * nat + j)]
                )
                for kb in range(ngauss):
                    for lb in range(ngauss):
                        var akl = one_t / (xpnt[unsafe_offset=kb] + xpnt[unsafe_offset=lb])
                        var dkl = (
                            dij
                            * coef[unsafe_offset=kb]
                            * coef[unsafe_offset=lb]
                            * r_exp(
                                -xpnt[unsafe_offset=kb]
                                * xpnt[unsafe_offset=lb]
                                * akl
                                * (
                                    sq(geom[unsafe_offset=Int(k)] - geom[unsafe_offset=Int(l)])
                                    + sq(geom[unsafe_offset=Int(nat + k)] - geom[unsafe_offset=Int(nat + l)])
                                    + sq(
                                        geom[unsafe_offset=Int(2 * nat + k)]
                                        - geom[unsafe_offset=Int(2 * nat + l)]
                                    )
                                )
                            )
                            * r_pow(akl, p15)
                        )
                        if abs(dkl) > DTOL:
                            var aijkl = (
                                (xpnt[unsafe_offset=ib] + xpnt[unsafe_offset=jb])
                                * (xpnt[unsafe_offset=kb] + xpnt[unsafe_offset=lb])
                                / (xpnt[unsafe_offset=ib] + xpnt[unsafe_offset=jb] + xpnt[unsafe_offset=kb] + xpnt[unsafe_offset=lb])
                            )
                            var tt = aijkl * (
                                sq(
                                    xij
                                    - akl
                                    * (
                                        xpnt[unsafe_offset=kb] * geom[unsafe_offset=Int(k)]
                                        + xpnt[unsafe_offset=lb] * geom[unsafe_offset=Int(l)]
                                    )
                                )
                                + sq(
                                    yij
                                    - akl
                                    * (
                                        xpnt[unsafe_offset=kb] * geom[unsafe_offset=Int(nat + k)]
                                        + xpnt[unsafe_offset=lb] * geom[unsafe_offset=Int(nat + l)]
                                    )
                                )
                                + sq(
                                    zij
                                    - akl
                                    * (
                                        xpnt[unsafe_offset=kb] * geom[unsafe_offset=Int(2 * nat + k)]
                                        + xpnt[unsafe_offset=lb] * geom[unsafe_offset=Int(2 * nat + l)]
                                    )
                                )
                            )
                            var f0t = SQRTPI2_DEV
                            if tt > RCUT:
                                f0t = r_pow(tt, pm05) * r_erf(sqrt_boys(tt))
                            eri += dkl * f0t * sqrt_boys(aijkl)

    if i == j:
        eri *= real_t(0.5)
    if k == l:
        eri *= real_t(0.5)
    if i == k and j == l:
        eri *= real_t(0.5)

    var four = real_t(4.0)
    atomic_add_real(fock.unsafe_offset(Int(i * nat + j)), dens[unsafe_offset=Int(k * nat + l)] * eri * four)
    atomic_add_real(fock.unsafe_offset(Int(k * nat + l)), dens[unsafe_offset=Int(i * nat + j)] * eri * four)
    atomic_add_real(fock.unsafe_offset(Int(i * nat + k)), -dens[unsafe_offset=Int(j * nat + l)] * eri)
    atomic_add_real(fock.unsafe_offset(Int(i * nat + l)), -dens[unsafe_offset=Int(j * nat + k)] * eri)
    atomic_add_real(fock.unsafe_offset(Int(j * nat + k)), -dens[unsafe_offset=Int(i * nat + l)] * eri)
    atomic_add_real(fock.unsafe_offset(Int(j * nat + l)), -dens[unsafe_offset=Int(i * nat + k)] * eri)


def ssss_host(
    i: Int, j: Int, k: Int, l: Int, ngauss: Int,
    xpnt: Pointer[real_t, MutUntrackedOrigin], coef: Pointer[real_t, MutUntrackedOrigin],
    geom: Pointer[real_t, MutUntrackedOrigin], natoms: Int,
) -> real_t:
    var eri = real_t(0.0)
    var one_t = real_t(1.0)
    var p15 = real_t(1.5)
    var pm05 = real_t(-0.5)
    for ib in range(ngauss):
        for jb in range(ngauss):
            var aij = one_t / (xpnt[unsafe_offset=ib] + xpnt[unsafe_offset=jb])
            var dij = (
                coef[unsafe_offset=ib] * coef[unsafe_offset=jb]
                * r_exp(-xpnt[unsafe_offset=ib] * xpnt[unsafe_offset=jb] * aij * (
                    sq(geom[unsafe_offset=i] - geom[unsafe_offset=j])
                    + sq(geom[unsafe_offset=natoms + i] - geom[unsafe_offset=natoms + j])
                    + sq(geom[unsafe_offset=2 * natoms + i] - geom[unsafe_offset=2 * natoms + j])))
                * r_pow(aij, p15)
            )
            if abs(dij) > DTOL:
                var xij = aij * (xpnt[unsafe_offset=ib] * geom[unsafe_offset=i] + xpnt[unsafe_offset=jb] * geom[unsafe_offset=j])
                var yij = aij * (xpnt[unsafe_offset=ib] * geom[unsafe_offset=natoms + i] + xpnt[unsafe_offset=jb] * geom[unsafe_offset=natoms + j])
                var zij = aij * (xpnt[unsafe_offset=ib] * geom[unsafe_offset=2 * natoms + i] + xpnt[unsafe_offset=jb] * geom[unsafe_offset=2 * natoms + j])
                for kb in range(ngauss):
                    for lb in range(ngauss):
                        var akl = one_t / (xpnt[unsafe_offset=kb] + xpnt[unsafe_offset=lb])
                        var dkl = (
                            dij * coef[unsafe_offset=kb] * coef[unsafe_offset=lb]
                            * r_exp(-xpnt[unsafe_offset=kb] * xpnt[unsafe_offset=lb] * akl * (
                                sq(geom[unsafe_offset=k] - geom[unsafe_offset=l])
                                + sq(geom[unsafe_offset=natoms + k] - geom[unsafe_offset=natoms + l])
                                + sq(geom[unsafe_offset=2 * natoms + k] - geom[unsafe_offset=2 * natoms + l])))
                            * r_pow(akl, p15)
                        )
                        if abs(dkl) > DTOL:
                            var aijkl = (
                                (xpnt[unsafe_offset=ib] + xpnt[unsafe_offset=jb]) * (xpnt[unsafe_offset=kb] + xpnt[unsafe_offset=lb])
                                / (xpnt[unsafe_offset=ib] + xpnt[unsafe_offset=jb] + xpnt[unsafe_offset=kb] + xpnt[unsafe_offset=lb])
                            )
                            var tt = aijkl * (
                                sq(xij - akl * (xpnt[unsafe_offset=kb] * geom[unsafe_offset=k] + xpnt[unsafe_offset=lb] * geom[unsafe_offset=l]))
                                + sq(yij - akl * (xpnt[unsafe_offset=kb] * geom[unsafe_offset=natoms + k] + xpnt[unsafe_offset=lb] * geom[unsafe_offset=natoms + l]))
                                + sq(zij - akl * (xpnt[unsafe_offset=kb] * geom[unsafe_offset=2 * natoms + k] + xpnt[unsafe_offset=lb] * geom[unsafe_offset=2 * natoms + l]))
                            )
                            var f0t = SQRTPI2_HOST
                            if tt > RCUT:
                                f0t = r_pow(tt, pm05) * r_erf(sqrt_boys(tt))
                            eri += dkl * f0t * sqrt_boys(aijkl)
    return eri


def ssss_ref(
    i: Int, j: Int, k: Int, l: Int, ngauss: Int,
    xpnt: Pointer[Float64, MutUntrackedOrigin], coef: Pointer[Float64, MutUntrackedOrigin],
    geom: Pointer[Float64, MutUntrackedOrigin], natoms: Int,
) -> Float64:
    var eri = Float64(0.0)
    for ib in range(ngauss):
        for jb in range(ngauss):
            var aij = Float64(1.0) / (xpnt[unsafe_offset=ib] + xpnt[unsafe_offset=jb])
            var dij = (
                coef[unsafe_offset=ib] * coef[unsafe_offset=jb]
                * exp(-xpnt[unsafe_offset=ib] * xpnt[unsafe_offset=jb] * aij * (
                    sq(geom[unsafe_offset=i] - geom[unsafe_offset=j])
                    + sq(geom[unsafe_offset=natoms + i] - geom[unsafe_offset=natoms + j])
                    + sq(geom[unsafe_offset=2 * natoms + i] - geom[unsafe_offset=2 * natoms + j])))
                * (aij ** Float64(1.5))
            )
            if abs(dij) > REF_DTOL:
                var xij = aij * (xpnt[unsafe_offset=ib] * geom[unsafe_offset=i] + xpnt[unsafe_offset=jb] * geom[unsafe_offset=j])
                var yij = aij * (xpnt[unsafe_offset=ib] * geom[unsafe_offset=natoms + i] + xpnt[unsafe_offset=jb] * geom[unsafe_offset=natoms + j])
                var zij = aij * (xpnt[unsafe_offset=ib] * geom[unsafe_offset=2 * natoms + i] + xpnt[unsafe_offset=jb] * geom[unsafe_offset=2 * natoms + j])
                for kb in range(ngauss):
                    for lb in range(ngauss):
                        var akl = Float64(1.0) / (xpnt[unsafe_offset=kb] + xpnt[unsafe_offset=lb])
                        var dkl = (
                            dij * coef[unsafe_offset=kb] * coef[unsafe_offset=lb]
                            * exp(-xpnt[unsafe_offset=kb] * xpnt[unsafe_offset=lb] * akl * (
                                sq(geom[unsafe_offset=k] - geom[unsafe_offset=l])
                                + sq(geom[unsafe_offset=natoms + k] - geom[unsafe_offset=natoms + l])
                                + sq(geom[unsafe_offset=2 * natoms + k] - geom[unsafe_offset=2 * natoms + l])))
                            * (akl ** Float64(1.5))
                        )
                        if abs(dkl) > REF_DTOL:
                            var aijkl = (
                                (xpnt[unsafe_offset=ib] + xpnt[unsafe_offset=jb]) * (xpnt[unsafe_offset=kb] + xpnt[unsafe_offset=lb])
                                / (xpnt[unsafe_offset=ib] + xpnt[unsafe_offset=jb] + xpnt[unsafe_offset=kb] + xpnt[unsafe_offset=lb])
                            )
                            var tt = aijkl * (
                                sq(xij - akl * (xpnt[unsafe_offset=kb] * geom[unsafe_offset=k] + xpnt[unsafe_offset=lb] * geom[unsafe_offset=l]))
                                + sq(yij - akl * (xpnt[unsafe_offset=kb] * geom[unsafe_offset=natoms + k] + xpnt[unsafe_offset=lb] * geom[unsafe_offset=natoms + l]))
                                + sq(zij - akl * (xpnt[unsafe_offset=kb] * geom[unsafe_offset=2 * natoms + k] + xpnt[unsafe_offset=lb] * geom[unsafe_offset=2 * natoms + l]))
                            )
                            var f0t = REF_SQRTPI2
                            if tt > REF_RCUT:
                                f0t = (tt ** Float64(-0.5)) * erf(sqrt(tt))
                            eri += dkl * f0t * sqrt(aijkl)
    return eri


def fock_build_cpu(
    ngauss: Int, natoms: Int,
    geom: Pointer[Float64, MutUntrackedOrigin], xpnt: Pointer[Float64, MutUntrackedOrigin],
    coef: Pointer[Float64, MutUntrackedOrigin], dens: Pointer[Float64, MutUntrackedOrigin],
    fock_out: Pointer[Float64, MutUntrackedOrigin],
):
    for t in range(natoms * natoms):
        fock_out[unsafe_offset=t] = 0.0
    var nn_ref = (natoms * natoms + natoms) // 2
    var schwarz = unsafe_alloc[Float64](nn_ref + 1)
    for t in range(nn_ref + 1):
        schwarz[unsafe_offset=t] = 0.0
    var ijr = 0
    for i in range(natoms):
        for j in range(i + 1):
            ijr += 1
            schwarz[unsafe_offset=ijr] = sqrt(abs(ssss_ref(i, j, i, j, ngauss, xpnt, coef, geom, natoms)))

    for i in range(natoms):
        for j in range(i + 1):
            var ij = i * (i + 1) // 2 + j + 1
            for k in range(natoms):
                for l in range(k + 1):
                    var kl = k * (k + 1) // 2 + l + 1
                    if kl > ij:
                        continue
                    if not (schwarz[unsafe_offset=ij] * schwarz[unsafe_offset=kl] > REF_DTOL):
                        continue
                    var eri = ssss_ref(i, j, k, l, ngauss, xpnt, coef, geom, natoms)
                    if i == j:
                        eri *= 0.5
                    if k == l:
                        eri *= 0.5
                    if i == k and j == l:
                        eri *= 0.5
                    fock_out[unsafe_offset=i * natoms + j] += dens[unsafe_offset=k * natoms + l] * eri * 4.0
                    fock_out[unsafe_offset=k * natoms + l] += dens[unsafe_offset=i * natoms + j] * eri * 4.0
                    fock_out[unsafe_offset=i * natoms + k] -= dens[unsafe_offset=j * natoms + l] * eri
                    fock_out[unsafe_offset=i * natoms + l] -= dens[unsafe_offset=j * natoms + k] * eri
                    fock_out[unsafe_offset=j * natoms + k] -= dens[unsafe_offset=i * natoms + l] * eri
                    fock_out[unsafe_offset=j * natoms + l] -= dens[unsafe_offset=i * natoms + k] * eri
    schwarz.unsafe_free()


def energy_from_fock(
    fock: Pointer[Float64, MutUntrackedOrigin], dens: Pointer[Float64, MutUntrackedOrigin], natoms: Int
) -> Float64:
    var e = Float64(0.0)
    for t in range(natoms * natoms):
        e += fock[unsafe_offset=t] * dens[unsafe_offset=t]
    return e * 0.5


def energy_symmetrized(
    fock: Pointer[Float64, MutUntrackedOrigin], dens: Pointer[Float64, MutUntrackedOrigin], natoms: Int
) -> Float64:
    var e = Float64(0.0)
    for i in range(natoms):
        for j in range(natoms):
            var f = fock[unsafe_offset=i * natoms + j]
            if i != j:
                f += fock[unsafe_offset=j * natoms + i]
            e += f * dens[unsafe_offset=i * natoms + j]
    return e * 0.5


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("ERROR: Please provide an input file")
        return

    var csv_output = False
    var run_check = True
    var force_check = False
    var max_check_atoms = 128
    var num_iter = 10
    for ai in range(2, len(args)):
        var s = String(args[ai])
        if s == "--csv":
            csv_output = True
        elif s == "--check":
            run_check = True
            force_check = True
        elif s == "--no-check":
            run_check = False
        elif s.startswith("--max-check-atoms="):
            max_check_atoms = Int(atol(s[byte=18:]))
        elif s.startswith("--iters="):
            num_iter = Int(atol(s[byte=8:]))

    var text: String
    with open(String(args[1]), "r") as f:
        text = f.read()
    var tok = text.split()
    var p = 0
    var ngauss = Int(atol(String(tok[p]))); p += 1
    var natoms = Int(atol(String(tok[p]))); p += 1

    var xpnt_d = unsafe_alloc[Float64](ngauss)
    var coef_d = unsafe_alloc[Float64](ngauss)
    for i in range(ngauss):
        xpnt_d[unsafe_offset=i] = atof(String(tok[p])); p += 1
        coef_d[unsafe_offset=i] = atof(String(tok[p])); p += 1
    var geom_d = unsafe_alloc[Float64](3 * natoms)
    for i in range(natoms):
        for j in range(3):
            geom_d[unsafe_offset=j * natoms + i] = atof(String(tok[p])); p += 1

    var nsq = natoms * natoms
    var h_xpnt = unsafe_alloc[real_t](ngauss)
    var h_coef = unsafe_alloc[real_t](ngauss)
    var h_geom = unsafe_alloc[real_t](3 * natoms)
    for i in range(ngauss):
        h_xpnt[unsafe_offset=i] = real_t(xpnt_d[unsafe_offset=i])
        h_coef[unsafe_offset=i] = real_t(coef_d[unsafe_offset=i])
    for i in range(3 * natoms):
        h_geom[unsafe_offset=i] = real_t(geom_d[unsafe_offset=i])

    var h_dens = unsafe_alloc[real_t](nsq)
    var dens_d = unsafe_alloc[Float64](nsq)
    for i in range(natoms):
        for j in range(natoms):
            h_dens[unsafe_offset=i * natoms + j] = real_t(0.1)
            dens_d[unsafe_offset=i * natoms + j] = 0.1
        h_dens[unsafe_offset=i * natoms + i] = real_t(1.0)
        dens_d[unsafe_offset=i * natoms + i] = 1.0

    for i in range(ngauss):
        h_coef[unsafe_offset=i] = h_coef[unsafe_offset=i] * r_pow(real_t(2.0) * h_xpnt[unsafe_offset=i], real_t(0.75))
        coef_d[unsafe_offset=i] = coef_d[unsafe_offset=i] * ((2.0 * xpnt_d[unsafe_offset=i]) ** Float64(0.75))
    for i in range(natoms):
        h_geom[unsafe_offset=0 * natoms + i] *= TOBOHRS
        h_geom[unsafe_offset=1 * natoms + i] *= TOBOHRS
        h_geom[unsafe_offset=2 * natoms + i] *= TOBOHRS
        geom_d[unsafe_offset=0 * natoms + i] *= 1.889725987722
        geom_d[unsafe_offset=1 * natoms + i] *= 1.889725987722
        geom_d[unsafe_offset=2 * natoms + i] *= 1.889725987722

    var h_fock = unsafe_alloc[real_t](nsq)
    for t in range(nsq):
        h_fock[unsafe_offset=t] = real_t(0.0)

    var nn = (natoms * natoms + natoms) // 2
    var h_schwarz = unsafe_alloc[real_t](nn + 1)
    for t in range(nn + 1):
        h_schwarz[unsafe_offset=t] = real_t(0.0)
    var ij = 0
    for i in range(natoms):
        for j in range(i + 1):
            ij += 1
            h_schwarz[unsafe_offset=ij] = sqrt(abs(ssss_host(i, j, i, j, ngauss, h_xpnt, h_coef, h_geom, natoms)))

    var nnnn = UInt64(nn) * UInt64(nn)
    nnnn = (nnnn + UInt64(nn)) // 2
    var n_blks = Int((nnnn + 1 + UInt64(BLK_SIZE) - 1) // UInt64(BLK_SIZE))

    var ctx = DeviceContext()
    var gpu_name = ctx.name()

    var d_xpnt = ctx.enqueue_create_buffer[dtype](ngauss)
    var d_coef = ctx.enqueue_create_buffer[dtype](ngauss)
    var d_geom = ctx.enqueue_create_buffer[dtype](3 * natoms)
    var d_dens = ctx.enqueue_create_buffer[dtype](nsq)
    var d_fock = ctx.enqueue_create_buffer[dtype](nsq)
    var d_schwarz = ctx.enqueue_create_buffer[dtype](nn + 1)

    ctx.enqueue_copy(d_xpnt, h_xpnt.as_imm())
    ctx.enqueue_copy(d_coef, h_coef.as_imm())
    ctx.enqueue_copy(d_geom, h_geom.as_imm())
    ctx.enqueue_copy(d_dens, h_dens.as_imm())
    ctx.enqueue_copy(d_fock, h_fock.as_imm())
    ctx.enqueue_copy(d_schwarz, h_schwarz.as_imm())
    ctx.synchronize()

    var ngauss_dev = Int64(ngauss)
    var natoms_dev = Int64(natoms)

    @always_inline
    def launch(c: DeviceContext) raises {d_geom, d_xpnt, d_coef, d_dens, d_schwarz, d_fock, nnnn, ngauss_dev, natoms_dev, n_blks}:
        c.enqueue_function[hartree_fock_kernel](
            nnnn,
            ngauss_dev,
            natoms_dev,
            d_geom,
            d_xpnt,
            d_coef,
            d_dens,
            d_schwarz,
            d_fock,
            grid_dim=n_blks,
            block_dim=BLK_SIZE,
        )

    launch(ctx)
    ctx.synchronize()

    ctx.enqueue_copy(h_fock, d_fock)
    ctx.synchronize()

    var erep = real_t(0.0)
    for i in range(natoms):
        for j in range(natoms):
            erep += h_fock[unsafe_offset=i * natoms + j] * h_dens[unsafe_offset=i * natoms + j]

    if not csv_output:
        print("2e- energy=" + cfmt_f64("%.16f", Float64(erep) * 0.5))
        if run_check and (force_check or natoms <= max_check_atoms):
            var fock_gpu = unsafe_alloc[Float64](nsq)
            var fock_ref = unsafe_alloc[Float64](nsq)
            for t in range(nsq):
                fock_gpu[unsafe_offset=t] = Float64(h_fock[unsafe_offset=t])
            fock_build_cpu(ngauss, natoms, geom_d, xpnt_d, coef_d, dens_d, fock_ref)

            var e_gpu = energy_from_fock(fock_gpu, dens_d, natoms)
            var e_ref = energy_from_fock(fock_ref, dens_d, natoms)
            var denom = abs(e_ref) if abs(e_ref) > 0.0 else 1.0
            var rel_loss = abs(e_gpu - e_ref) / denom
            var e_inpipe = Float64(erep) * 0.5
            var rel_inpipe = abs(e_inpipe - e_ref) / denom

            var maxd = Float64(0.0)
            var sumsq = Float64(0.0)
            var fscale = Float64(0.0)
            for t in range(nsq):
                var d = abs(fock_gpu[unsafe_offset=t] - fock_ref[unsafe_offset=t])
                if d > maxd:
                    maxd = d
                sumsq += d * d
                if abs(fock_ref[unsafe_offset=t]) > fscale:
                    fscale = abs(fock_ref[unsafe_offset=t])
            var rmsd = sqrt(sumsq / Float64(nsq))
            if fscale == 0.0:
                fscale = 1.0

            var asym = Float64(0.0)
            for i in range(natoms):
                for j in range(natoms):
                    var d = abs(fock_gpu[unsafe_offset=i * natoms + j] - fock_gpu[unsafe_offset=j * natoms + i])
                    if d > asym:
                        asym = d

            print("")
            print("--- precision check (build: " + REAL_NAME + ", sqrt: " + SQRT_MODE + ") ---")
            print("2e- energy (" + REAL_NAME + ", " + REAL_NAME + " reduction)      = "
                  + cfmt_f64("%.16f", e_inpipe))
            print("2e- energy (" + REAL_NAME + " Fock, fp64 reduction) = "
                  + cfmt_f64("%.16f", e_gpu))
            print("2e- energy (fp64 CPU reference)       = " + cfmt_f64("%.16f", e_ref))
            print("precision loss, Fock build only       = " + cfmt_f64("%.6e", rel_loss))
            print("precision loss, incl. " + REAL_NAME + " reduction  = "
                  + cfmt_f64("%.6e", rel_inpipe))
            print("max |F - F_ref| (elementwise)         = " + cfmt_f64("%.6e", maxd)
                  + "  (rel " + cfmt_f64("%.6e", maxd / fscale) + ")")
            print("rms |F - F_ref| (elementwise)         = " + cfmt_f64("%.6e", rmsd))
            print("max |F(i,j) - F(j,i)|                 = " + cfmt_f64("%.6e", asym))
            print("2e- energy from symmetrized F=A+A^T   = "
                  + cfmt_f64("%.16f", energy_symmetrized(fock_gpu, dens_d, natoms))
                  + "   (diagnostic)")
            print("---------------------------------------------------------------")
            fock_gpu.unsafe_free()
            fock_ref.unsafe_free()
        elif run_check:
            print("(correctness check skipped: natoms=" + String(natoms) + " > "
                  + String(max_check_atoms) + "; use --check to force)")
    else:
        print("backend,GPU,precision,sqrt_mode,natoms,ngauss,exec_time_ms")
        for _ in range(num_iter):
            ctx.synchronize()
            var ns = ctx.execution_time(launch, 1)
            var ms = Float64(ns) / 1.0e6
            print(
                "Mojo," + gpu_name + "," + REAL_NAME + "," + SQRT_MODE + ","
                + String(natoms) + "," + String(ngauss) + ","
                + cfmt_f64("%f", ms)
            )

    _ = d_xpnt
    _ = d_coef
    _ = d_geom
    _ = d_dens
    _ = d_fock
    _ = d_schwarz
    h_xpnt.unsafe_free(); h_coef.unsafe_free(); h_geom.unsafe_free()
    h_dens.unsafe_free(); h_fock.unsafe_free(); h_schwarz.unsafe_free()
    xpnt_d.unsafe_free(); coef_d.unsafe_free(); geom_d.unsafe_free(); dens_d.unsafe_free()
