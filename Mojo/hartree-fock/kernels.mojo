# Hartree-Fock device code (Mojo 1.2.0 / MAX 26.7): the two-electron Fock
# build kernel and the scalar helpers it shares with the host Schwarz pass.
#
# Kernel and helper math are unchanged from the original hartree-fock.mojo.
# Changes, all forced:
#   * precision and sqrt mode are run-time flags (--double / --float,
#     --legacy-sqrt), so the kernel and helpers take `dtype` and `legacy` as
#     compile-time parameters instead of the module-level `real_t` and
#     `LEGACY_SQRTF`; the constants DTOL / RCUT / SQRTPI2_DEV are converted to
#     `Scalar[dtype]` at the point of use, exactly as `real_t(...)` did;
#   * Mojo 1.1+: GPU index builtins come from max.gpu (std.gpu is private) and
#     `Atomic` is parameterized on the value type: `Atomic[Scalar[dtype]]`
#     instead of `Atomic[dtype]`.

from max.gpu import block_dim, block_idx, thread_idx
from std.atomic import Atomic
from std.math import erf, exp, sqrt
from std.memory import Pointer
from std.sys.info import is_nvidia_gpu
from std.sys.intrinsics import llvm_intrinsic

comptime SQRTPI2_HOST = Float64(3.1415926535897931) ** Float64(-0.5) * 2.0
comptime SQRTPI2_DEV = 1.12837916709551255856
comptime DTOL = 1.0e-12
comptime RCUT = 1.0e-12
comptime TOBOHRS = 1.889725987722


@always_inline
def sq[T: DType](x: Scalar[T]) -> Scalar[T]:
    return x * x


@always_inline
def r_exp[dtype: DType, //](x: Scalar[dtype]) -> Scalar[dtype]:
    comptime if dtype == DType.float64:
        return Scalar[dtype](exp(Float64(x)))
    else:
        return Scalar[dtype](exp(Float32(x)))


@always_inline
def r_erf[dtype: DType, //](x: Scalar[dtype]) -> Scalar[dtype]:
    comptime if dtype == DType.float64:
        return Scalar[dtype](erf(Float64(x)))
    else:
        return Scalar[dtype](erf(Float32(x)))


@always_inline
def r_pow[dtype: DType, //](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    comptime if dtype == DType.float64:
        return Scalar[dtype](Float64(a) ** Float64(b))
    else:
        return Scalar[dtype](Float32(a) ** Float32(b))


@always_inline
def r_sqrt[dtype: DType, //](x: Scalar[dtype]) -> Scalar[dtype]:
    comptime if dtype == DType.float64:
        comptime if is_nvidia_gpu():
            var d: Float64 = Float64(x)
            var r: Float64 = llvm_intrinsic["llvm.nvvm.sqrt.rn.d", Float64](d)
            return Scalar[dtype](r)
        else:
            return Scalar[dtype](sqrt(Float64(x)))
    else:
        return Scalar[dtype](sqrt(Float32(x)))


@always_inline
def idx_sqrt(x: UInt64) -> UInt64:
    return UInt64(sqrt(Float32(x)))


@always_inline
def sqrt_boys[dtype: DType, //, legacy: Bool](x: Scalar[dtype]) -> Scalar[dtype]:
    comptime if legacy:
        return Scalar[dtype](sqrt(Float32(x)))
    else:
        return r_sqrt(x)


@always_inline
def atomic_add_real[
    dtype: DType, //
](ptr: Pointer[Scalar[dtype], MutAnyOrigin], v: Scalar[dtype]):
    _ = Atomic[Scalar[dtype]].fetch_add(ptr, v)


def hartree_fock_kernel[
    dtype: DType, legacy: Bool
](
    nnnn: UInt64,
    ngauss_arg: Int64,
    natoms_arg: Int64,
    geom: Pointer[Scalar[dtype], MutAnyOrigin],
    xpnt: Pointer[Scalar[dtype], MutAnyOrigin],
    coef: Pointer[Scalar[dtype], MutAnyOrigin],
    dens: Pointer[Scalar[dtype], MutAnyOrigin],
    schwarz: Pointer[Scalar[dtype], MutAnyOrigin],
    fock: Pointer[Scalar[dtype], MutAnyOrigin],
):
    # Int is fine for device-local indexing; only host/device ABI arguments
    # need fixed-width integer types.
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

    if not (schwarz[unsafe_offset=Int(ij)] * schwarz[unsafe_offset=Int(kl)] > Scalar[dtype](DTOL)):
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

    var eri = Scalar[dtype](0.0)
    var one_t = Scalar[dtype](1.0)
    var p15 = Scalar[dtype](1.5)
    var pm05 = Scalar[dtype](-0.5)

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
            if abs(dij) > Scalar[dtype](DTOL):
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
                        if abs(dkl) > Scalar[dtype](DTOL):
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
                            var f0t = Scalar[dtype](SQRTPI2_DEV)
                            if tt > Scalar[dtype](RCUT):
                                f0t = r_pow(tt, pm05) * r_erf(sqrt_boys[legacy](tt))
                            eri += dkl * f0t * sqrt_boys[legacy](aijkl)

    if i == j:
        eri *= Scalar[dtype](0.5)
    if k == l:
        eri *= Scalar[dtype](0.5)
    if i == k and j == l:
        eri *= Scalar[dtype](0.5)

    var four = Scalar[dtype](4.0)
    atomic_add_real(fock.unsafe_offset(Int(i * nat + j)), dens[unsafe_offset=Int(k * nat + l)] * eri * four)
    atomic_add_real(fock.unsafe_offset(Int(k * nat + l)), dens[unsafe_offset=Int(i * nat + j)] * eri * four)
    atomic_add_real(fock.unsafe_offset(Int(i * nat + k)), -dens[unsafe_offset=Int(j * nat + l)] * eri)
    atomic_add_real(fock.unsafe_offset(Int(i * nat + l)), -dens[unsafe_offset=Int(j * nat + k)] * eri)
    atomic_add_real(fock.unsafe_offset(Int(j * nat + k)), -dens[unsafe_offset=Int(i * nat + l)] * eri)
    atomic_add_real(fock.unsafe_offset(Int(j * nat + l)), -dens[unsafe_offset=Int(i * nat + k)] * eri)
