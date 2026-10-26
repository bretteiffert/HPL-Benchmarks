# Hartree-Fock problem: input parsing, host preparation, the host Schwarz
# pass, device buffers and timed launches, and the fp64 CPU reference.
#
# The arithmetic follows the original hartree-fock.mojo line for line:
#   * input -> fp64 arrays, and separately -> run-precision arrays, with the
#     contraction-coefficient normalization and Bohr scaling done in each
#     precision exactly as before;
#   * the Schwarz bounds are computed on the host in the run precision
#     (ssss_host), the reference uses its own fp64 Schwarz pass;
#   * the CPU reference (ssss_ref, fock_build_cpu) and the energy functions are
#     unchanged.
# Changes: generic over precision / sqrt mode; run-time block size; errors for
# malformed input instead of crashes; no printing.

from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import erf, exp, sqrt
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.os.path import basename

from kernels import (
    hartree_fock_kernel,
    sq,
    r_erf,
    r_exp,
    r_pow,
    sqrt_boys,
    DTOL,
    RCUT,
    SQRTPI2_HOST,
    TOBOHRS,
)

comptime REF_SQRTPI2 = Float64(1.12837916709551255856)
comptime REF_DTOL = Float64(1.0e-12)
comptime REF_RCUT = Float64(1.0e-12)


# ===----------------------------------------------------------------------=== #
# Input
# ===----------------------------------------------------------------------=== #


def _tok_int(tok: List[String], p: Int, path: String, what: String) raises -> Int:
    if p >= len(tok):
        raise Error("input file '" + path + "' ends before " + what)
    try:
        return atol(tok[p])
    except:
        raise Error(
            "input file '" + path + "': " + what + " '" + tok[p]
            + "' is not an integer"
        )


def _tok_float(
    tok: List[String], p: Int, path: String, what: String
) raises -> Float64:
    if p >= len(tok):
        raise Error("input file '" + path + "' ends before " + what)
    try:
        return atof(tok[p])
    except:
        raise Error(
            "input file '" + path + "': " + what + " '" + tok[p]
            + "' is not a number"
        )


def _read_text(path: String) raises -> String:
    try:
        with open(path, "r") as f:
            return f.read()
    except e:
        raise Error("cannot read input file '" + path + "': " + String(e))


struct Deck(Movable):
    """The input file as parsed, in fp64, before normalization/scaling."""

    var path: String
    var name: String
    var ngauss: Int
    var natoms: Int
    var xpnt: List[Float64]
    var coef: List[Float64]
    var geom: List[Float64]  # geom[j * natoms + i], j = x / y / z

    def __init__(out self, path: String) raises:
        self.path = path
        self.name = basename(path)
        var text = _read_text(path)
        var tok = List[String]()
        for t in text.split():
            tok.append(String(t))
        var p = 0
        self.ngauss = _tok_int(tok, p, path, "the Gaussian count")
        p += 1
        self.natoms = _tok_int(tok, p, path, "the atom count")
        p += 1
        if self.ngauss < 1:
            raise Error("input file '" + path + "': Gaussian count must be >= 1")
        if self.natoms < 1:
            raise Error("input file '" + path + "': atom count must be >= 1")
        self.xpnt = List[Float64](length=self.ngauss, fill=0.0)
        self.coef = List[Float64](length=self.ngauss, fill=0.0)
        for i in range(self.ngauss):
            self.xpnt[i] = _tok_float(tok, p, path, "an exponent")
            p += 1
            self.coef[i] = _tok_float(tok, p, path, "a coefficient")
            p += 1
        self.geom = List[Float64](length=3 * self.natoms, fill=0.0)
        for i in range(self.natoms):
            for j in range(3):
                self.geom[j * self.natoms + i] = _tok_float(
                    tok, p, path, "a coordinate"
                )
                p += 1


def pair_count(natoms: Int) -> Int:
    return (natoms * natoms + natoms) // 2


def quartet_count(natoms: Int) -> UInt64:
    var nn = pair_count(natoms)
    var nnnn = UInt64(nn) * UInt64(nn)
    return (nnnn + UInt64(nn)) // 2


# ===----------------------------------------------------------------------=== #
# fp64 reference arrays (as the original's *_d arrays)
# ===----------------------------------------------------------------------=== #


struct RefArrays(Movable):
    var natoms: Int
    var ngauss: Int
    var xpnt: Pointer[Float64, MutUntrackedOrigin]
    var coef: Pointer[Float64, MutUntrackedOrigin]
    var geom: Pointer[Float64, MutUntrackedOrigin]
    var dens: Pointer[Float64, MutUntrackedOrigin]

    def __init__(out self, deck: Deck):
        var ngauss = deck.ngauss
        var natoms = deck.natoms
        self.natoms = natoms
        self.ngauss = ngauss
        self.xpnt = unsafe_alloc[Float64](ngauss)
        self.coef = unsafe_alloc[Float64](ngauss)
        for i in range(ngauss):
            self.xpnt[unsafe_offset=i] = deck.xpnt[i]
            self.coef[unsafe_offset=i] = deck.coef[i]
        self.geom = unsafe_alloc[Float64](3 * natoms)
        for t in range(3 * natoms):
            self.geom[unsafe_offset=t] = deck.geom[t]
        self.dens = unsafe_alloc[Float64](natoms * natoms)
        for i in range(natoms):
            for j in range(natoms):
                self.dens[unsafe_offset=i * natoms + j] = 0.1
            self.dens[unsafe_offset=i * natoms + i] = 1.0
        for i in range(ngauss):
            self.coef[unsafe_offset=i] = self.coef[unsafe_offset=i] * (
                (2.0 * self.xpnt[unsafe_offset=i]) ** Float64(0.75)
            )
        for i in range(natoms):
            self.geom[unsafe_offset=0 * natoms + i] *= 1.889725987722
            self.geom[unsafe_offset=1 * natoms + i] *= 1.889725987722
            self.geom[unsafe_offset=2 * natoms + i] *= 1.889725987722

    def free(self):
        self.xpnt.unsafe_free()
        self.coef.unsafe_free()
        self.geom.unsafe_free()
        self.dens.unsafe_free()


# ===----------------------------------------------------------------------=== #
# Run-precision host arrays (as the original's h_* arrays)
# ===----------------------------------------------------------------------=== #


struct HostArrays[dtype: DType, legacy: Bool](Movable):
    var natoms: Int
    var ngauss: Int
    var nn: Int
    var xpnt: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var coef: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var geom: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var dens: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var fock: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var schwarz: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]

    def __init__(out self, deck: Deck):
        comptime real_t = Scalar[Self.dtype]
        var ngauss = deck.ngauss
        var natoms = deck.natoms
        var nsq = natoms * natoms
        self.natoms = natoms
        self.ngauss = ngauss
        self.xpnt = unsafe_alloc[real_t](ngauss)
        self.coef = unsafe_alloc[real_t](ngauss)
        self.geom = unsafe_alloc[real_t](3 * natoms)
        for i in range(ngauss):
            self.xpnt[unsafe_offset=i] = real_t(deck.xpnt[i])
            self.coef[unsafe_offset=i] = real_t(deck.coef[i])
        for i in range(3 * natoms):
            self.geom[unsafe_offset=i] = real_t(deck.geom[i])

        self.dens = unsafe_alloc[real_t](nsq)
        for i in range(natoms):
            for j in range(natoms):
                self.dens[unsafe_offset=i * natoms + j] = real_t(0.1)
            self.dens[unsafe_offset=i * natoms + i] = real_t(1.0)

        for i in range(ngauss):
            self.coef[unsafe_offset=i] = self.coef[unsafe_offset=i] * r_pow(
                real_t(2.0) * self.xpnt[unsafe_offset=i], real_t(0.75)
            )
        for i in range(natoms):
            self.geom[unsafe_offset=0 * natoms + i] *= real_t(TOBOHRS)
            self.geom[unsafe_offset=1 * natoms + i] *= real_t(TOBOHRS)
            self.geom[unsafe_offset=2 * natoms + i] *= real_t(TOBOHRS)

        self.fock = unsafe_alloc[real_t](nsq)
        for t in range(nsq):
            self.fock[unsafe_offset=t] = real_t(0.0)

        var nn = pair_count(natoms)
        self.nn = nn
        self.schwarz = unsafe_alloc[real_t](nn + 1)
        for t in range(nn + 1):
            self.schwarz[unsafe_offset=t] = real_t(0.0)
        var ij = 0
        for i in range(natoms):
            for j in range(i + 1):
                ij += 1
                self.schwarz[unsafe_offset=ij] = sqrt(
                    abs(
                        ssss_host[Self.dtype, Self.legacy](
                            i, j, i, j, ngauss, self.xpnt, self.coef, self.geom,
                            natoms,
                        )
                    )
                )

    def e2_energy(self) -> Float64:
        """2e- energy from self.fock, reduced in the run precision (as the
        original's `erep`)."""
        var natoms = self.natoms
        var erep = Scalar[Self.dtype](0.0)
        for i in range(natoms):
            for j in range(natoms):
                erep += (
                    self.fock[unsafe_offset=i * natoms + j]
                    * self.dens[unsafe_offset=i * natoms + j]
                )
        return Float64(erep) * 0.5

    def free(self):
        self.xpnt.unsafe_free()
        self.coef.unsafe_free()
        self.geom.unsafe_free()
        self.dens.unsafe_free()
        self.fock.unsafe_free()
        self.schwarz.unsafe_free()


# ===----------------------------------------------------------------------=== #
# Device buffers and launches
# ===----------------------------------------------------------------------=== #


def _seconds(ns: Int) raises -> Float64:
    if ns <= 0:
        raise Error("device timer returned no measurement for the Fock build")
    return Float64(ns) * 1.0e-9


struct HFDevice[dtype: DType, legacy: Bool](Movable):
    var ctx: DeviceContext
    var d_xpnt: DeviceBuffer[Self.dtype]
    var d_coef: DeviceBuffer[Self.dtype]
    var d_geom: DeviceBuffer[Self.dtype]
    var d_dens: DeviceBuffer[Self.dtype]
    var d_fock: DeviceBuffer[Self.dtype]
    var d_schwarz: DeviceBuffer[Self.dtype]
    var nnnn: UInt64
    var n_blks: Int
    var block_size: Int
    var ngauss_dev: Int64
    var natoms_dev: Int64

    def __init__(
        out self,
        ctx: DeviceContext,
        host: HostArrays[Self.dtype, Self.legacy],
        block_size: Int,
    ) raises:
        var natoms = host.natoms
        var nsq = natoms * natoms
        self.ctx = ctx
        self.block_size = block_size
        self.nnnn = quartet_count(natoms)
        self.n_blks = Int(
            (self.nnnn + 1 + UInt64(block_size) - 1) // UInt64(block_size)
        )
        self.ngauss_dev = Int64(host.ngauss)
        self.natoms_dev = Int64(natoms)

        self.d_xpnt = ctx.enqueue_create_buffer[Self.dtype](host.ngauss)
        self.d_coef = ctx.enqueue_create_buffer[Self.dtype](host.ngauss)
        self.d_geom = ctx.enqueue_create_buffer[Self.dtype](3 * natoms)
        self.d_dens = ctx.enqueue_create_buffer[Self.dtype](nsq)
        self.d_fock = ctx.enqueue_create_buffer[Self.dtype](nsq)
        self.d_schwarz = ctx.enqueue_create_buffer[Self.dtype](host.nn + 1)

        ctx.enqueue_copy(self.d_xpnt, host.xpnt.as_imm())
        ctx.enqueue_copy(self.d_coef, host.coef.as_imm())
        ctx.enqueue_copy(self.d_geom, host.geom.as_imm())
        ctx.enqueue_copy(self.d_dens, host.dens.as_imm())
        ctx.enqueue_copy(self.d_fock, host.fock.as_imm())
        ctx.enqueue_copy(self.d_schwarz, host.schwarz.as_imm())
        ctx.synchronize()

    def run_untimed(self) raises:
        """One Fock build, not timed; its result is the one checked."""
        var d_geom = self.d_geom
        var d_xpnt = self.d_xpnt
        var d_coef = self.d_coef
        var d_dens = self.d_dens
        var d_schwarz = self.d_schwarz
        var d_fock = self.d_fock
        self.ctx.enqueue_function[
            hartree_fock_kernel[Self.dtype, Self.legacy]
        ](
            self.nnnn,
            self.ngauss_dev,
            self.natoms_dev,
            d_geom,
            d_xpnt,
            d_coef,
            d_dens,
            d_schwarz,
            d_fock,
            grid_dim=self.n_blks,
            block_dim=self.block_size,
        )
        self.ctx.synchronize()

    def run_timed(self) raises -> Float64:
        """One timed Fock build (keeps accumulating into the Fock matrix, as
        the original's timed loop did). Returns seconds."""
        var d_geom = self.d_geom
        var d_xpnt = self.d_xpnt
        var d_coef = self.d_coef
        var d_dens = self.d_dens
        var d_schwarz = self.d_schwarz
        var d_fock = self.d_fock
        var nnnn = self.nnnn
        var ngauss_dev = self.ngauss_dev
        var natoms_dev = self.natoms_dev
        var n_blks = self.n_blks
        var bs = self.block_size

        @always_inline
        def launch(c: DeviceContext) raises {d_geom, d_xpnt, d_coef, d_dens, d_schwarz, d_fock, nnnn, ngauss_dev, natoms_dev, n_blks, bs}:
            c.enqueue_function[hartree_fock_kernel[Self.dtype, Self.legacy]](
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
                block_dim=bs,
            )

        self.ctx.synchronize()
        var ns = self.ctx.execution_time(launch, 1)
        return _seconds(ns)

    def read_fock(self, host: HostArrays[Self.dtype, Self.legacy]) raises:
        self.ctx.enqueue_copy(host.fock, self.d_fock)
        self.ctx.synchronize()


# ===----------------------------------------------------------------------=== #
# Host Schwarz pass in the run precision (unchanged math)
# ===----------------------------------------------------------------------=== #


def ssss_host[
    dtype: DType, legacy: Bool
](
    i: Int, j: Int, k: Int, l: Int, ngauss: Int,
    xpnt: Pointer[Scalar[dtype], MutUntrackedOrigin], coef: Pointer[Scalar[dtype], MutUntrackedOrigin],
    geom: Pointer[Scalar[dtype], MutUntrackedOrigin], natoms: Int,
) -> Scalar[dtype]:
    var eri = Scalar[dtype](0.0)
    var one_t = Scalar[dtype](1.0)
    var p15 = Scalar[dtype](1.5)
    var pm05 = Scalar[dtype](-0.5)
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
            if abs(dij) > Scalar[dtype](DTOL):
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
                        if abs(dkl) > Scalar[dtype](DTOL):
                            var aijkl = (
                                (xpnt[unsafe_offset=ib] + xpnt[unsafe_offset=jb]) * (xpnt[unsafe_offset=kb] + xpnt[unsafe_offset=lb])
                                / (xpnt[unsafe_offset=ib] + xpnt[unsafe_offset=jb] + xpnt[unsafe_offset=kb] + xpnt[unsafe_offset=lb])
                            )
                            var tt = aijkl * (
                                sq(xij - akl * (xpnt[unsafe_offset=kb] * geom[unsafe_offset=k] + xpnt[unsafe_offset=lb] * geom[unsafe_offset=l]))
                                + sq(yij - akl * (xpnt[unsafe_offset=kb] * geom[unsafe_offset=natoms + k] + xpnt[unsafe_offset=lb] * geom[unsafe_offset=natoms + l]))
                                + sq(zij - akl * (xpnt[unsafe_offset=kb] * geom[unsafe_offset=2 * natoms + k] + xpnt[unsafe_offset=lb] * geom[unsafe_offset=2 * natoms + l]))
                            )
                            var f0t = Scalar[dtype](SQRTPI2_HOST)
                            if tt > Scalar[dtype](RCUT):
                                f0t = r_pow(tt, pm05) * r_erf(sqrt_boys[legacy](tt))
                            eri += dkl * f0t * sqrt_boys[legacy](aijkl)
    return eri




# ===----------------------------------------------------------------------=== #
# fp64 CPU reference and correctness metrics
# ===----------------------------------------------------------------------=== #


struct CheckResult(Copyable):
    var e_gpu: Float64  # run-precision Fock, fp64 reduction
    var e_ref: Float64  # fp64 CPU reference
    var rel_loss: Float64  # Fock build only
    var rel_inpipe: Float64  # incl. run-precision reduction
    var maxd: Float64
    var maxd_rel: Float64
    var rmsd: Float64
    var asym: Float64
    var e_sym: Float64

    def __init__(out self):
        self.e_gpu = 0.0
        self.e_ref = 0.0
        self.rel_loss = 0.0
        self.rel_inpipe = 0.0
        self.maxd = 0.0
        self.maxd_rel = 0.0
        self.rmsd = 0.0
        self.asym = 0.0
        self.e_sym = 0.0


def check_against_reference[
    dtype: DType, legacy: Bool
](
    ref_arrays: RefArrays,
    host: HostArrays[dtype, legacy],
    e_inpipe: Float64,
) -> CheckResult:
    """Metrics exactly as the original's precision check."""
    var natoms = ref_arrays.natoms
    var nsq = natoms * natoms
    var r = CheckResult()
    var fock_gpu = unsafe_alloc[Float64](nsq)
    var fock_ref = unsafe_alloc[Float64](nsq)
    for t in range(nsq):
        fock_gpu[unsafe_offset=t] = Float64(host.fock[unsafe_offset=t])
    fock_build_cpu(
        ref_arrays.ngauss,
        natoms,
        ref_arrays.geom,
        ref_arrays.xpnt,
        ref_arrays.coef,
        ref_arrays.dens,
        fock_ref,
    )

    r.e_gpu = energy_from_fock(fock_gpu, ref_arrays.dens, natoms)
    r.e_ref = energy_from_fock(fock_ref, ref_arrays.dens, natoms)
    var denom = abs(r.e_ref) if abs(r.e_ref) > 0.0 else 1.0
    r.rel_loss = abs(r.e_gpu - r.e_ref) / denom
    r.rel_inpipe = abs(e_inpipe - r.e_ref) / denom

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
    r.rmsd = sqrt(sumsq / Float64(nsq))
    if fscale == 0.0:
        fscale = 1.0
    r.maxd = maxd
    r.maxd_rel = maxd / fscale

    var asym = Float64(0.0)
    for i in range(natoms):
        for j in range(natoms):
            var d = abs(
                fock_gpu[unsafe_offset=i * natoms + j]
                - fock_gpu[unsafe_offset=j * natoms + i]
            )
            if d > asym:
                asym = d
    r.asym = asym
    r.e_sym = energy_symmetrized(fock_gpu, ref_arrays.dens, natoms)
    fock_gpu.unsafe_free()
    fock_ref.unsafe_free()
    return r^


# Unchanged from the original: fp64 two-electron integral, CPU Fock build and
# energy functions.


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
