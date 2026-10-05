# Mojo 1.0.0 / MAX 26.5 port of the miniBUDE GPU benchmark.
# Loads the reference binary deck, benchmarks selected PPWI/work-group configurations
# with device-event timing, and reports performance plus numerical metrics.
#
# Preserves the original precision, PPWI dispatch, deck/golden comparison, CSV output,
# and custom formatting behavior. PRECISION_BITS remains a source-level comptime value.
# This native-math variant uses std.math.sin/cos directly. On Mojo 1.0.0 / MAX 26.5,
# GPU fp32 is supported; GPU fp64 trig is expected to fail on NVIDIA/AMD backends.

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import cos, isfinite, sin, sqrt
from std.memory import Pointer
from std.sys import argv
from std.time import perf_counter_ns
from std.utils.numerics import max_finite
from std.collections import List, InlineArray

comptime DEFAULT_ITERS = 8
comptime DEFAULT_WARMUP = 2
comptime DEFAULT_ENERGY_ENTRIES = 8
comptime DEFAULT_DATA_DIR = "../../data/miniBUDE/bm1"

comptime REPORT_TOLERANCE_PCT = 0.025

comptime PRECISION_BITS = 64

comptime FP = DType.float32 if PRECISION_BITS == 32 else DType.float64


comptime HBTYPE_F: Int32 = 70
comptime HBTYPE_E: Int32 = 69

comptime PPWIS_STR = "1,2,4,8,16,32,64,128"


def is_supported_ppwi(p: Int) -> Bool:
    return p == 1 or p == 2 or p == 4 or p == 8 or p == 16 or p == 32 or p == 64 or p == 128


comptime ULP_NBUCKETS = 10


def ulp_bucket_bound(i: Int) -> Int64:
    if i == 0:
        return 1
    if i == 1:
        return 2
    if i == 2:
        return 4
    if i == 3:
        return 8
    if i == 4:
        return 16
    if i == 5:
        return 64
    if i == 6:
        return 256
    if i == 7:
        return 1024
    return 4096


def fasten_main[
    dtype: DType, ppwi: Int
](
    etotals: Pointer[Scalar[dtype], MutAnyOrigin],
    prot_x: Pointer[Scalar[dtype], MutAnyOrigin],
    prot_y: Pointer[Scalar[dtype], MutAnyOrigin],
    prot_z: Pointer[Scalar[dtype], MutAnyOrigin],
    prot_t: Pointer[Int32, MutAnyOrigin],
    lig_x: Pointer[Scalar[dtype], MutAnyOrigin],
    lig_y: Pointer[Scalar[dtype], MutAnyOrigin],
    lig_z: Pointer[Scalar[dtype], MutAnyOrigin],
    lig_t: Pointer[Int32, MutAnyOrigin],
    t0: Pointer[Scalar[dtype], MutAnyOrigin],
    t1: Pointer[Scalar[dtype], MutAnyOrigin],
    t2: Pointer[Scalar[dtype], MutAnyOrigin],
    t3: Pointer[Scalar[dtype], MutAnyOrigin],
    t4: Pointer[Scalar[dtype], MutAnyOrigin],
    t5: Pointer[Scalar[dtype], MutAnyOrigin],
    ff_hbtype: Pointer[Int32, MutAnyOrigin],
    ff_radius: Pointer[Scalar[dtype], MutAnyOrigin],
    ff_hphb: Pointer[Scalar[dtype], MutAnyOrigin],
    ff_elsc: Pointer[Scalar[dtype], MutAnyOrigin],
    natlig: Int32,
    natpro: Int32,
    num_transforms: Int32,
):
    comptime assert dtype.is_floating_point(), "fasten_main requires a floating-point dtype"

    comptime ZERO = Scalar[dtype](0)
    comptime QUARTER = Scalar[dtype](0.25)
    comptime HALF = Scalar[dtype](0.5)
    comptime ONE = Scalar[dtype](1)
    comptime TWO = Scalar[dtype](2)
    comptime FOUR = Scalar[dtype](4)
    comptime CNSTNT = Scalar[dtype](45)
    comptime HARDNESS = Scalar[dtype](38)
    comptime NPNPDIST = Scalar[dtype](5.5)
    comptime NPPDIST = Scalar[dtype](1)
    comptime MAXVAL = max_finite[dtype]()

    var lsz = Int32(block_dim.x)

    var base = Int32(block_idx.x) * lsz * Int32(ppwi) + Int32(thread_idx.x)
    var ix = base if base < num_transforms else (
        num_transforms - Int32(ppwi) * lsz + Int32(thread_idx.x)
    )

    var etot = InlineArray[Scalar[dtype], ppwi](fill=ZERO)
    var tf = InlineArray[Scalar[dtype], ppwi * 12](uninitialized=True)
    var lpos = InlineArray[Scalar[dtype], ppwi * 3](uninitialized=True)

    comptime for i in range(ppwi):
        var index = Int(ix + Int32(i) * lsz)

        var sx = sin(t0[unsafe_offset=index])
        var cx = cos(t0[unsafe_offset=index])
        var sy = sin(t1[unsafe_offset=index])
        var cy = cos(t1[unsafe_offset=index])
        var sz = sin(t2[unsafe_offset=index])
        var cz = cos(t2[unsafe_offset=index])

        comptime b = i * 12
        tf[b + 0] = cy * cz
        tf[b + 1] = sx * sy * cz - cx * sz
        tf[b + 2] = cx * sy * cz + sx * sz
        tf[b + 3] = t3[unsafe_offset=index]
        tf[b + 4] = cy * sz
        tf[b + 5] = sx * sy * sz + cx * cz
        tf[b + 6] = cx * sy * sz - sx * cz
        tf[b + 7] = t4[unsafe_offset=index]
        tf[b + 8] = -sy
        tf[b + 9] = sx * cy
        tf[b + 10] = cx * cy
        tf[b + 11] = t5[unsafe_offset=index]

    for il in range(Int(natlig)):
        var lx = lig_x[unsafe_offset=il]
        var ly = lig_y[unsafe_offset=il]
        var lz = lig_z[unsafe_offset=il]
        var ltype = Int(lig_t[unsafe_offset=il])

        var l_radius = ff_radius[unsafe_offset=ltype]
        var l_hphb = ff_hphb[unsafe_offset=ltype]
        var l_elsc = ff_elsc[unsafe_offset=ltype]
        var l_hbtype = ff_hbtype[unsafe_offset=ltype]

        var lhphb_ltz = l_hphb < ZERO
        var lhphb_gtz = l_hphb > ZERO

        comptime for i in range(ppwi):
            comptime b = i * 12
            comptime o = i * 3
            lpos[o + 0] = tf[b + 3] + lx * tf[b + 0] + ly * tf[b + 1] + lz * tf[b + 2]
            lpos[o + 1] = tf[b + 7] + lx * tf[b + 4] + ly * tf[b + 5] + lz * tf[b + 6]
            lpos[o + 2] = tf[b + 11] + lx * tf[b + 8] + ly * tf[b + 9] + lz * tf[b + 10]

        for ip in range(Int(natpro)):
            var px = prot_x[unsafe_offset=ip]
            var py = prot_y[unsafe_offset=ip]
            var pz = prot_z[unsafe_offset=ip]
            var ptype = Int(prot_t[unsafe_offset=ip])

            var p_radius = ff_radius[unsafe_offset=ptype]
            var p_hphb = ff_hphb[unsafe_offset=ptype]
            var p_elsc = ff_elsc[unsafe_offset=ptype]
            var p_hbtype = ff_hbtype[unsafe_offset=ptype]

            var radij = p_radius + l_radius
            var r_radij = ONE / radij

            var both_f = (p_hbtype == HBTYPE_F) and (l_hbtype == HBTYPE_F)
            var elcdst = FOUR if both_f else TWO
            var elcdst1 = QUARTER if both_f else HALF
            var type_E = (p_hbtype == HBTYPE_E) or (l_hbtype == HBTYPE_E)

            var phphb_ltz = p_hphb < ZERO
            var phphb_gtz = p_hphb > ZERO
            var phphb_nz = p_hphb != ZERO
            var p_hphb_s = p_hphb * (
                -ONE if (phphb_ltz and lhphb_gtz) else ONE
            )
            var l_hphb_s = l_hphb * (
                -ONE if (phphb_gtz and lhphb_ltz) else ONE
            )

            var distdslv: Scalar[dtype]
            if phphb_ltz:
                distdslv = NPNPDIST if lhphb_ltz else NPPDIST
            else:
                distdslv = NPPDIST if lhphb_ltz else -MAXVAL
            var r_distdslv = ONE / distdslv

            var chrg_init = l_elsc * p_elsc
            var dslv_init = p_hphb_s + l_hphb_s

            comptime for i in range(ppwi):
                comptime o = i * 3
                var x = lpos[o + 0] - px
                var y = lpos[o + 1] - py
                var z = lpos[o + 2] - pz
                var distij = sqrt(x * x + y * y + z * z)

                var distbb = distij - radij
                var zone1 = distbb < ZERO

                etot[i] += (ONE - (distij * r_radij)) * (
                    TWO * HARDNESS if zone1 else ZERO
                )

                var chrg_e = chrg_init * (
                    (ONE if zone1 else (ONE - distbb * elcdst1))
                    * (ONE if distbb < elcdst else ZERO)
                )
                var neg_chrg_e = -abs(chrg_e)
                chrg_e = neg_chrg_e if type_E else chrg_e
                etot[i] += chrg_e * CNSTNT

                var coeff = ONE - (distbb * r_distdslv)
                var dslv_e = dslv_init * (
                    ONE if (distbb < distdslv and phphb_nz) else ZERO
                )
                dslv_e *= ONE if zone1 else coeff
                etot[i] += dslv_e

    if base < num_transforms:

        comptime for i in range(ppwi):
            etotals[unsafe_offset=Int(base + Int32(i) * lsz)] = etot[i] * HALF


def _pow10(n: Int) -> Float64:
    var p = 1.0
    for _ in range(n):
        p *= 10.0
    return p


def _ipow10(n: Int) -> Int:
    var p = 1
    for _ in range(n):
        p *= 10
    return p


def _render(whole: Int, nd: Int, neg: Bool) -> String:
    var scale = _ipow10(nd)
    var ip = whole // scale
    var fp = whole - ip * scale
    var out = String(ip)
    if nd > 0:
        var frac = String(fp)
        while frac.byte_length() < nd:
            frac = "0" + frac
        out += "." + frac
    return ("-" + out) if neg else out


def fixed(x: Float64, nd: Int) -> String:
    if x != x:
        return "nan"
    var neg = x < 0.0
    var v = -x if neg else x
    var whole = Int(v * _pow10(nd) + 0.5)
    return _render(whole, nd, neg)


def sci(x: Float64, nd: Int) -> String:
    if x != x:
        return "nan"
    var neg = x < 0.0
    var v = -x if neg else x
    if v == 0.0:
        return ("-" if neg else "") + _render(0, nd, False) + "e+00"
    var e = 0
    while v >= 10.0:
        v /= 10.0
        e += 1
    while v < 1.0:
        v *= 10.0
        e -= 1
    var scale = _ipow10(nd)
    var whole = Int(v * Float64(scale) + 0.5)
    if whole >= 10 * scale:
        whole = whole // 10
        e += 1
    var esign = "-" if e < 0 else "+"
    var ea = -e if e < 0 else e
    var edig = String(ea)
    if edig.byte_length() < 2:
        edig = "0" + edig
    return ("-" if neg else "") + _render(whole, nd, False) + "e" + esign + edig


def gfmt(x: Float64) -> String:
    if x == 0.0:
        return "0"
    if x != x:
        return "nan"
    var neg = x < 0.0
    var v = -x if neg else x
    var e = 0
    var m = v
    while m >= 10.0:
        m /= 10.0
        e += 1
    while m < 1.0:
        m *= 10.0
        e -= 1
    if e < -4 or e >= 6:
        return sci(x, 5)
    var nd = 5 - e
    if nd < 0:
        nd = 0
    var whole = Int(v * _pow10(nd) + 0.5)
    while nd > 0 and whole % 10 == 0:
        whole = whole // 10
        nd -= 1
    return _render(whole, nd, neg)


def ulp_ordered_f32(f: Float32) -> Int64:
    var u = Int64(f.to_bits())
    if u >= 0x80000000:
        return 0x80000000 - u
    return u


def ulp_ordered_f64(d: Float64) -> Int64:
    var u = UInt64(d.to_bits())
    comptime SIGN = UInt64(1) << 63
    if u >= SIGN:
        return -Int64(u - SIGN)
    return Int64(u)


def ulp_distance_f32(a: Float32, b: Float32) -> Int64:
    if a != a or b != b:
        return Int64.MAX
    if a == b:
        return 0
    var d = ulp_ordered_f32(a) - ulp_ordered_f32(b)
    return -d if d < 0 else d


def ulp_distance_f64(a: Float64, b: Float64) -> Int64:
    if a != a or b != b:
        return Int64.MAX
    if a == b:
        return 0
    var d = ulp_ordered_f64(a) - ulp_ordered_f64(b)
    return -d if d < 0 else d


def ulp_bucket_label(i: Int) -> String:
    if i == 0:
        return "0"
    if i == ULP_NBUCKETS - 1:
        return ">=" + String(ulp_bucket_bound(ULP_NBUCKETS - 2))
    var lo = ulp_bucket_bound(i - 1)
    var hi = ulp_bucket_bound(i) - 1
    if lo == hi:
        return String(lo)
    return String(lo) + "-" + String(hi)


@fieldwise_init
struct Metrics(Copyable, Movable):
    var l2_rel_norm: Float64
    var rms_abs: Float64
    var mean_signed: Float64
    var max_abs: Float64
    var max_abs_idx: Int
    var max_rel: Float64
    var max_rel_idx: Int
    var max_ulp: Int64
    var max_ulp_idx: Int
    var ulp_histogram: InlineArray[Int, ULP_NBUCKETS]
    var compared: Int
    var non_finite: Int
    var beyond_tolerance: Int


def empty_metrics() -> Metrics:
    var h = InlineArray[Int, ULP_NBUCKETS](fill=0)
    return Metrics(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0, h^, 0, 0, 0)


struct Kahan(Copyable, Movable):
    var sum: Float64
    var c: Float64

    def __init__(out self):
        self.sum = 0.0
        self.c = 0.0

    def add(mut self, x: Float64):
        var y = x - self.c
        var t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t


def compute_metrics(
    measured: List[Float64], reference: List[Float64], precision_bits: Int
) -> Metrics:
    var m = empty_metrics()
    var n = min(len(measured), len(reference))

    var err_sq = Kahan()
    var ref_sq = Kahan()
    var signed_err = Kahan()

    for i in range(n):
        var x = measured[i]
        var r = reference[i]

        if not isfinite(x) or not isfinite(r):
            m.non_finite += 1
            continue

        var err = x - r
        var abs_err = -err if err < 0.0 else err

        err_sq.add(err * err)
        ref_sq.add(r * r)
        signed_err.add(err)

        if abs_err > m.max_abs:
            m.max_abs = abs_err
            m.max_abs_idx = i

        var denom = -r if r < 0.0 else r
        if denom > 0.0:
            var rel = abs_err / denom
            if rel > m.max_rel:
                m.max_rel = rel
                m.max_rel_idx = i
            if rel * 100.0 > REPORT_TOLERANCE_PCT:
                m.beyond_tolerance += 1

        var u: Int64
        if precision_bits == 32:
            u = ulp_distance_f32(Float32(x), Float32(r))
        else:
            u = ulp_distance_f64(x, r)
        if u > m.max_ulp:
            m.max_ulp = u
            m.max_ulp_idx = i
        var bucket = ULP_NBUCKETS - 1
        for b in range(ULP_NBUCKETS - 1):
            if u < ulp_bucket_bound(b):
                bucket = b
                break
        m.ulp_histogram[bucket] += 1

        m.compared += 1

    if m.compared > 0:
        m.rms_abs = sqrt(err_sq.sum / Float64(m.compared))
        m.mean_signed = signed_err.sum / Float64(m.compared)
        if ref_sq.sum > 0.0:
            m.l2_rel_norm = sqrt(err_sq.sum) / sqrt(ref_sq.sum)
    return m^


def compare_to_deck(
    measured: List[Float64], ref_energies: List[Float32]
) -> Metrics:
    var n = min(len(ref_energies), len(measured))
    var ref_vals = List[Float64]()
    for i in range(n):
        ref_vals.append(Float64(ref_energies[i]))
    return compute_metrics(measured, ref_vals, 64)


@fieldwise_init
struct Deck(Copyable, Movable):
    var prot_x: List[Float32]
    var prot_y: List[Float32]
    var prot_z: List[Float32]
    var prot_t: List[Int32]
    var lig_x: List[Float32]
    var lig_y: List[Float32]
    var lig_z: List[Float32]
    var lig_t: List[Int32]
    var ff_hbtype: List[Int32]
    var ff_radius: List[Float32]
    var ff_hphb: List[Float32]
    var ff_elsc: List[Float32]
    var poses: List[List[Float32]]
    var ref_energies: List[Float32]

    def natpro(self) -> Int:
        return len(self.prot_x)

    def natlig(self) -> Int:
        return len(self.lig_x)

    def nposes(self) -> Int:
        return len(self.poses[0])


def le_f32(b: List[UInt8], off: Int) -> Float32:
    return b.unsafe_ptr().unsafe_offset(off).unsafe_bitcast[Float32]()[]


def le_i32(b: List[UInt8], off: Int) -> Int32:
    return b.unsafe_ptr().unsafe_offset(off).unsafe_bitcast[Int32]()[]


def read_bytes(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        return f.read_bytes()


@fieldwise_init
struct Sample(Copyable, Movable):
    var ppwi: Int
    var wgsize: Int
    var precision_bits: Int
    var energies: List[Float64]
    var kernel_millis: List[Float64]
    var wall_millis: List[Float64]
    var context_millis: Float64


def fasten[
    dtype: DType, ppwi: Int
](ctx: DeviceContext, deck: Deck, wgsize: Int, iters: Int, warmup: Int) raises -> Sample:
    var n = deck.nposes()
    var natpro = deck.natpro()
    var natlig = deck.natlig()
    var ntypes = len(deck.ff_hbtype)

    var h_prot_x = List[Scalar[dtype]]()
    var h_prot_y = List[Scalar[dtype]]()
    var h_prot_z = List[Scalar[dtype]]()
    for i in range(natpro):
        h_prot_x.append(Scalar[dtype](deck.prot_x[i]))
        h_prot_y.append(Scalar[dtype](deck.prot_y[i]))
        h_prot_z.append(Scalar[dtype](deck.prot_z[i]))

    var h_lig_x = List[Scalar[dtype]]()
    var h_lig_y = List[Scalar[dtype]]()
    var h_lig_z = List[Scalar[dtype]]()
    for i in range(natlig):
        h_lig_x.append(Scalar[dtype](deck.lig_x[i]))
        h_lig_y.append(Scalar[dtype](deck.lig_y[i]))
        h_lig_z.append(Scalar[dtype](deck.lig_z[i]))

    var h_ff_rad = List[Scalar[dtype]]()
    var h_ff_hphb = List[Scalar[dtype]]()
    var h_ff_elsc = List[Scalar[dtype]]()
    for i in range(ntypes):
        h_ff_rad.append(Scalar[dtype](deck.ff_radius[i]))
        h_ff_hphb.append(Scalar[dtype](deck.ff_hphb[i]))
        h_ff_elsc.append(Scalar[dtype](deck.ff_elsc[i]))

    # Keep all six pose host buffers alive until the asynchronous H->D copies
    # have completed. This also avoids origin tracking through List[DeviceBuffer].
    var h_t0 = List[Scalar[dtype]]()
    var h_t1 = List[Scalar[dtype]]()
    var h_t2 = List[Scalar[dtype]]()
    var h_t3 = List[Scalar[dtype]]()
    var h_t4 = List[Scalar[dtype]]()
    var h_t5 = List[Scalar[dtype]]()
    for i in range(n):
        h_t0.append(Scalar[dtype](deck.poses[0][i]))
        h_t1.append(Scalar[dtype](deck.poses[1][i]))
        h_t2.append(Scalar[dtype](deck.poses[2][i]))
        h_t3.append(Scalar[dtype](deck.poses[3][i]))
        h_t4.append(Scalar[dtype](deck.poses[4][i]))
        h_t5.append(Scalar[dtype](deck.poses[5][i]))

    var context_start = perf_counter_ns()

    var d_prot_x = ctx.enqueue_create_buffer[dtype](natpro)
    var d_prot_y = ctx.enqueue_create_buffer[dtype](natpro)
    var d_prot_z = ctx.enqueue_create_buffer[dtype](natpro)
    var d_prot_t = ctx.enqueue_create_buffer[DType.int32](natpro)
    var d_lig_x = ctx.enqueue_create_buffer[dtype](natlig)
    var d_lig_y = ctx.enqueue_create_buffer[dtype](natlig)
    var d_lig_z = ctx.enqueue_create_buffer[dtype](natlig)
    var d_lig_t = ctx.enqueue_create_buffer[DType.int32](natlig)
    var d_ff_hb = ctx.enqueue_create_buffer[DType.int32](ntypes)
    var d_ff_rad = ctx.enqueue_create_buffer[dtype](ntypes)
    var d_ff_hphb = ctx.enqueue_create_buffer[dtype](ntypes)
    var d_ff_elsc = ctx.enqueue_create_buffer[dtype](ntypes)
    var d_out = ctx.enqueue_create_buffer[dtype](n)

    var d_t0 = ctx.enqueue_create_buffer[dtype](n)
    var d_t1 = ctx.enqueue_create_buffer[dtype](n)
    var d_t2 = ctx.enqueue_create_buffer[dtype](n)
    var d_t3 = ctx.enqueue_create_buffer[dtype](n)
    var d_t4 = ctx.enqueue_create_buffer[dtype](n)
    var d_t5 = ctx.enqueue_create_buffer[dtype](n)

    ctx.enqueue_copy(d_prot_x, h_prot_x.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_prot_y, h_prot_y.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_prot_z, h_prot_z.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_prot_t, deck.prot_t.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_lig_x, h_lig_x.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_lig_y, h_lig_y.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_lig_z, h_lig_z.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_lig_t, deck.lig_t.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_ff_hb, deck.ff_hbtype.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_ff_rad, h_ff_rad.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_ff_hphb, h_ff_hphb.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_ff_elsc, h_ff_elsc.unsafe_ptr().as_imm())

    ctx.enqueue_copy(d_t0, h_t0.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_t1, h_t1.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_t2, h_t2.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_t3, h_t3.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_t4, h_t4.unsafe_ptr().as_imm())
    ctx.enqueue_copy(d_t5, h_t5.unsafe_ptr().as_imm())

    ctx.synchronize()
    var context_millis = Float64(perf_counter_ns() - context_start) * 1e-6

    var blocks = ((n + ppwi - 1) // ppwi + wgsize - 1) // wgsize

    # The GPU ABI uses fixed-width integer arguments.
    var natlig_dev = Int32(natlig)
    var natpro_dev = Int32(natpro)
    var n_dev = Int32(n)

    @always_inline
    def launch(c: DeviceContext) raises {
        d_out,
        d_prot_x,
        d_prot_y,
        d_prot_z,
        d_prot_t,
        d_lig_x,
        d_lig_y,
        d_lig_z,
        d_lig_t,
        d_t0,
        d_t1,
        d_t2,
        d_t3,
        d_t4,
        d_t5,
        d_ff_hb,
        d_ff_rad,
        d_ff_hphb,
        d_ff_elsc,
        natlig_dev,
        natpro_dev,
        n_dev,
        blocks,
        wgsize,
    }:
        c.enqueue_function[fasten_main[dtype, ppwi]](
            d_out,
            d_prot_x,
            d_prot_y,
            d_prot_z,
            d_prot_t,
            d_lig_x,
            d_lig_y,
            d_lig_z,
            d_lig_t,
            d_t0,
            d_t1,
            d_t2,
            d_t3,
            d_t4,
            d_t5,
            d_ff_hb,
            d_ff_rad,
            d_ff_hphb,
            d_ff_elsc,
            natlig_dev,
            natpro_dev,
            n_dev,
            grid_dim=blocks,
            block_dim=wgsize,
        )

    var kernel_millis = List[Float64]()
    var wall_millis = List[Float64]()

    for _ in range(iters + warmup):
        var wall_start = perf_counter_ns()
        var ns = ctx.execution_time(launch, 1)
        var wall_ns = perf_counter_ns() - wall_start

        kernel_millis.append(Float64(ns) * 1e-6)
        wall_millis.append(Float64(wall_ns) * 1e-6)

    var host_out = List[Scalar[dtype]]()
    for _ in range(n):
        host_out.append(Scalar[dtype](0))
    ctx.enqueue_copy(host_out.unsafe_ptr(), d_out)
    ctx.synchronize()

    var energies = List[Float64]()
    for i in range(n):
        energies.append(Float64(host_out[i]))

    return Sample(
        ppwi,
        wgsize,
        64 if dtype == DType.float64 else 32,
        energies^,
        kernel_millis^,
        wall_millis^,
        context_millis,
    )

def fasten_dispatch[
    dtype: DType
](ctx: DeviceContext, deck: Deck, wgsize: Int, ppwi: Int, iters: Int, warmup: Int) raises -> Sample:
    if ppwi == 1:
        return fasten[dtype, 1](ctx, deck, wgsize, iters, warmup)
    if ppwi == 2:
        return fasten[dtype, 2](ctx, deck, wgsize, iters, warmup)
    if ppwi == 4:
        return fasten[dtype, 4](ctx, deck, wgsize, iters, warmup)
    if ppwi == 8:
        return fasten[dtype, 8](ctx, deck, wgsize, iters, warmup)
    if ppwi == 16:
        return fasten[dtype, 16](ctx, deck, wgsize, iters, warmup)
    if ppwi == 32:
        return fasten[dtype, 32](ctx, deck, wgsize, iters, warmup)
    if ppwi == 64:
        return fasten[dtype, 64](ctx, deck, wgsize, iters, warmup)
    if ppwi == 128:
        return fasten[dtype, 128](ctx, deck, wgsize, iters, warmup)
    raise Error("unsupported ppwi: " + String(ppwi))


@fieldwise_init
struct SummaryStats(ImplicitlyCopyable, Movable):
    var min: Float64
    var max: Float64
    var sum: Float64
    var mean: Float64
    var std_dev: Float64


def summarise(ys: List[Float64]) -> SummaryStats:
    var lo = ys[0]
    var hi = ys[0]
    var total = 0.0
    for i in range(len(ys)):
        if ys[i] < lo:
            lo = ys[i]
        if ys[i] > hi:
            hi = ys[i]
        total += ys[i]
    var mean = total / Float64(len(ys))
    var variance = 0.0
    for i in range(len(ys)):
        var d = ys[i] - mean
        variance += d * d
    variance /= Float64(len(ys))
    return SummaryStats(lo, hi, total, mean, sqrt(variance))


@fieldwise_init
struct Result(Copyable, Movable):
    var sample: Sample
    var ms: SummaryStats
    var gflops: Float64
    var ginsts: Float64
    var interactions_per_sec: Float64
    var launch_overhead_ms: Float64
    var vs_reference: Metrics
    var has_differential: Bool
    var vs_deck: Metrics


def evaluate(
    deck: Deck,
    s: Sample,
    reference: List[Float64],
    has_differential: Bool,
    iterations: Int,
    warmup: Int,
) raises -> Result:
    if len(s.kernel_millis) == 0:
        raise Error(
            "Sample size is 0, this implementation did not measure kernel runtime!"
        )

    var without_warmup = List[Float64]()
    for i in range(warmup, len(s.kernel_millis)):
        without_warmup.append(s.kernel_millis[i])

    var elapsed = 0.0
    for i in range(len(without_warmup)):
        elapsed += without_warmup[i]
    var ms = elapsed / Float64(iterations)
    var runtime = ms * 1e-3

    var natlig = deck.natlig()
    var natpro = deck.natpro()
    var nposes = deck.nposes()

    var ops_per_wg = (
        s.ppwi * 27
        + natlig * (2 + s.ppwi * 18 + natpro * (10 + s.ppwi * 30))
        + s.ppwi
    )
    var total_ops = Float64(ops_per_wg) * (Float64(nposes) / Float64(s.ppwi))
    var gflops = (total_ops / runtime) / 1e9

    var total_finsts = 25 * natpro * natlig * nposes
    var ginsts = (Float64(total_finsts) / runtime) / 1e9

    var interactions = nposes * natlig * natpro
    var interactions_per_sec = Float64(interactions) / runtime

    var vs_reference = empty_metrics()
    if has_differential:
        vs_reference = compute_metrics(s.energies, reference, s.precision_bits)
    var vs_deck = compare_to_deck(s.energies, deck.ref_energies)

    var launch_overhead_ms = 0.0
    if len(s.wall_millis) == len(s.kernel_millis) and len(s.kernel_millis) > warmup:
        var acc = 0.0
        for i in range(warmup, len(s.kernel_millis)):
            acc += s.wall_millis[i] - s.kernel_millis[i]
        launch_overhead_ms = acc / Float64(len(s.kernel_millis) - warmup)

    return Result(
        s.copy(),
        summarise(without_warmup),
        gflops,
        ginsts,
        interactions_per_sec,
        launch_overhead_ms,
        vs_reference^,
        has_differential,
        vs_deck^,
    )


def show_human_readable(r: Result, out_rows: Int):
    var p = " "
    var raw = String("")
    for i in range(len(r.sample.kernel_millis)):
        if i > 0:
            raw += ","
        raw += gfmt(r.sample.kernel_millis[i])

    print(p, "- precision_bits:      ", r.sample.precision_bits)
    print(
        p,
        "  param:               { ppwi: ",
        r.sample.ppwi,
        ", wgsize: ",
        r.sample.wgsize,
        " }",
    )
    print(p, "  raw_iterations:      [" + raw + "]")
    print(p, "  context_ms:          " + fixed(r.sample.context_millis, 6))
    print(p, "  sum_ms:              " + fixed(r.ms.sum, 3))
    print(p, "  avg_ms:              " + fixed(r.ms.mean, 3))
    print(p, "  min_ms:              " + fixed(r.ms.min, 3))
    print(p, "  max_ms:              " + fixed(r.ms.max, 3))
    print(p, "  stddev_ms:           " + fixed(r.ms.std_dev, 3))
    print(
        p,
        "  giga_interactions/s: " + fixed(r.interactions_per_sec / 1e9, 3),
    )
    print(p, "  gflop/s:             " + fixed(r.gflops, 3))
    print(p, "  gfinst/s:            " + fixed(r.ginsts, 3))
    print(p, "  launch_overhead_ms:  " + fixed(r.launch_overhead_ms, 3))
    print(p, "  energies:            ")
    var rows = min(out_rows, len(r.sample.energies))
    for i in range(rows):
        print(p, "    - " + fixed(r.sample.energies[i], 6))

    if r.has_differential:
        var m = r.vs_reference.copy()
        print(p, "  vs_fp64_reference:")
        print(p, "    l2_rel_norm:       " + sci(m.l2_rel_norm, 6))
        print(p, "    rms_abs:           " + sci(m.rms_abs, 6))
        print(p, "    mean_signed:       " + sci(m.mean_signed, 6))
        print(
            p,
            "    max_abs:           { value: "
            + sci(m.max_abs, 6)
            + ", index: "
            + String(m.max_abs_idx)
            + " }",
        )
        print(
            p,
            "    max_rel:           { value: "
            + sci(m.max_rel, 6)
            + ", index: "
            + String(m.max_rel_idx)
            + " }",
        )
        print(
            p,
            "    max_ulp:           { value: "
            + String(m.max_ulp)
            + ", index: "
            + String(m.max_ulp_idx)
            + " }",
        )
        print(p, "    compared:          ", m.compared)
        print(p, "    non_finite:        ", m.non_finite)
        print(p, "    ulp_histogram:")
        for i in range(ULP_NBUCKETS):
            if m.ulp_histogram[i] > 0:
                print(
                    p,
                    '      "'
                    + ulp_bucket_label(i)
                    + '": '
                    + String(m.ulp_histogram[i]),
                )
    else:
        print(
            p,
            "  vs_fp64_reference:   ~ # benchmarked precision is the reference",
        )

    var d = r.vs_deck.copy()
    print(
        p,
        "  vs_deck_reference:   # informational; ref_energies.out is low precision",
    )
    print(p, "    l2_rel_norm:       " + sci(d.l2_rel_norm, 6))
    print(
        p,
        "    max_rel:           { value: "
        + sci(d.max_rel, 6)
        + ", index: "
        + String(d.max_rel_idx)
        + " }",
    )
    print(
        p,
        "    beyond_tolerance:  "
        + String(d.beyond_tolerance)
        + "/"
        + String(d.compared)
        + " # tolerance = 0.025%",
    )


def show_csv(r: Result, header: Bool):
    if header:
        print(
            "precision_bits,ppwi,wgsize,sum_ms,avg_ms,min_ms,max_ms,stddev_ms,"
            + "interactions/s,gflops/s,gfinst/s,launch_overhead_ms,"
            + "l2_rel_norm,rms_abs,mean_signed,max_abs,max_rel,max_ulp,deck_l2_rel_norm"
        )
    var line = String(r.sample.precision_bits)
    line += "," + String(r.sample.ppwi)
    line += "," + String(r.sample.wgsize)
    line += "," + fixed(r.ms.sum, 3)
    line += "," + fixed(r.ms.mean, 3)
    line += "," + fixed(r.ms.min, 3)
    line += "," + fixed(r.ms.max, 3)
    line += "," + fixed(r.ms.std_dev, 3)
    line += "," + fixed(r.interactions_per_sec, 3)
    line += "," + fixed(r.gflops, 3)
    line += "," + fixed(r.ginsts, 3)
    line += "," + fixed(r.launch_overhead_ms, 3)
    if r.has_differential:
        var m = r.vs_reference.copy()
        line += "," + sci(m.l2_rel_norm, 6)
        line += "," + sci(m.rms_abs, 6)
        line += "," + sci(m.mean_signed, 6)
        line += "," + sci(m.max_abs, 6)
        line += "," + sci(m.max_rel, 6)
        line += "," + String(m.max_ulp)
    else:
        line += ",,,,,"
    line += "," + sci(r.vs_deck.l2_rel_norm, 6)
    print(line)


def load_deck(deck_dir: String, nposes_arg: Int) raises -> Deck:
    var atoms_l = read_bytes(deck_dir + "/ligand.in")
    var atoms_p = read_bytes(deck_dir + "/protein.in")
    var ffb = read_bytes(deck_dir + "/forcefield.in")
    var posb = read_bytes(deck_dir + "/poses.in")

    var lig_x = List[Float32]()
    var lig_y = List[Float32]()
    var lig_z = List[Float32]()
    var lig_t = List[Int32]()
    for i in range(len(atoms_l) // 16):
        lig_x.append(le_f32(atoms_l, i * 16 + 0))
        lig_y.append(le_f32(atoms_l, i * 16 + 4))
        lig_z.append(le_f32(atoms_l, i * 16 + 8))
        lig_t.append(le_i32(atoms_l, i * 16 + 12))

    var prot_x = List[Float32]()
    var prot_y = List[Float32]()
    var prot_z = List[Float32]()
    var prot_t = List[Int32]()
    for i in range(len(atoms_p) // 16):
        prot_x.append(le_f32(atoms_p, i * 16 + 0))
        prot_y.append(le_f32(atoms_p, i * 16 + 4))
        prot_z.append(le_f32(atoms_p, i * 16 + 8))
        prot_t.append(le_i32(atoms_p, i * 16 + 12))

    var ff_hbtype = List[Int32]()
    var ff_radius = List[Float32]()
    var ff_hphb = List[Float32]()
    var ff_elsc = List[Float32]()
    for i in range(len(ffb) // 16):
        ff_hbtype.append(le_i32(ffb, i * 16 + 0))
        ff_radius.append(le_f32(ffb, i * 16 + 4))
        ff_hphb.append(le_f32(ffb, i * 16 + 8))
        ff_elsc.append(le_f32(ffb, i * 16 + 12))

    var total = len(posb) // 4
    if total % 6 != 0:
        raise Error("Pose size (" + String(total) + ") not divisible my 6!")
    var max_poses = total // 6
    var n = max_poses if nposes_arg == 0 else nposes_arg
    if n > max_poses:
        raise Error(
            "Requested poses ("
            + String(n)
            + ") exceeded max poses ("
            + String(max_poses)
            + ") for deck"
        )
    var poses = List[List[Float32]]()
    for k in range(6):
        var col = List[Float32]()
        for i in range(n):
            col.append(le_f32(posb, (k * max_poses + i) * 4))
        poses.append(col^)

    var ref_energies = List[Float32]()
    with open(deck_dir + "/ref_energies.out", "r") as f:
        var text = f.read()
        for line in text.split("\n"):
            var t = String(line).strip()
            if t.byte_length() > 0:
                ref_energies.append(Float32(atof(t)))
    if n > len(ref_energies):
        raise Error(
            "Size of reference energies ("
            + String(len(ref_energies))
            + ") is less than poses ("
            + String(n)
            + ")"
        )

    return Deck(
        prot_x^, prot_y^, prot_z^, prot_t^,
        lig_x^, lig_y^, lig_z^, lig_t^,
        ff_hbtype^, ff_radius^, ff_hphb^, ff_elsc^,
        poses^, ref_energies^,
    )


def load_golden(path: String) raises -> List[Float64]:
    var out = List[Float64]()
    with open(path, "r") as f:
        var text = f.read()
        for line in text.split("\n"):
            var t = String(line).strip()
            if t.byte_length() > 0:
                out.append(atof(t))
    return out^


def print_help():
    print("")
    print("Usage: bude_mojo [COMMAND|OPTIONS]")
    print("")
    print("Commands:")
    print("  help -h --help       Print this message")
    print("  list -l --list       List available devices")
    print("Options:")
    print("  -i --iter    I       Repeat kernel I times [optional] default=8")
    print("  -n --poses   N       Compute energies for only N poses, 0 for deck max")
    print("                       [optional] default=0")
    print("  -p --ppwi    PPWI    A CSV list of poses per work-item, `all` for everything")
    print("                       [optional] default=1; available=" + PPWIS_STR)
    print("  -w --wgsize  WGSIZE  A CSV list of work-group sizes [optional] default=1")
    print("     --deck    DIR     Use the DIR directory as input deck")
    print("  -r --rows    N       Output first N row(s) of energy values [default=8]")
    print("     --csv             Output results in CSV format [default=false]")
    print("     --golden  PATH    fp64 reference energies, one per line, for the")
    print("                       differential metrics. Mojo cannot compute an fp64")
    print("                       reference itself on NVIDIA (no fp64 sin/cos), so")
    print("                       pass the CUDA build's: bude_cuda64 -o FILE")
    print("")
    print("Precision is not a flag: edit the PRECISION_BITS comptime value at the top of")
    print("file and rebuild, as with the reference's -DPRECISION_BITS.")
    print("")
    print("Not implemented in this port (present in the CUDA reference):")
    print("  -o --out, -d --device")
    print("")


def split_ints(s: String) raises -> List[Int]:
    var out = List[Int]()
    for tok in s.split(","):
        var t = String(tok).strip()
        if t.byte_length() > 0:
            out.append(Int(atol(t)))
    return out^


def main() raises:
    var args = argv()

    var deck_dir = String(DEFAULT_DATA_DIR)
    var iterations = DEFAULT_ITERS
    var warmup = DEFAULT_WARMUP
    var out_rows = DEFAULT_ENERGY_ENTRIES
    var nposes_arg = 0
    var csv = False
    var want_list = False
    var precision_bits = PRECISION_BITS
    var golden_path = String("")
    var wgsizes = List[Int]()
    var ppwis = List[Int]()

    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--iter" or a == "-i":
            i += 1
            iterations = Int(atol(String(args[i])))
        elif a == "--poses" or a == "-n":
            i += 1
            nposes_arg = Int(atol(String(args[i])))
        elif a == "--deck":
            i += 1
            deck_dir = String(args[i])
        elif a == "--rows" or a == "-r":
            i += 1
            out_rows = Int(atol(String(args[i])))
        elif a == "--wgsize" or a == "-w":
            i += 1
            wgsizes = split_ints(String(args[i]))
        elif a == "--ppwi" or a == "-p":
            i += 1
            var s = String(args[i])
            if s == "all":
                ppwis.append(1)
                ppwis.append(2)
                ppwis.append(4)
                ppwis.append(8)
                ppwis.append(16)
                ppwis.append(32)
                ppwis.append(64)
                ppwis.append(128)
            else:
                ppwis = split_ints(s)
                for k in range(len(ppwis)):
                    if not is_supported_ppwi(ppwis[k]):
                        print(
                            "PPWI",
                            ppwis[k],
                            "is not a supported value, should be one of `"
                            + PPWIS_STR
                            + "`",
                        )
                        return
        elif a == "--golden":
            i += 1
            golden_path = String(args[i])
        elif a == "--csv":
            csv = True
        elif a == "list" or a == "--list" or a == "-l":
            want_list = True
        elif a == "help" or a == "--help" or a == "-h":
            print_help()
            return
        else:
            print("Unrecognized argument '" + a + "' (try '--help')")
            return
        i += 1

    if len(ppwis) == 0:
        ppwis.append(1)
    if len(wgsizes) == 0:
        wgsizes.append(1)
    if precision_bits != 32 and precision_bits != 64:
        raise Error("precision must be 32 or 64")

    var ctx = DeviceContext()

    if want_list:
        print("devices:" if not csv else "index,name")
        print('  0: "' + ctx.name() + '"')
        return

    if not csv:
        print("miniBUDE:  mojo")
        print("device:    " + ctx.name())
        print("precision: ", precision_bits, "# PRECISION_BITS comptime value in source")

    var deck = load_deck(deck_dir, nposes_arg)
    var n = deck.nposes()

    var golden = List[Float64]()
    var golden_ready = False
    if golden_path.byte_length() > 0:
        golden = load_golden(golden_path)
        golden_ready = True
        if len(golden) < n:
            raise Error(
                "golden file has "
                + String(len(golden))
                + " energies, fewer than the "
                + String(n)
                + " poses being computed"
            )
    elif not csv:
        print("# no --golden file: vs_fp64_reference will be empty. Pass the")
        print("#   CUDA build's fp64 energies (bude_cuda64 -o FILE) to fill it.")

    var results = List[Result]()

    for pi in range(len(ppwis)):
        var ppwi = ppwis[pi]
        for wi in range(len(wgsizes)):
            var wgsize = wgsizes[wi]

            if n < ppwi * wgsize:
                print(
                    " # WARNING: pose count",
                    n,
                    "<= (",
                    wgsize,
                    "(wgsize) *",
                    ppwi,
                    "(ppwi)), skipping",
                )
                continue
            if n % (ppwi * wgsize) != 0:
                print(
                    " # WARNING: pose count",
                    n,
                    "% (",
                    wgsize,
                    "(wgsize) *",
                    ppwi,
                    "(ppwi)) != 0, skipping",
                )
                continue

            var sample = fasten_dispatch[FP](
                ctx, deck, wgsize, ppwi, iterations, warmup
            )

            results.append(
                evaluate(
                    deck, sample, golden, golden_ready, iterations, warmup
                )
            )

    if len(results) == 0:
        print("# no runnable configurations")
        return

    if not csv:
        print("results:")
    for k in range(len(results)):
        if csv:
            show_csv(results[k], k == 0)
        else:
            show_human_readable(results[k], out_rows)

    var best = 0
    for k in range(len(results)):
        if results[k].ms.sum < results[best].ms.sum:
            best = k
    print(
        ("# " if csv else "")
        + "best: { min_ms: "
        + fixed(results[best].ms.min, 3)
        + ", max_ms: "
        + fixed(results[best].ms.max, 3)
        + ", sum_ms: "
        + fixed(results[best].ms.sum, 3)
        + ", avg_ms: "
        + fixed(results[best].ms.mean, 3)
        + ", ppwi: "
        + String(results[best].sample.ppwi)
        + ", wgsize: "
        + String(results[best].sample.wgsize)
        + " }"
    )
