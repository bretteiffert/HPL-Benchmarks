# miniBUDE problem: device buffers, timed launches, performance figures and the
# accuracy metrics (vs a float64 reference and vs the deck's energies).
#
# Metric, ULP and FLOP-count arithmetic are unchanged from the original
# miniBUDE.mojo. Changes:
#   * times are kept in seconds; warm-up launches are run but not recorded;
#   * board energy is measured over the timed launches of each configuration;
#   * a launch whose device timer returns nothing is an error, not zero;
#   * no printing: results go to the caller, which fills the report.

from max.gpu.host import DeviceBuffer, DeviceContext
from std.collections import Array, List, Optional
from std.math import isfinite, sqrt
from std.time import perf_counter_ns

from deck import Deck
from energy import BoardEnergy
from kernels import fasten_main

comptime REPORT_TOLERANCE_PCT = 0.025


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
    var ulp_histogram: Array[Int, ULP_NBUCKETS]
    var compared: Int
    var non_finite: Int
    var beyond_tolerance: Int


def empty_metrics() -> Metrics:
    var h = Array[Int, ULP_NBUCKETS](fill=0)
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



# ===----------------------------------------------------------------------=== #
# One (ppwi, wgsize) configuration
# ===----------------------------------------------------------------------=== #


struct Sample(Copyable):
    var ppwi: Int
    var wgsize: Int
    var precision_bits: Int
    var energies: List[Float64]
    var kernel_s: List[Float64]  # timed launches only
    var wall_s: List[Float64]  # host wall time around each timed launch
    var context_s: Float64
    var energy_measured: Bool
    var joules: Float64
    var energy_time_s: Float64
    var avg_power_w: Float64
    var samples: Int

    def __init__(out self, ppwi: Int, wgsize: Int, precision_bits: Int):
        self.ppwi = ppwi
        self.wgsize = wgsize
        self.precision_bits = precision_bits
        self.energies = List[Float64]()
        self.kernel_s = List[Float64]()
        self.wall_s = List[Float64]()
        self.context_s = 0.0
        self.energy_measured = False
        self.joules = 0.0
        self.energy_time_s = 0.0
        self.avg_power_w = 0.0
        self.samples = 0


def fasten[
    dtype: DType, ppwi: Int
](
    ctx: DeviceContext,
    deck: Deck,
    wgsize: Int,
    iters: Int,
    warmup: Int,
    measure_energy: Bool,
    device_index: Int,
) raises -> Sample:
    var n = deck.nposes()
    var natpro = deck.natpro()
    var natlig = deck.natlig()
    var ntypes = deck.ntypes()

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
    var context_s = Float64(perf_counter_ns() - context_start) * 1e-9

    var blocks = ((n + ppwi - 1) // ppwi + wgsize - 1) // wgsize

    # The GPU ABI uses fixed-width integer arguments.
    var natlig_dev = Int32(natlig)
    var natpro_dev = Int32(natpro)
    var n_dev = Int32(n)

    @always_inline
    def launch(c: DeviceContext) raises {d_out, d_prot_x, d_prot_y, d_prot_z, d_prot_t, d_lig_x, d_lig_y, d_lig_z, d_lig_t, d_t0, d_t1, d_t2, d_t3, d_t4, d_t5, d_ff_hb, d_ff_rad, d_ff_hphb, d_ff_elsc, natlig_dev, natpro_dev, n_dev, blocks, wgsize}:
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

    var s = Sample(ppwi, wgsize, 64 if dtype == DType.float64 else 32)
    s.context_s = context_s

    for _ in range(warmup):
        _ = ctx.execution_time(launch, 1)

    var prof = Optional[BoardEnergy](None)
    if measure_energy:
        try:
            prof = Optional[BoardEnergy](BoardEnergy(device_index))
        except e:
            raise Error(
                "energy profiler failed to start: "
                + String(e)
                + " (try '--no-energy')"
            )
    var t_start = Float64(0)
    if prof:
        ctx.synchronize()
        prof.value().start()
        t_start = Float64(perf_counter_ns()) * 1e-9

    for _ in range(iters):
        var wall_start = perf_counter_ns()
        var ns = ctx.execution_time(launch, 1)
        var wall_ns = perf_counter_ns() - wall_start
        if ns <= 0:
            raise Error("device timer returned no measurement for fasten_main")
        s.kernel_s.append(Float64(ns) * 1e-9)
        s.wall_s.append(Float64(wall_ns) * 1e-9)

    if prof:
        ctx.synchronize()
        var reading = prof.value().stop()
        var host_s = Float64(perf_counter_ns()) * 1e-9 - t_start
        prof.value().close()
        s.energy_measured = True
        s.joules = reading.joules
        s.avg_power_w = reading.avg_power_w
        s.samples = reading.samples
        s.energy_time_s = reading.elapsed_s if reading.elapsed_s > 0.0 else host_s

    var host_out = List[Scalar[dtype]]()
    for _ in range(n):
        host_out.append(Scalar[dtype](0))
    ctx.enqueue_copy(host_out.unsafe_ptr(), d_out)
    ctx.synchronize()

    for i in range(n):
        s.energies.append(Float64(host_out[i]))
    return s^


def fasten_dispatch[
    dtype: DType
](
    ctx: DeviceContext,
    deck: Deck,
    wgsize: Int,
    ppwi: Int,
    iters: Int,
    warmup: Int,
    measure_energy: Bool,
    device_index: Int,
) raises -> Sample:
    if ppwi == 1:
        return fasten[dtype, 1](ctx, deck, wgsize, iters, warmup, measure_energy, device_index)
    if ppwi == 2:
        return fasten[dtype, 2](ctx, deck, wgsize, iters, warmup, measure_energy, device_index)
    if ppwi == 4:
        return fasten[dtype, 4](ctx, deck, wgsize, iters, warmup, measure_energy, device_index)
    if ppwi == 8:
        return fasten[dtype, 8](ctx, deck, wgsize, iters, warmup, measure_energy, device_index)
    if ppwi == 16:
        return fasten[dtype, 16](ctx, deck, wgsize, iters, warmup, measure_energy, device_index)
    if ppwi == 32:
        return fasten[dtype, 32](ctx, deck, wgsize, iters, warmup, measure_energy, device_index)
    if ppwi == 64:
        return fasten[dtype, 64](ctx, deck, wgsize, iters, warmup, measure_energy, device_index)
    if ppwi == 128:
        return fasten[dtype, 128](ctx, deck, wgsize, iters, warmup, measure_energy, device_index)
    raise Error("unsupported ppwi: " + String(ppwi))


# ===----------------------------------------------------------------------=== #
# Performance figures
# ===----------------------------------------------------------------------=== #


struct SummaryStats(ImplicitlyCopyable, Movable):
    var min: Float64
    var max: Float64
    var sum: Float64
    var mean: Float64
    var std_dev: Float64

    def __init__(
        out self, min: Float64, max: Float64, sum: Float64, mean: Float64, std_dev: Float64
    ):
        self.min = min
        self.max = max
        self.sum = sum
        self.mean = mean
        self.std_dev = std_dev


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


struct Result(Copyable):
    var sample: Sample
    var stats: SummaryStats
    var gflops: Float64
    var ginsts: Float64
    var interactions_per_sec: Float64
    var launch_overhead_s: Float64
    var vs_reference: Metrics
    var has_reference: Bool
    var vs_deck: Metrics

    def __init__(
        out self,
        var sample: Sample,
        stats: SummaryStats,
        gflops: Float64,
        ginsts: Float64,
        interactions_per_sec: Float64,
        launch_overhead_s: Float64,
        var vs_reference: Metrics,
        has_reference: Bool,
        var vs_deck: Metrics,
    ):
        self.sample = sample^
        self.stats = stats
        self.gflops = gflops
        self.ginsts = ginsts
        self.interactions_per_sec = interactions_per_sec
        self.launch_overhead_s = launch_overhead_s
        self.vs_reference = vs_reference^
        self.has_reference = has_reference
        self.vs_deck = vs_deck^


def evaluate(
    deck: Deck,
    s: Sample,
    reference: List[Float64],
    has_reference: Bool,
) raises -> Result:
    if len(s.kernel_s) == 0:
        raise Error("no timed iterations recorded")

    var stats = summarise(s.kernel_s)
    var runtime = stats.mean  # seconds per launch, as the original's ms * 1e-3

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
    if has_reference:
        vs_reference = compute_metrics(s.energies, reference, s.precision_bits)
    var vs_deck = compare_to_deck(s.energies, deck.ref_energies)

    var acc = 0.0
    for i in range(len(s.kernel_s)):
        acc += s.wall_s[i] - s.kernel_s[i]
    var launch_overhead_s = acc / Float64(len(s.kernel_s))

    return Result(
        s.copy(),
        stats,
        gflops,
        ginsts,
        interactions_per_sec,
        launch_overhead_s,
        vs_reference^,
        has_reference,
        vs_deck^,
    )
