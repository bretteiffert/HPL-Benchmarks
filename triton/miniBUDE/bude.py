"""miniBUDE buffers, timed launches, the float64 reference pass and the
correctness metrics. Nothing here prints; main.py fills report.Report."""

import math
import time

import numpy as np
import torch
import triton
import triton.language as tl

from kernels import fasten_main

REPORT_TOLERANCE_PCT = 0.025

NUM_STAGES_DEFAULT = 3

PPWIS = (1, 2, 4, 8, 16, 32, 64, 128)

try:
    from triton.runtime.errors import OutOfResources
except Exception:  # pragma: no cover
    class OutOfResources(Exception):
        pass


class ConfigUnavailable(RuntimeError):
    pass


class TimerError(Exception):
    pass


def resolve_warp_size():
    """Returns (warp_size, how). Call after the device is selected."""
    from triton.runtime import driver

    target = None
    try:
        target = driver.active.get_current_target()
    except Exception:
        pass
    backend = str(getattr(target, "backend", "unknown"))
    ws = getattr(target, "warp_size", None)
    if ws:
        return int(ws), "Triton target"
    try:
        props = driver.active.utils.get_device_properties(driver.active.get_current_device())
        ws = props.get("warpSize")
        if ws:
            return int(ws), "device properties"
    except Exception:
        pass
    return (64 if backend == "hip" else 32), "guessed"


# -- device run ------------------------------------------------------------------

class Buffers:
    """Deck arrays on the device in one precision."""

    def __init__(self, deck, dtype, device):
        def dev(a):
            return torch.from_numpy(np.ascontiguousarray(a)).to(device)

        start = time.perf_counter()
        prot, lig, ff = deck.protein, deck.ligand, deck.forcefield
        self.prot = [dev(prot["x"].astype(dtype)), dev(prot["y"].astype(dtype)),
                     dev(prot["z"].astype(dtype)), dev(prot["type"].astype(np.int32))]
        self.lig = [dev(lig["x"].astype(dtype)), dev(lig["y"].astype(dtype)),
                    dev(lig["z"].astype(dtype)), dev(lig["type"].astype(np.int32))]
        self.ff = [dev(ff["hbtype"].astype(np.int32)), dev(ff["radius"].astype(dtype)),
                   dev(ff["hphb"].astype(dtype)), dev(ff["elsc"].astype(dtype))]
        self.t = [dev(deck.poses[i].astype(dtype)) for i in range(6)]
        self.out = torch.empty(deck.nposes, dtype=torch.float32 if dtype == np.float32
                               else torch.float64, device=device)
        torch.cuda.synchronize()
        self.context_s = time.perf_counter() - start
        self.dtype = dtype
        self.natlig, self.natpro, self.nposes = deck.natlig, deck.natpro, deck.nposes

    def launch(self, ppwi, wgsize, warp_size, num_stages):
        tl_dtype = tl.float32 if self.dtype == np.float32 else tl.float64
        grid = (triton.cdiv(triton.cdiv(self.nposes, ppwi), wgsize),)
        fasten_main[grid](
            self.out,
            *self.prot, *self.lig, *self.t, *self.ff,
            self.natlig, self.natpro, self.nposes,
            BLOCK=ppwi * wgsize,
            DTYPE=tl_dtype,
            MAXVAL=float(np.finfo(self.dtype).max),
            num_warps=wgsize // warp_size,
            num_stages=num_stages,
        )

    def energies(self):
        return self.out.detach().cpu().numpy().astype(np.float64)


class Timer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)

    def time(self, fn, what):
        """Returns (kernel seconds from device events, host wall seconds)."""
        wall0 = time.perf_counter()
        self.start.record()
        fn()
        self.stop.record()
        self.stop.synchronize()
        wall = time.perf_counter() - wall0
        ms = self.start.elapsed_time(self.stop)
        if not ms or ms <= 0.0:
            raise TimerError("device timer returned no measurement for %s" % what)
        return ms * 1.0e-3, wall


def reference_pass(deck, ppwi, wgsize, warp_size, device):
    """ONE untimed float64 launch, the reference for float32 runs."""
    buf = Buffers(deck, np.float64, device)
    last = None
    for stages in (NUM_STAGES_DEFAULT,):
        try:
            buf.launch(ppwi, wgsize, warp_size, stages)
            torch.cuda.synchronize()
            return buf.energies()
        except OutOfResources as e:
            last = e
    raise ConfigUnavailable("float64 reference pass exceeded device resources (%s)"
                            % str(last).replace("\n", " ")[:160])


class Sample:
    def __init__(self, ppwi, wgsize, num_warps):
        self.ppwi = ppwi
        self.wgsize = wgsize
        self.num_warps = num_warps
        self.num_stages = 0
        self.context_s = 0.0
        self.times = []        # timed iterations only, seconds
        self.walls = []        # host wall time of the same launches
        self.energies = None
        self.energy = None     # energy.EnergyResult
        self.stats = None
        self.gflops = self.gfinsts = self.ginteractions = 0.0
        self.launch_overhead_s = 0.0
        self.vs_ref = None     # Metrics, float32 runs only
        self.vs_deck = None


def run_config(buf, ppwi, wgsize, warp_size, warmup, iterations, timer, meter_factory):
    s = Sample(ppwi, wgsize, wgsize // warp_size)
    s.context_s = buf.context_s
    s.num_stages = NUM_STAGES_DEFAULT
    run = lambda: buf.launch(ppwi, wgsize, warp_size, s.num_stages)  # noqa: E731
    for i in range(warmup):
        timer.time(run, "warm-up iteration %d" % (i + 1))
    meter = meter_factory() if meter_factory else None
    if meter is not None:
        meter.start()
    for i in range(iterations):
        k, w = timer.time(run, "timed iteration %d (ppwi %d, wgsize %d)" % (i + 1, ppwi, wgsize))
        s.times.append(k)
        s.walls.append(w)
    if meter is not None:
        s.energy = meter.stop(iterations)
    s.energies = buf.energies()
    return s


# -- statistics and metrics (unchanged from the original) --------------------------

class SummaryStats:
    def __init__(self, ys):
        self.min = min(ys)
        self.max = max(ys)
        self.sum = sum(ys)
        self.mean = self.sum / len(ys)
        self.variance = sum((y - self.mean) ** 2 for y in ys) / len(ys)
        self.stdDev = math.sqrt(self.variance)


def evaluate(deck, s, reference):
    s.stats = SummaryStats(s.times)
    runtime = s.stats.mean
    ops_per_wg = (
        s.ppwi * 27
        + deck.natlig * (2 + s.ppwi * 18 + deck.natpro * (10 + s.ppwi * 30))
        + s.ppwi
    )
    total_ops = float(ops_per_wg) * (float(deck.nposes) / float(s.ppwi))
    s.gflops = (total_ops / runtime) / 1e9
    total_finsts = 25 * deck.natpro * deck.natlig * deck.nposes
    s.gfinsts = (float(total_finsts) / runtime) / 1e9
    s.ginteractions = float(deck.nposes * deck.natlig * deck.natpro) / runtime / 1e9
    s.launch_overhead_s = sum(w - k for w, k in zip(s.walls, s.times)) / len(s.times)
    s.vs_ref = compute_metrics(s.energies, reference, 32) if reference is not None else None
    s.vs_deck = compare_to_deck(s.energies, deck.ref_energies)


ULP_BUCKETS = np.array([1, 2, 4, 8, 16, 64, 256, 1024, 4096], dtype=np.int64)
ULP_NBUCKETS = len(ULP_BUCKETS) + 1


def ulp_bucket_label(i):
    if i == 0:
        return "0"
    if i == ULP_NBUCKETS - 1:
        return ">=" + str(int(ULP_BUCKETS[-1]))
    lo = int(ULP_BUCKETS[i - 1])
    hi = int(ULP_BUCKETS[i]) - 1
    return str(lo) if lo == hi else "%d-%d" % (lo, hi)


def _ordered(arr):
    if arr.dtype == np.float32:
        i = arr.view(np.int32).astype(np.int64)
        return np.where(i < 0, np.int64(0x80000000) - (i & 0xFFFFFFFF), i)
    i = arr.view(np.int64)
    with np.errstate(over="ignore"):
        return np.where(i < 0, np.int64(np.iinfo(np.int64).min) - i, i)


def ulp_distance_array(a, b):
    out = np.zeros(a.shape, dtype=np.int64)
    with np.errstate(over="ignore", invalid="ignore"):
        d = _ordered(a) - _ordered(b)
        out[:] = np.abs(d)
    out[a == b] = 0
    bad = np.isnan(a) | np.isnan(b) | np.isinf(a) | np.isinf(b)
    bad &= a != b
    out[bad] = np.iinfo(np.int64).max
    return out


class Metrics:
    def __init__(self):
        self.l2RelNorm = 0.0
        self.rmsAbs = 0.0
        self.meanSigned = 0.0
        self.maxAbs = 0.0
        self.maxAbsIdx = 0
        self.maxRel = 0.0
        self.maxRelIdx = 0
        self.maxUlp = 0
        self.maxUlpIdx = 0
        self.ulpHistogram = [0] * ULP_NBUCKETS
        self.compared = 0
        self.nonFinite = 0
        self.beyondTolerance = 0


class Kahan:
    __slots__ = ("sum", "c")

    def __init__(self):
        self.sum = 0.0
        self.c = 0.0

    def add(self, x):
        y = x - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t


def compute_metrics(measured, reference, precision_bits):
    m = Metrics()
    n = min(len(measured), len(reference))
    x_all = np.asarray(measured[:n], dtype=np.float64)
    r_all = np.asarray(reference[:n], dtype=np.float64)

    if precision_bits == 32:
        ulps = ulp_distance_array(x_all.astype(np.float32), r_all.astype(np.float32))
    else:
        ulps = ulp_distance_array(x_all, r_all)

    buckets = np.searchsorted(ULP_BUCKETS, ulps, side="right")

    errSq, refSq, signedErr = Kahan(), Kahan(), Kahan()

    for i in range(n):
        x = float(x_all[i])
        r = float(r_all[i])

        if not math.isfinite(x) or not math.isfinite(r):
            m.nonFinite += 1
            continue

        err = x - r
        absErr = abs(err)

        errSq.add(err * err)
        refSq.add(r * r)
        signedErr.add(err)

        if absErr > m.maxAbs:
            m.maxAbs = absErr
            m.maxAbsIdx = i

        denom = abs(r)
        if denom > 0.0:
            rel = absErr / denom
            if rel > m.maxRel:
                m.maxRel = rel
                m.maxRelIdx = i
            if rel * 100.0 > REPORT_TOLERANCE_PCT:
                m.beyondTolerance += 1

        u = int(ulps[i])
        if u > m.maxUlp:
            m.maxUlp = u
            m.maxUlpIdx = i
        m.ulpHistogram[int(buckets[i])] += 1

        m.compared += 1

    if m.compared > 0:
        m.rmsAbs = math.sqrt(errSq.sum / m.compared)
        m.meanSigned = signedErr.sum / m.compared
        m.l2RelNorm = (
            math.sqrt(errSq.sum) / math.sqrt(refSq.sum) if refSq.sum > 0.0 else 0.0
        )

    return m


def compare_to_deck(measured, ref_energies):
    n = min(len(ref_energies), len(measured))
    return compute_metrics(measured, np.asarray(ref_energies[:n], dtype=np.float64), 64)
