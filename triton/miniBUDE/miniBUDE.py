#!/usr/bin/env python3
# Triton miniBUDE benchmark that reads the reference binary decks, reports numerical metrics, and times CUDA or ROCm launches.
# PRECISION_BITS defaults to 32, while an fp64 differential pass is run when needed; no environment variable selects precision.
# Work-group size determines num_warps from the resolved target warp width, and num_stages is optionally tuned before measured iterations.

import math
import os
import struct
import sys

import numpy as np
import torch
import triton
import triton.language as tl

DEFAULT_ITERS = 8
DEFAULT_WARMUP = 2
DEFAULT_ENERGY_ENTRIES = 8
DEFAULT_DATA_DIR = "../../data/miniBUDE/bm1"

REPORT_TOLERANCE_PCT = 0.025

PRECISION_BITS = 32

AUTOTUNE_NUM_STAGES = True
NUM_STAGES_CANDIDATES = (1, 2, 3, 4)
NUM_STAGES_DEFAULT = 3
NUM_STAGES_TRIALS = 3

HBTYPE_F = tl.constexpr(70)
HBTYPE_E = tl.constexpr(69)

PPWIS = (1, 2, 4, 8, 16, 32, 64, 128)


def resolve_warp_size():
    from triton.runtime import driver

    target = None
    try:
        target = driver.active.get_current_target()
    except Exception:
        pass

    backend = str(getattr(target, "backend", "unknown"))
    arch = getattr(target, "arch", "unknown")

    ws = getattr(target, "warp_size", None)
    if ws:
        return int(ws), backend, arch, "triton target"

    try:
        props = driver.active.utils.get_device_properties(
            driver.active.get_current_device()
        )
        ws = props.get("warpSize")
        if ws:
            return int(ws), backend, arch, "device properties"
    except Exception:
        pass

    guess = 64 if backend == "hip" else 32
    return guess, backend, arch, "GUESSED from backend"


try:
    from triton.runtime.errors import OutOfResources as _OutOfResources
except Exception:  # pragma: no cover

    class _OutOfResources(Exception):
        pass


class ConfigUnavailable(RuntimeError):
    pass


@triton.jit
def fasten_main(
    etotals_ptr,
    prot_x, prot_y, prot_z, prot_t,
    lig_x, lig_y, lig_z, lig_t,
    t0, t1, t2, t3, t4, t5,
    ff_hbtype, ff_radius, ff_hphb, ff_elsc,
    natlig, natpro, num_transforms,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
    MAXVAL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_transforms

    a0 = tl.load(t0 + offs, mask=mask, other=0.0)
    a1 = tl.load(t1 + offs, mask=mask, other=0.0)
    a2 = tl.load(t2 + offs, mask=mask, other=0.0)

    sx = tl.sin(a0)
    cx = tl.cos(a0)
    sy = tl.sin(a1)
    cy = tl.cos(a1)
    sz = tl.sin(a2)
    cz = tl.cos(a2)

    m00 = cy * cz
    m01 = sx * sy * cz - cx * sz
    m02 = cx * sy * cz + sx * sz
    m03 = tl.load(t3 + offs, mask=mask, other=0.0)
    m10 = cy * sz
    m11 = sx * sy * sz + cx * cz
    m12 = cx * sy * sz - sx * cz
    m13 = tl.load(t4 + offs, mask=mask, other=0.0)
    m20 = -sy
    m21 = sx * cy
    m22 = cx * cy
    m23 = tl.load(t5 + offs, mask=mask, other=0.0)

    etot = tl.zeros([BLOCK], dtype=DTYPE)

    for il in range(natlig):
        lx = tl.load(lig_x + il)
        ly = tl.load(lig_y + il)
        lz = tl.load(lig_z + il)
        ltype = tl.load(lig_t + il)

        l_radius = tl.load(ff_radius + ltype)
        l_hphb = tl.load(ff_hphb + ltype)
        l_elsc = tl.load(ff_elsc + ltype)
        l_hbtype = tl.load(ff_hbtype + ltype)

        lhphb_ltz = l_hphb < 0.0
        lhphb_gtz = l_hphb > 0.0

        lpos_x = m03 + lx * m00 + ly * m01 + lz * m02
        lpos_y = m13 + lx * m10 + ly * m11 + lz * m12
        lpos_z = m23 + lx * m20 + ly * m21 + lz * m22

        for ip in range(natpro):
            px = tl.load(prot_x + ip)
            py = tl.load(prot_y + ip)
            pz = tl.load(prot_z + ip)
            ptype = tl.load(prot_t + ip)

            p_radius = tl.load(ff_radius + ptype)
            p_hphb = tl.load(ff_hphb + ptype)
            p_elsc = tl.load(ff_elsc + ptype)
            p_hbtype = tl.load(ff_hbtype + ptype)

            radij = p_radius + l_radius
            r_radij = 1.0 / radij

            both_f = (p_hbtype == HBTYPE_F) & (l_hbtype == HBTYPE_F)
            elcdst = tl.where(both_f, 4.0, 2.0).to(DTYPE)
            elcdst1 = tl.where(both_f, 0.25, 0.5).to(DTYPE)
            type_E = (p_hbtype == HBTYPE_E) | (l_hbtype == HBTYPE_E)

            phphb_ltz = p_hphb < 0.0
            phphb_gtz = p_hphb > 0.0
            phphb_nz = p_hphb != 0.0
            p_hphb_s = p_hphb * tl.where(phphb_ltz & lhphb_gtz, -1.0, 1.0).to(DTYPE)
            l_hphb_s = l_hphb * tl.where(phphb_gtz & lhphb_ltz, -1.0, 1.0).to(DTYPE)

            distdslv = tl.where(
                phphb_ltz,
                tl.where(lhphb_ltz, 5.5, 1.0),
                tl.where(lhphb_ltz, 1.0, -MAXVAL),
            ).to(DTYPE)
            r_distdslv = 1.0 / distdslv

            chrg_init = l_elsc * p_elsc
            dslv_init = p_hphb_s + l_hphb_s

            x = lpos_x - px
            y = lpos_y - py
            z = lpos_z - pz
            distij = tl.sqrt(x * x + y * y + z * z)

            distbb = distij - radij
            zone1 = distbb < 0.0

            etot += (1.0 - (distij * r_radij)) * tl.where(zone1, 2.0 * 38.0, 0.0)

            chrg_e = chrg_init * (
                tl.where(zone1, 1.0, 1.0 - distbb * elcdst1)
                * tl.where(distbb < elcdst, 1.0, 0.0)
            )
            neg_chrg_e = -tl.abs(chrg_e)
            chrg_e = tl.where(type_E, neg_chrg_e, chrg_e)
            etot += chrg_e * 45.0

            coeff = 1.0 - (distbb * r_distdslv)
            dslv_e = dslv_init * tl.where((distbb < distdslv) & phphb_nz, 1.0, 0.0)
            dslv_e *= tl.where(zone1, 1.0, coeff)
            etot += dslv_e

    tl.store(etotals_ptr + offs, etot * 0.5, mask=mask)


DISK_ATOM = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("type", "<i4")]
)
DISK_FFPARAMS = np.dtype(
    [("hbtype", "<i4"), ("radius", "<f4"), ("hphb", "<f4"), ("elsc", "<f4")]
)


def read_nstruct(dtype, path):
    if not os.path.isfile(path):
        raise ValueError("Bad file: " + path)
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // dtype.itemsize
    return np.frombuffer(raw[: n * dtype.itemsize], dtype=dtype).copy()


class Params:
    def __init__(self):
        self.protein = None
        self.ligand = None
        self.forcefield = None
        self.poses = [None] * 6
        self.refEnergies = np.zeros(0, dtype=np.float32)
        self.maxPoses = 0
        self.iterations = DEFAULT_ITERS
        self.warmupIterations = DEFAULT_WARMUP
        self.outRows = DEFAULT_ENERGY_ENTRIES
        self.deckDir = DEFAULT_DATA_DIR
        self.output = ""
        self.deviceSelector = ""
        self.csv = False
        self.list = False

    def total_iterations(self):
        return self.iterations + self.warmupIterations

    def natpro(self):
        return len(self.protein)

    def natlig(self):
        return len(self.ligand)

    def ntypes(self):
        return len(self.forcefield)

    def nposes(self):
        return len(self.poses[0])


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


class Sample:
    def __init__(self, ppwi, wgsize, precision_bits, nposes):
        self.ppwi = ppwi
        self.wgsize = wgsize
        self.precisionBits = precision_bits
        self.energies = np.zeros(nposes, dtype=np.float64)
        self.kernelMillis = []
        self.wallMillis = []
        self.contextMillis = 0.0
        self.numWarps = 0
        self.numStages = 0


def _host_time_ms():
    import time

    return time.perf_counter() * 1e3


def fasten(dtype, p, wgsize, ppwi, warp_size, device):
    torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
    tl_dtype = tl.float32 if dtype == np.float32 else tl.float64

    n = p.nposes()
    block = ppwi * wgsize
    num_warps = wgsize // warp_size

    prot = p.protein
    lig = p.ligand
    ff = p.forcefield

    def dev(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device)

    context_start = _host_time_ms()
    d_prot_x = dev(prot["x"].astype(dtype))
    d_prot_y = dev(prot["y"].astype(dtype))
    d_prot_z = dev(prot["z"].astype(dtype))
    d_prot_t = dev(prot["type"].astype(np.int32))
    d_lig_x = dev(lig["x"].astype(dtype))
    d_lig_y = dev(lig["y"].astype(dtype))
    d_lig_z = dev(lig["z"].astype(dtype))
    d_lig_t = dev(lig["type"].astype(np.int32))
    d_ff_hb = dev(ff["hbtype"].astype(np.int32))
    d_ff_rad = dev(ff["radius"].astype(dtype))
    d_ff_hphb = dev(ff["hphb"].astype(dtype))
    d_ff_elsc = dev(ff["elsc"].astype(dtype))
    d_t = [dev(p.poses[i].astype(dtype)) for i in range(6)]
    d_out = torch.empty(n, dtype=torch_dtype, device=device)
    torch.cuda.synchronize()
    context_end = _host_time_ms()

    sample = Sample(ppwi, wgsize, 32 if dtype == np.float32 else 64, n)
    sample.contextMillis = context_end - context_start
    sample.numWarps = num_warps

    grid = (triton.cdiv(triton.cdiv(n, ppwi), wgsize),)
    maxval = float(np.finfo(dtype).max)

    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)

    def launch(num_stages):
        fasten_main[grid](
            d_out,
            d_prot_x, d_prot_y, d_prot_z, d_prot_t,
            d_lig_x, d_lig_y, d_lig_z, d_lig_t,
            d_t[0], d_t[1], d_t[2], d_t[3], d_t[4], d_t[5],
            d_ff_hb, d_ff_rad, d_ff_hphb, d_ff_elsc,
            p.natlig(), p.natpro(), n,
            BLOCK=block,
            DTYPE=tl_dtype,
            MAXVAL=maxval,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def time_once(num_stages):
        start_evt.record()
        launch(num_stages)
        stop_evt.record()
        stop_evt.synchronize()
        return float(start_evt.elapsed_time(stop_evt))

    num_stages = NUM_STAGES_DEFAULT
    if AUTOTUNE_NUM_STAGES:
        best = None
        first_error = None
        for candidate in NUM_STAGES_CANDIDATES:
            try:
                time_once(candidate)
                t = min(time_once(candidate) for _ in range(NUM_STAGES_TRIALS))
            except _OutOfResources:
                continue
            except Exception as e:
                if first_error is None:
                    first_error = e
                continue
            if best is None or t < best[1]:
                best = (candidate, t)
        if best is None:
            if first_error is not None:
                raise first_error
            raise ConfigUnavailable(
                "no num_stages candidate fit for ppwi=%d wgsize=%d (all "
                "candidates exceeded device resources)" % (ppwi, wgsize)
            )
        num_stages = best[0]
    else:
        time_once(num_stages)
    sample.numStages = num_stages

    for _ in range(p.total_iterations()):
        wall_start = _host_time_ms()
        ms = time_once(num_stages)
        wall_end = _host_time_ms()

        sample.kernelMillis.append(ms)
        sample.wallMillis.append(wall_end - wall_start)

    sample.energies[:] = d_out.detach().cpu().numpy().astype(np.float64)
    del d_out
    return sample


class SummaryStats:
    def __init__(self, ys):
        self.min = min(ys)
        self.max = max(ys)
        self.sum = sum(ys)
        self.mean = self.sum / len(ys)
        self.variance = sum((y - self.mean) ** 2 for y in ys) / len(ys)
        self.stdDev = math.sqrt(self.variance)


class Result:
    pass


def evaluate(p, s, reference):
    if not s.kernelMillis:
        raise RuntimeError(
            "Sample size is 0, this implementation did not measure kernel runtime!"
        )

    ms_without_warmup = s.kernelMillis[p.warmupIterations:]
    elapsed = sum(ms_without_warmup)
    ms = elapsed / p.iterations
    runtime = ms * 1e-3

    ops_per_wg = (
        s.ppwi * 27
        + p.natlig() * (2 + s.ppwi * 18 + p.natpro() * (10 + s.ppwi * 30))
        + s.ppwi
    )
    total_ops = float(ops_per_wg) * (float(p.nposes()) / float(s.ppwi))
    gflops = (total_ops / runtime) / 1e9

    total_finsts = 25 * p.natpro() * p.natlig() * p.nposes()
    ginsts = (float(total_finsts) / runtime) / 1e9

    interactions = p.nposes() * p.natlig() * p.natpro()
    interactions_per_sec = float(interactions) / runtime

    r = Result()
    r.sample = s
    r.ms = SummaryStats(ms_without_warmup)
    r.gflops = gflops
    r.ginsts = ginsts
    r.interactionsPerSec = interactions_per_sec
    r.hasDifferential = reference is not None
    r.vsReference = (
        compute_metrics(s.energies, reference, s.precisionBits)
        if r.hasDifferential
        else Metrics()
    )
    r.vsDeck = compare_to_deck(s.energies, p.refEnergies)

    r.launchOverheadMs = 0.0
    if len(s.wallMillis) == len(s.kernelMillis) and len(s.kernelMillis) > p.warmupIterations:
        acc = sum(
            s.wallMillis[i] - s.kernelMillis[i]
            for i in range(p.warmupIterations, len(s.kernelMillis))
        )
        r.launchOverheadMs = acc / (len(s.kernelMillis) - p.warmupIterations)

    return r


def dump_results(p, r):
    if not p.output:
        return
    with open(p.output, "w") as f:
        for e in r.sample.energies:
            f.write("%.17g\n" % e)


def show_human_readable(p, r, indent=1):
    prefix = " " * indent
    raw = ",".join("%g" % x for x in r.sample.kernelMillis)

    print("%s - precision_bits:      %d" % (prefix, r.sample.precisionBits))
    print(
        "%s   param:               { ppwi: %d, wgsize: %d, num_warps: %d, "
        "num_stages: %d%s }"
        % (prefix, r.sample.ppwi, r.sample.wgsize, r.sample.numWarps,
           r.sample.numStages, " (tuned)" if AUTOTUNE_NUM_STAGES else " (pinned)")
    )
    print("%s   raw_iterations:      [%s]" % (prefix, raw))
    print("%s   context_ms:          %.6f" % (prefix, r.sample.contextMillis))
    print("%s   sum_ms:              %.3f" % (prefix, r.ms.sum))
    print("%s   avg_ms:              %.3f" % (prefix, r.ms.mean))
    print("%s   min_ms:              %.3f" % (prefix, r.ms.min))
    print("%s   max_ms:              %.3f" % (prefix, r.ms.max))
    print("%s   stddev_ms:           %.3f" % (prefix, r.ms.stdDev))
    print("%s   giga_interactions/s: %.3f" % (prefix, r.interactionsPerSec / 1e9))
    print("%s   gflop/s:             %.3f" % (prefix, r.gflops))
    print("%s   gfinst/s:            %.3f" % (prefix, r.ginsts))
    print("%s   launch_overhead_ms:  %.3f" % (prefix, r.launchOverheadMs))
    print("%s   energies:            " % prefix)
    for i in range(min(p.outRows, len(r.sample.energies))):
        print("%s     - %.6f" % (prefix, r.sample.energies[i]))

    if r.hasDifferential:
        m = r.vsReference
        print("%s   vs_fp64_reference:" % prefix)
        print("%s     l2_rel_norm:       %.6e" % (prefix, m.l2RelNorm))
        print("%s     rms_abs:           %.6e" % (prefix, m.rmsAbs))
        print("%s     mean_signed:       %.6e" % (prefix, m.meanSigned))
        print("%s     max_abs:           { value: %.6e, index: %d }" % (prefix, m.maxAbs, m.maxAbsIdx))
        print("%s     max_rel:           { value: %.6e, index: %d }" % (prefix, m.maxRel, m.maxRelIdx))
        print("%s     max_ulp:           { value: %d, index: %d }" % (prefix, m.maxUlp, m.maxUlpIdx))
        print("%s     compared:          %d" % (prefix, m.compared))
        print("%s     non_finite:        %d" % (prefix, m.nonFinite))
        print("%s     ulp_histogram:" % prefix)
        for i in range(ULP_NBUCKETS):
            if m.ulpHistogram[i] > 0:
                print('%s       "%s": %d' % (prefix, ulp_bucket_label(i), m.ulpHistogram[i]))
    else:
        print("%s   vs_fp64_reference:   ~ # benchmarked precision is the reference" % prefix)

    m = r.vsDeck
    print("%s   vs_deck_reference:   # informational; ref_energies.out is low precision" % prefix)
    print("%s     l2_rel_norm:       %.6e" % (prefix, m.l2RelNorm))
    print("%s     max_rel:           { value: %.6e, index: %d }" % (prefix, m.maxRel, m.maxRelIdx))
    print(
        "%s     beyond_tolerance:  %d/%d # tolerance = %s%%"
        % (prefix, m.beyondTolerance, m.compared, REPORT_TOLERANCE_PCT)
    )
    sys.stdout.flush()


def show_csv(p, r, header):
    if header:
        print(
            "precision_bits,ppwi,wgsize,sum_ms,avg_ms,min_ms,max_ms,stddev_ms,"
            "interactions/s,gflops/s,gfinst/s,launch_overhead_ms,"
            "l2_rel_norm,rms_abs,mean_signed,max_abs,max_rel,max_ulp,deck_l2_rel_norm"
        )
    sys.stdout.write(
        "%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"
        % (
            r.sample.precisionBits, r.sample.ppwi, r.sample.wgsize,
            r.ms.sum, r.ms.mean, r.ms.min, r.ms.max, r.ms.stdDev,
            r.interactionsPerSec, r.gflops, r.ginsts, r.launchOverheadMs,
        )
    )
    if r.hasDifferential:
        m = r.vsReference
        sys.stdout.write(
            ",%.6e,%.6e,%.6e,%.6e,%.6e,%d"
            % (m.l2RelNorm, m.rmsAbs, m.meanSigned, m.maxAbs, m.maxRel, m.maxUlp)
        )
    else:
        sys.stdout.write(",,,,,")
    sys.stdout.write(",%.6e\n" % r.vsDeck.l2RelNorm)
    sys.stdout.flush()


def print_help():
    print()
    print("Usage: python bude_triton.py [COMMAND|OPTIONS]\n")
    print("Commands:")
    print("  help -h --help       Print this message")
    print("  list -l --list       List available devices")
    print("Options:")
    print("  -d --device  INDEX   Select device at INDEX from output of --list, performs a substring match of device names if INDEX is not an integer")
    print("                       [optional] default=0")
    print("  -i --iter    I       Repeat kernel I times")
    print("                       [optional] default=%d" % DEFAULT_ITERS)
    print("  -n --poses   N       Compute energies for only N poses, use 0 for deck max")
    print("                       [optional] default=0 ")
    print("  -p --ppwi    PPWI    A CSV list of poses per work-item for the kernel, use `all` for everything")
    print("                       [optional] default=%d; available=%s" % (PPWIS[0], ",".join(map(str, PPWIS))))
    print("  -w --wgsize  WGSIZE  A CSV list of work-group sizes, not all implementations support this parameter")
    print("                       [optional] default=<warp_size>; the reference's default of 1")
    print("                       is not expressible in Triton")
    print("                       NOTE: Triton derives num_warps = wgsize / warp_size, so")
    print("                       wgsize must be a multiple of the target's warp width.")
    print("     --deck    DIR     Use the DIR directory as input deck")
    print("                       [optional] default=`%s`" % DEFAULT_DATA_DIR)
    print("  -o --out     PATH    Save resulting energies to PATH (no-op if more than one PPWI/WGSIZE specified)")
    print("                       [optional]")
    print("  -r --rows    N       Output first N row(s) of energy values as part of the on-screen result")
    print("                       [optional] default=%d" % DEFAULT_ENERGY_ENTRIES)
    print("     --csv             Output results in CSV format")
    print("                       [optional] default=false")
    print("")
    print("Precision is not a flag: edit the PRECISION_BITS constant at the top of")
    print("this file, as with the reference's -DPRECISION_BITS.")
    print()


def parse_params(args):
    p = Params()
    wgsizes, ppwis = [], []
    nposes_arg = 0
    precision_bits = PRECISION_BITS

    i = 0

    def value(flag):
        nonlocal i
        if i + 1 >= len(args):
            print("[%s] specified but no value was given" % flag, file=sys.stderr)
            sys.exit(1)
        i += 1
        return args[i]

    def parse_ints(s, name):
        out = []
        for tok in s.split(","):
            try:
                v = int(tok)
                if v < 0:
                    raise ValueError
            except ValueError:
                print("malformed value, positive integer required for <%s>: `%s`" % (name, tok), file=sys.stderr)
                sys.exit(1)
            out.append(v)
        return out

    while i < len(args):
        a = args[i]
        if a in ("--iter", "-i"):
            p.iterations = parse_ints(value(a), "iter")[0]
        elif a in ("--poses", "-n"):
            nposes_arg = parse_ints(value(a), "poses")[0]
        elif a in ("--device", "-d"):
            p.deviceSelector = value(a)
        elif a == "--deck":
            p.deckDir = value(a)
        elif a in ("--out", "-o"):
            p.output = value(a)
        elif a in ("--rows", "-r"):
            p.outRows = parse_ints(value(a), "rows")[0]
        elif a in ("--wgsize", "-w"):
            wgsizes = parse_ints(value(a), "wgsize")
        elif a in ("--ppwi", "-p"):
            s = value(a)
            if s == "all":
                ppwis = list(PPWIS)
            else:
                ppwis = parse_ints(s, "ppwi")
                for v in ppwis:
                    if v not in PPWIS:
                        print("PPWI %d is not a supported value, should be one of `%s`"
                              % (v, ",".join(map(str, PPWIS))), file=sys.stderr)
                        sys.exit(1)
        elif a == "--csv":
            p.csv = True
        elif a in ("list", "--list", "-l"):
            p.list = True
        elif a in ("help", "--help", "-h"):
            print_help()
            sys.exit(0)
        else:
            print("Unrecognized argument '%s' (try '--help')" % a)
            sys.exit(1)
        i += 1

    if not ppwis:
        ppwis = [PPWIS[0]]
    if precision_bits not in (32, 64):
        raise ValueError("precision must be 32 or 64, got %d" % precision_bits)

    if p.list:
        return p, wgsizes, ppwis, precision_bits

    p.ligand = read_nstruct(DISK_ATOM, os.path.join(p.deckDir, "ligand.in"))
    p.protein = read_nstruct(DISK_ATOM, os.path.join(p.deckDir, "protein.in"))
    p.forcefield = read_nstruct(DISK_FFPARAMS, os.path.join(p.deckDir, "forcefield.in"))

    poses = read_nstruct(np.dtype("<f4"), os.path.join(p.deckDir, "poses.in"))
    if len(poses) % 6 != 0:
        raise ValueError("Pose size (%d) not divisible my 6!" % len(poses))
    max_poses = len(poses) // 6
    n = max_poses if nposes_arg == 0 else nposes_arg
    if n > max_poses:
        raise ValueError("Requested poses (%d) exceeded max poses (%d) for deck" % (n, max_poses))
    for k in range(6):
        p.poses[k] = poses[k * max_poses: k * max_poses + n].copy()

    refs = []
    with open(os.path.join(p.deckDir, "ref_energies.out")) as f:
        for line in f:
            if line.strip():
                refs.append(float(line.strip()))
    p.refEnergies = np.array(refs, dtype=np.float32)
    if p.nposes() > len(p.refEnergies):
        raise ValueError(
            "Size of reference energies (%d) is less than poses (%d)"
            % (len(p.refEnergies), p.nposes())
        )

    p.maxPoses = max_poses
    return p, wgsizes, ppwis, precision_bits


def enumerate_devices():
    out = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem = props.total_memory // 1024 // 1024
        arch = getattr(props, "gcnArchName", None)
        tag = arch if arch else "sm_%d%d" % (props.major, props.minor)
        out.append((i, "%s (%dMB;%s)" % (props.name, mem, tag)))
    return out


def select_device(needle, haystack):
    if not needle:
        return haystack[0]
    try:
        return haystack[int(needle)]
    except (ValueError, IndexError):
        print("# Attempting to match device with substring  `%s`" % needle, file=sys.stderr)
    for d in haystack:
        if needle in d[1]:
            return d
    if len(haystack) == 1:
        print("# No matching device but there's only one device, using it anyway", file=sys.stderr)
        return haystack[0]
    print("# No matching devices", file=sys.stderr)
    return (-1, "")


def run(p, wgsizes, ppwis, precision_bits, warp_size):
    devices = enumerate_devices()
    if not devices:
        print(" # (no devices available)", file=sys.stderr)

    if p.list:
        print("index,name" if p.csv else "devices:")
        for idx, name in devices:
            print("%d,%s" % (idx, name)) if p.csv else print('  %d: "%s"' % (idx, name))
        return True

    dev = select_device(p.deviceSelector, devices)
    if dev[0] < 0:
        return False
    torch.cuda.set_device(dev[0])
    device = torch.device("cuda", dev[0])

    if not p.csv:
        print('device: { index: %d,  name: "%s" }' % (dev[0], dev[1]))

    fp = np.float32 if precision_bits == 32 else np.float64

    wantDifferential = precision_bits != 64
    golden = None

    dumped = False
    results = []
    for ppwi in ppwis:
        for wgsize in wgsizes:
            n = p.nposes()
            if n < ppwi * wgsize:
                print(" # WARNING: pose count %d <= (%d (wgsize) * %d (ppwi)), skipping" % (n, wgsize, ppwi))
                continue
            if n % (ppwi * wgsize) != 0:
                print(" # WARNING: pose count %d %% (%d (wgsize) * %d (ppwi)) != 0, skipping" % (n, wgsize, ppwi))
                continue
            if wgsize % warp_size != 0:
                print(" # WARNING: wgsize %d is not a multiple of warp_size %d, "
                      "not expressible in Triton (num_warps = wgsize / warp_size), skipping"
                      % (wgsize, warp_size))
                continue

            try:
                if wantDifferential and golden is None:
                    print("# computing fp64 reference pass (ppwi=%d, wgsize=%d)" % (ppwi, wgsize))
                    golden = fasten(np.float64, p, wgsize, ppwi, warp_size, device).energies

                sample = fasten(fp, p, wgsize, ppwi, warp_size, device)
            except (ConfigUnavailable, _OutOfResources) as e:
                print(" # WARNING: ppwi=%d, wgsize=%d exceeded device resources (%s), skipping"
                      % (ppwi, wgsize, str(e).replace("\n", " ")[:160]))
                continue

            result = evaluate(p, sample, golden if golden is not None else None)
            results.append(result)

            if not dumped:
                dumped = True
                dump_results(p, result)

    if not results:
        print("# no runnable configurations", file=sys.stderr)
        return False

    if not p.csv:
        print("results:")
    for i, r in enumerate(results):
        show_csv(p, r, i == 0) if p.csv else show_human_readable(p, r)

    best = min(results, key=lambda r: r.ms.sum)
    print(
        "%sbest: { min_ms: %.3f, max_ms: %.3f, sum_ms: %.3f, avg_ms: %.3f, ppwi: %d, wgsize: %d }"
        % ("# " if p.csv else "", best.ms.min, best.ms.max, best.ms.sum,
           best.ms.mean, best.sample.ppwi, best.sample.wgsize)
    )
    return True


def main():
    p, wgsizes, ppwis, precision_bits = parse_params(sys.argv[1:])
    warp_size, backend, arch, how = resolve_warp_size()
    default_wgsize = not wgsizes
    if default_wgsize:
        wgsizes = [warp_size]
    if not p.csv:
        print("miniBUDE:  triton %s" % triton.__version__)
        print("backend:   %s (arch %s)" % (backend, arch))
        print("warp_size: %d # resolved from: %s" % (warp_size, how))
        print("precision: %d # PRECISION_BITS constant in source" % precision_bits)
        print("num_stages: %s" % ("autotuned per config over %s"
                                  % (list(NUM_STAGES_CANDIDATES),)
                                  if AUTOTUNE_NUM_STAGES
                                  else "pinned at %d" % NUM_STAGES_DEFAULT))
        print("ppwi:      [%s]" % ",".join(map(str, ppwis)))
        print("wgsize:    [%s]%s" % (",".join(map(str, wgsizes)),
              " # defaulted to one warp; reference default of 1 is not expressible"
              if default_wgsize else ""))
    return 0 if run(p, wgsizes, ppwis, precision_bits, warp_size) else 1


if __name__ == "__main__":
    sys.exit(main())
