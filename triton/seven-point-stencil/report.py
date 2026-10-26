"""Seven-point stencil report: one struct filled by the run, three formatters
that read only from it (text, --csv summary, --csv-raw)."""

import math

ROUTINE = "Laplacian"
DASH = "—"


class Report:
    def __init__(self):
        self.benchmark = "Seven-point stencil"
        self.implementation = "Triton"
        self.backend = ""
        self.device = ""
        self.driver = ""
        self.precision = ""          # float16 / float32 / float64
        self.itemsize = 8
        self.nx = self.ny = self.nz = 0
        self.bx = self.by = self.bz = 0
        self.grid = (0, 0, 0)        # programs per axis
        self.num_warps = 0
        self.fetch_bytes = 0
        self.write_bytes = 0
        self.num_times = 0           # timed iterations, after one untimed run
        self.timing = "device timestamps"
        self.energy_enabled = True
        self.header_notes = []

        self.times = []              # timed iterations, seconds
        self.energy = None           # energy.EnergyResult
        self.correctness = None      # stencil.Correctness

    @property
    def traffic(self):
        return self.fetch_bytes + self.write_bytes


# -- number formatting ---------------------------------------------------------

def sci(x, digits):
    """Scientific notation without exponent padding: 1.230000e-1, 6.83e5."""
    if math.isnan(x) or math.isinf(x):
        return str(x)
    mantissa, exp = ("%.*e" % (digits, x)).split("e")
    return "%se%d" % (mantissa, int(exp))


def count(n):
    return "{:,}".format(n)


def csv_field(s):
    s = str(s)
    if any(ch in s for ch in ',"\n\r'):
        return '"' + s.replace('"', '""') + '"'
    return s


def _stats(t):
    return min(t), max(t), sum(t) / len(t)


def _gbs(bytes_, seconds):
    return bytes_ / seconds / 1.0e9


# -- text ----------------------------------------------------------------------

def _key(name, value):
    return "  %-13s%s" % (name, value)


def text_header(r):
    points = r.nx * r.ny * r.nz
    lines = [
        "%s (%s)" % (r.benchmark, r.implementation),
        _key("Backend", r.backend),
        _key("Device", r.device),
        _key("Driver", r.driver),
        _key("Precision", r.precision),
        _key("Grid", "%s × %s × %s (%s points, %.1f MB per array)"
             % (count(r.nx), count(r.ny), count(r.nz), count(points),
                points * r.itemsize * 1.0e-6)),
        _key("Block size", "%s × %s × %s (256 points per program, %s %s)"
             % (count(r.bx), count(r.by), count(r.bz), count(r.num_warps),
                "warps" if r.backend == "CUDA" else "waves")),
        _key("Programs", "%s × %s × %s" % tuple(count(g) for g in r.grid)),
        _key("Traffic", "%.1f MB read + %.1f MB written (theoretical)"
             % (r.fetch_bytes * 1.0e-6, r.write_bytes * 1.0e-6)),
        _key("Iterations", "%s (after 1 untimed run)" % count(r.num_times)),
        _key("Timing", r.timing),
        _key("Energy", "libkernelenergy (board power)" if r.energy_enabled
             else "disabled (--no-energy)"),
    ]
    lines += ["  note: %s" % n for n in r.header_notes]
    return "\n".join(lines) + "\n"


def text_results(r):
    fmt = "%-12s%16s%14s%14s%14s"
    tmin, tmax, tavg = _stats(r.times)
    lines = [fmt % ("Function", "Bandwidth (GB/s)", "Min (s)", "Max (s)", "Avg (s)"),
             fmt % (ROUTINE, "%.3f" % _gbs(r.traffic, tmin),
                    "%.8f" % tmin, "%.8f" % tmax, "%.8f" % tavg),
             "  note: bandwidth uses the minimum time; from the average time it is %.3f GB/s"
             % _gbs(r.traffic, tavg)]
    return "\n".join(lines) + "\n"


def text_energy(r):
    if r.energy is None:
        return "Energy  disabled (--no-energy)\n"
    e = r.energy
    fmt = "%-8s%14s%14s%16s%12s%13s%10s"
    lines = [fmt % ("Energy", "J/iter", "Total (Wh)", "Avg power (W)", "Time (s)",
                    "Iterations", "Samples"),
             fmt % ("board", "%.6f" % e.joules_per_iter, "%.6f" % e.watt_hours,
                    "%.2f" % e.avg_power_w, "%.6f" % e.seconds,
                    count(e.iterations), count(e.samples))]
    if e.samples < 2:
        lines.append("  note: only %d power sample%s over the timed iterations; energy "
                     "figures are unreliable (increase -n or the grid)"
                     % (e.samples, "" if e.samples == 1 else "s"))
    return "\n".join(lines) + "\n"


def text_correctness(r):
    c = r.correctness
    fmt = "%-22s%16s"
    lines = ["Correctness (%s, untimed run, exact Laplacian = 3 at %s interior points)"
             % (r.precision, count(c.points)),
             fmt % ("Metric", "Value"),
             fmt % ("Max abs err", sci(c.max_abs_err, 3)),
             fmt % ("Mean abs err", sci(c.mean_abs_err, 3)),
             fmt % ("L2 rel err", sci(c.l2_rel_err, 3))]
    lines += ["  note: %s" % n for n in c.notes]
    return "\n".join(lines) + "\n"


def text_body(r):
    return "\n" + text_results(r) + "\n" + text_energy(r) + "\n" + text_correctness(r)


def text_stderr_tail(r):
    return "\n" + text_energy(r) + "\n" + text_correctness(r)


# -- CSV -----------------------------------------------------------------------

_PREFIX = "implementation,backend,device,precision"


def _prefix(r):
    return ",".join(csv_field(x) for x in (r.implementation, r.backend, r.device, r.precision))


def csv_summary(r):
    lines = [_PREFIX + ",nx,ny,nz,block_x,block_y,block_z,num_times,routine,bw_gbs,"
             "min_s,max_s,avg_s,energy_j_per_iter,avg_power_w"]
    tmin, tmax, tavg = _stats(r.times)
    energy = ("%.9f,%.4f" % (r.energy.joules_per_iter, r.energy.avg_power_w)
              if r.energy is not None else ",")
    lines.append("%s,%d,%d,%d,%d,%d,%d,%d,%s,%.4f,%.9f,%.9f,%.9f,%s" % (
        _prefix(r), r.nx, r.ny, r.nz, r.bx, r.by, r.bz, r.num_times, ROUTINE,
        _gbs(r.traffic, tmin), tmin, tmax, tavg, energy))
    return "\n".join(lines) + "\n"


def csv_raw(r):
    lines = [_PREFIX + ",nx,ny,nz,block_x,block_y,block_z,routine,iteration,time_s,bw_gbs"]
    head = "%s,%d,%d,%d,%d,%d,%d,%s" % (_prefix(r), r.nx, r.ny, r.nz, r.bx, r.by, r.bz, ROUTINE)
    for i, t in enumerate(r.times, start=1):
        lines.append("%s,%d,%.9f,%.4f" % (head, i, t, _gbs(r.traffic, t)))
    return "\n".join(lines) + "\n"
