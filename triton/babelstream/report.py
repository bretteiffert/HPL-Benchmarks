"""BabelStream report: one struct filled by the run, three formatters that read
only from it (text, --csv summary, --csv-raw)."""

import math

ROUTINES = ["Copy", "Mul", "Add", "Triad", "Dot"]
DASH = "—"


class Report:
    def __init__(self):
        self.benchmark = "BabelStream"
        self.implementation = "Triton"
        self.backend = ""
        self.device = ""
        self.driver = ""
        self.precision = ""          # float16 / float32 / float64
        self.itemsize = 8
        self.array_size = 0
        self.block_size = 0
        self.dot_groups = 0
        self.dot_groups_detail = ""  # "(4 × 108 SMs)" or fallback text
        self.num_times = 0
        self.fmad = "unset"
        self.timing = "device timestamps"
        self.energy_enabled = True
        self.mibibytes = False
        self.header_notes = []

        self.init_time = None        # seconds, one-shot
        self.times = {r: [] for r in ROUTINES}  # timed iterations only
        self.energy = None           # energy.EnergyResult
        self.correctness = None      # stream.Correctness

    def bytes_moved(self, routine):
        arrays = {"Init": 3, "Copy": 2, "Mul": 2, "Add": 3, "Triad": 3, "Dot": 2}[routine]
        return arrays * self.itemsize * self.array_size

    @property
    def timed_iterations(self):
        return self.num_times - 1


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


def _bandwidth(bytes_, seconds, mibibytes):
    return bytes_ / seconds / (2.0 ** 20 if mibibytes else 1.0e9)


# -- text ----------------------------------------------------------------------

def _key(name, value):
    return "  %-13s%s" % (name, value)


def text_header(r):
    if r.mibibytes:
        unit, div = "MiB", 2.0 ** 20
    else:
        unit, div = "MB", 1.0e6
    per_array = r.array_size * r.itemsize / div
    lines = [
        "%s (%s)" % (r.benchmark, r.implementation),
        _key("Backend", r.backend),
        _key("Device", r.device),
        _key("Driver", r.driver),
        _key("Precision", r.precision),
        _key("Array size", "%s elements (%.1f %s per array, %.1f %s total)"
             % (count(r.array_size), per_array, unit, 3 * per_array, unit)),
        _key("Block size", count(r.block_size)),
        _key("Dot groups", "%s %s" % (count(r.dot_groups), r.dot_groups_detail)),
        _key("Iterations", "%s (first excluded from statistics)" % count(r.num_times)),
        _key("FMAD", r.fmad),
        _key("Timing", r.timing),
        _key("Energy", "libkernelenergy (board power)" if r.energy_enabled
             else "disabled (--no-energy)"),
    ]
    lines += ["  note: %s" % n for n in r.header_notes]
    return "\n".join(lines) + "\n"


def text_results(r):
    bw_head = "Bandwidth (%s)" % ("MiB/s" if r.mibibytes else "GB/s")
    w = max(16, len(bw_head))
    fmt = "%-12s%" + str(w) + "s%14s%14s%14s"
    lines = [fmt % ("Function", bw_head, "Min (s)", "Max (s)", "Avg (s)")]
    lines.append(fmt % ("Init", "%.3f" % _bandwidth(r.bytes_moved("Init"), r.init_time, r.mibibytes),
                        "%.8f" % r.init_time, DASH, DASH))
    for routine in ROUTINES:
        tmin, tmax, tavg = _stats(r.times[routine])
        lines.append(fmt % (routine,
                            "%.3f" % _bandwidth(r.bytes_moved(routine), tmin, r.mibibytes),
                            "%.8f" % tmin, "%.8f" % tmax, "%.8f" % tavg))
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
                     "figures are unreliable (increase -s or -n)"
                     % (e.samples, "" if e.samples == 1 else "s"))
    return "\n".join(lines) + "\n"


def text_correctness(r):
    c = r.correctness
    fmt = "%-5s%21s%20s%20s"
    lines = ["Correctness (%s, %s iterations)" % (r.precision, count(r.num_times)),
             fmt % ("Array", "Expected", "Checksum rel err", "Max elem rel err")]
    for name, expected, chk, mx in c.rows + [c.dot_row]:
        lines.append(fmt % (name, sci(expected, 6), sci(chk, 3),
                            DASH if mx is None else sci(mx, 3)))
    lines.append("  note: actual dot = %s" % sci(c.actual_dot, 9))
    lines += ["  note: %s" % n for n in c.notes]
    return "\n".join(lines) + "\n"


def text_body(r):
    """Sections 2-4 of the text report."""
    return "\n" + text_results(r) + "\n" + text_energy(r) + "\n" + text_correctness(r)


def text_stderr_tail(r):
    """Energy and correctness, written to stderr in the CSV modes."""
    return "\n" + text_energy(r) + "\n" + text_correctness(r)


# -- CSV -----------------------------------------------------------------------

_PREFIX = "implementation,backend,device,precision"


def _prefix(r):
    return ",".join(csv_field(x) for x in (r.implementation, r.backend, r.device, r.precision))


def csv_summary(r):
    lines = [_PREFIX + ",array_size,block_size,dot_groups,num_times,routine,bw_gbs,"
             "min_s,max_s,avg_s,energy_j_per_iter,avg_power_w"]
    head = "%s,%d,%d,%d,%d" % (_prefix(r), r.array_size, r.block_size, r.dot_groups, r.num_times)
    t = r.init_time
    lines.append("%s,Init,%.4f,%.9f,,,," % (head, _bandwidth(r.bytes_moved("Init"), t, False), t))
    if r.energy is not None:
        energy = "%.9f,%.4f" % (r.energy.joules_per_iter, r.energy.avg_power_w)
    else:
        energy = ","
    for routine in ROUTINES:
        tmin, tmax, tavg = _stats(r.times[routine])
        lines.append("%s,%s,%.4f,%.9f,%.9f,%.9f,%s" % (
            head, routine, _bandwidth(r.bytes_moved(routine), tmin, False),
            tmin, tmax, tavg, energy))
    return "\n".join(lines) + "\n"


def csv_raw(r):
    lines = [_PREFIX + ",array_size,block_size,routine,iteration,time_s,bw_gbs"]
    head = "%s,%d,%d" % (_prefix(r), r.array_size, r.block_size)
    t = r.init_time
    lines.append("%s,Init,1,%.9f,%.4f" % (head, t, _bandwidth(r.bytes_moved("Init"), t, False)))
    for routine in ROUTINES:
        size = r.bytes_moved(routine)
        for i, t in enumerate(r.times[routine], start=1):
            lines.append("%s,%s,%d,%.9f,%.4f" % (head, routine, i, t, _bandwidth(size, t, False)))
    return "\n".join(lines) + "\n"
