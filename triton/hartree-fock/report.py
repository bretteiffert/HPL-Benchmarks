"""Hartree-Fock report: one struct filled by the run, three formatters that read
only from it (text, --csv summary, --csv-raw)."""

import math

ROUTINE = "Fock build"
DASH = "—"


class Report:
    def __init__(self):
        self.benchmark = "Hartree-Fock"
        self.implementation = "Triton"
        self.backend = ""
        self.device = ""
        self.driver = ""
        self.precision = ""          # float32 / float64
        self.sqrt_mode = "native"    # native / legacy-sqrtf
        self.input = ""              # input file name (basename)
        self.natoms = 0
        self.ngauss = 0
        self.atom_pairs = 0
        self.quartets = 0
        self.block_size = 0
        self.grid = 0
        self.num_times = 0           # timed iterations, after one untimed run
        self.check = ""              # header text for the Check line
        self.timing = "device timestamps"
        self.energy_enabled = True
        self.header_notes = []

        self.times = []              # timed iterations, seconds
        self.energy = None           # energy.EnergyResult
        self.correctness = None      # hf.Correctness


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


# -- text ----------------------------------------------------------------------

def _key(name, value):
    return "  %-13s%s" % (name, value)


def text_header(r):
    lines = [
        "%s (%s)" % (r.benchmark, r.implementation),
        _key("Backend", r.backend),
        _key("Device", r.device),
        _key("Driver", r.driver),
        _key("Precision", r.precision),
        _key("Sqrt", "native" if r.sqrt_mode == "native"
             else "legacy (float32 sqrt, --legacy-sqrt)"),
        _key("Input", "%s (%s atoms, %s Gaussians)"
             % (r.input, count(r.natoms), count(r.ngauss))),
        _key("Quartets", "%s (%s atom pairs)" % (count(r.quartets), count(r.atom_pairs))),
        _key("Block size", count(r.block_size)),
        _key("Grid", "%s program%s" % (count(r.grid), "" if r.grid == 1 else "s")),
        _key("Iterations", "%s (after 1 untimed run)" % count(r.num_times)),
        _key("Check", r.check),
        _key("Timing", r.timing),
        _key("Energy", "libkernelenergy (board power)" if r.energy_enabled
             else "disabled (--no-energy)"),
    ]
    lines += ["  note: %s" % n for n in r.header_notes]
    return "\n".join(lines) + "\n"


def text_results(r):
    fmt = "%-12s%14s%14s%14s"
    tmin, tmax, tavg = _stats(r.times)
    return "\n".join([fmt % ("Routine", "Min (s)", "Max (s)", "Avg (s)"),
                      fmt % (ROUTINE, "%.8f" % tmin, "%.8f" % tmax, "%.8f" % tavg)]) + "\n"


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
                     "figures are unreliable (increase -n)"
                     % (e.samples, "" if e.samples == 1 else "s"))
    return "\n".join(lines) + "\n"


def text_correctness(r):
    c = r.correctness
    p = r.precision
    fmt = "%-42s%24s"
    lines = ["Correctness (%s, %s sqrt, untimed run)" % (p, r.sqrt_mode),
             fmt % ("Quantity", "Value"),
             fmt % ("2e- energy, %s reduction" % p, sci(c.e2_energy, 15))]
    if c.checked:
        lines += [
            fmt % ("2e- energy, %s Fock, float64 reduction" % p, sci(c.e_gpu_fp64, 15)),
            fmt % ("2e- energy, float64 CPU reference", sci(c.e_ref, 15)),
            fmt % ("rel err vs reference, Fock build only", sci(c.rel_err_fock, 3)),
            fmt % ("rel err vs reference, incl. reduction", sci(c.rel_err_ref, 3)),
            fmt % ("max |F - F_ref|", sci(c.max_abs, 3)),
            fmt % ("max |F - F_ref| / max |F_ref|", sci(c.max_rel, 3)),
            fmt % ("rms |F - F_ref|", sci(c.rms, 3)),
            fmt % ("max |F(i,j) - F(j,i)|", sci(c.asym, 3)),
            fmt % ("2e- energy, symmetrized F (diagnostic)", sci(c.e_symmetrized, 15)),
        ]
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
    lines = [_PREFIX + ",sqrt_mode,input,natoms,ngauss,block_size,num_times,routine,"
             "min_s,max_s,avg_s,e2_energy,rel_err_ref,energy_j_per_iter,avg_power_w"]
    tmin, tmax, tavg = _stats(r.times)
    c = r.correctness
    rel = "" if c.rel_err_ref is None else "%.6e" % c.rel_err_ref
    if r.energy is not None:
        energy = "%.9f,%.4f" % (r.energy.joules_per_iter, r.energy.avg_power_w)
    else:
        energy = ","
    lines.append("%s,%s,%s,%d,%d,%d,%d,%s,%.9f,%.9f,%.9f,%.16e,%s,%s" % (
        _prefix(r), r.sqrt_mode, csv_field(r.input), r.natoms, r.ngauss, r.block_size,
        r.num_times, ROUTINE, tmin, tmax, tavg, c.e2_energy, rel, energy))
    return "\n".join(lines) + "\n"


def csv_raw(r):
    lines = [_PREFIX + ",sqrt_mode,input,natoms,ngauss,block_size,routine,iteration,time_s"]
    head = "%s,%s,%s,%d,%d,%d,%s" % (_prefix(r), r.sqrt_mode, csv_field(r.input),
                                     r.natoms, r.ngauss, r.block_size, ROUTINE)
    for i, t in enumerate(r.times, start=1):
        lines.append("%s,%d,%.9f" % (head, i, t))
    return "\n".join(lines) + "\n"
