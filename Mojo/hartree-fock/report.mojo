# Report struct and its three formatters (text, --csv, --csv-raw).
#
# The run fills one Report; the formatters only read from it. Nothing here
# touches the device. Stream rules (applied by main):
#   text mode -> everything on stdout;
#   CSV modes -> only CSV on stdout; header, energy, correctness on stderr.

from std.collections import List
from std.math import floor, log10
from std.utils.numerics import isnan, isinf


# ===----------------------------------------------------------------------=== #
# Number formatting (Mojo has no printf-style float formatting)
# ===----------------------------------------------------------------------=== #


def _pow10i(d: Int) -> Int:
    var p = 1
    for _ in range(d):
        p *= 10
    return p


def _pow10f(e: Int) -> Float64:
    var p = Float64(1.0)
    if e >= 0:
        for _ in range(e):
            p *= 10.0
    else:
        for _ in range(-e):
            p /= 10.0
    return p


def _zero_pad(v: Int, width: Int) -> String:
    var s = String(v)
    var missing = width - s.byte_length()
    if missing > 0:
        return String("0") * missing + s
    return s


def _non_finite(x: Float64) -> String:
    if Bool(isnan(x)):
        return "nan"
    if x > 0:
        return "inf"
    return "-inf"


def fmt_fixed(x: Float64, decimals: Int) -> String:
    """`x` with exactly `decimals` digits after the point (round half up)."""
    if Bool(isnan(x)) or Bool(isinf(x)):
        return _non_finite(x)
    var neg = x < 0.0
    var p = _pow10i(decimals)
    var scaled = abs(x) * Float64(p)
    if scaled >= 9.0e18:
        return String(x)
    var r = Int(scaled + 0.5)
    var s = String(r // p)
    if decimals > 0:
        s += "." + _zero_pad(r % p, decimals)
    if neg and r != 0:
        s = "-" + s
    return s


def fmt_sci(x: Float64, decimals: Int) -> String:
    """Scientific notation with `decimals` mantissa digits, e.g. 1.230e-7."""
    if Bool(isnan(x)) or Bool(isinf(x)):
        return _non_finite(x)
    var p = _pow10i(decimals)
    if x == 0.0:
        var z = String("0")
        if decimals > 0:
            z += "." + _zero_pad(0, decimals)
        return z + "e0"
    var neg = x < 0.0
    var ax = abs(x)
    var e = Int(floor(log10(ax)))
    var m: Float64
    if e < -290:
        m = (ax * _pow10f(300)) / _pow10f(e + 300)
    else:
        m = ax / _pow10f(e)
    if m >= 10.0:
        m /= 10.0
        e += 1
    if m < 1.0:
        m *= 10.0
        e -= 1
    var r = Int(m * Float64(p) + 0.5)
    if r >= 10 * p:
        r //= 10
        e += 1
    var s = String(r // p)
    if decimals > 0:
        s += "." + _zero_pad(r % p, decimals)
    s += "e" + String(e)
    if neg:
        s = "-" + s
    return s


def thousands(n: Int) -> String:
    var neg = n < 0
    var digits = String(abs(n))
    var out = String("")
    var count = digits.byte_length()
    for i in range(count):
        if i > 0 and (count - i) % 3 == 0:
            out += ","
        out += digits[byte=i]
    if neg:
        return "-" + out
    return out


def rjust(s: String, width: Int) -> String:
    var missing = width - s.count_codepoints()
    if missing > 0:
        return String(" ") * missing + s
    return s


def ljust(s: String, width: Int) -> String:
    var missing = width - s.count_codepoints()
    if missing > 0:
        return s + String(" ") * missing
    return s


def csv_field(s: String) -> String:
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


# ===----------------------------------------------------------------------=== #
# Report
# ===----------------------------------------------------------------------=== #


struct Report(Movable):
    # header
    var implementation: String
    var backend: String
    var device: String
    var driver: String
    var precision: String
    var input: String  # file name as given on the command line
    var input_name: String  # basename, used in CSV
    var natoms: Int
    var ngauss: Int
    var pairs: Int
    var quartets: UInt64
    var block_size: Int
    var blocks: Int
    var sqrt_mode: String  # "native" | "legacy-sqrtf"
    var num_times: Int
    var timing: String
    var energy_enabled: Bool
    var check_line: String
    var header_notes: List[String]
    # results (timed runs only)
    var times: List[Float64]
    var min_s: Float64
    var max_s: Float64
    var avg_s: Float64
    # energy
    var energy_measured: Bool
    var joules: Float64
    var energy_time_s: Float64
    var avg_power_w: Float64
    var energy_iterations: Int
    var samples: Int
    var energy_notes: List[String]
    # correctness (from the untimed run)
    var e2_energy: Float64  # run-precision Fock, run-precision reduction
    var checked: Bool
    var e_gpu: Float64
    var e_ref: Float64
    var rel_loss: Float64
    var rel_inpipe: Float64
    var maxd: Float64
    var maxd_rel: Float64
    var rmsd: Float64
    var asym: Float64
    var e_sym: Float64
    var check_notes: List[String]

    def __init__(out self):
        self.implementation = "Mojo"
        self.backend = ""
        self.device = ""
        self.driver = ""
        self.precision = ""
        self.input = ""
        self.input_name = ""
        self.natoms = 0
        self.ngauss = 0
        self.pairs = 0
        self.quartets = 0
        self.block_size = 0
        self.blocks = 0
        self.sqrt_mode = "native"
        self.num_times = 0
        self.timing = "device timestamps"
        self.energy_enabled = True
        self.check_line = ""
        self.header_notes = List[String]()
        self.times = List[Float64]()
        self.min_s = 0.0
        self.max_s = 0.0
        self.avg_s = 0.0
        self.energy_measured = False
        self.joules = 0.0
        self.energy_time_s = 0.0
        self.avg_power_w = 0.0
        self.energy_iterations = 0
        self.samples = 0
        self.energy_notes = List[String]()
        self.e2_energy = 0.0
        self.checked = False
        self.e_gpu = 0.0
        self.e_ref = 0.0
        self.rel_loss = 0.0
        self.rel_inpipe = 0.0
        self.maxd = 0.0
        self.maxd_rel = 0.0
        self.rmsd = 0.0
        self.asym = 0.0
        self.e_sym = 0.0
        self.check_notes = List[String]()

    def finish_times(mut self) raises:
        var n = len(self.times)
        if n == 0:
            raise Error("no timed iterations recorded")
        self.min_s = self.times[0]
        self.max_s = self.times[0]
        var total = Float64(0)
        for i in range(n):
            var t = self.times[i]
            if t < self.min_s:
                self.min_s = t
            if t > self.max_s:
                self.max_s = t
            total += t
        self.avg_s = total / Float64(n)

    def j_per_iter(self) -> Float64:
        return self.joules / Float64(self.energy_iterations)


def _key(k: String, v: String) -> String:
    return "  " + ljust(k, 13) + v + "\n"


def _notes(notes: List[String]) -> String:
    var s = String("")
    for i in range(len(notes)):
        s += "  note: " + notes[i] + "\n"
    return s


# ===----------------------------------------------------------------------=== #
# Text formatter
# ===----------------------------------------------------------------------=== #


def format_header(r: Report) -> String:
    var s = "Hartree-Fock (" + r.implementation + ")\n"
    s += _key("Backend", r.backend)
    s += _key("Device", r.device)
    s += _key("Driver", r.driver)
    s += _key("Precision", r.precision)
    s += _key(
        "Input",
        r.input
        + " ("
        + thousands(r.natoms)
        + " atoms, "
        + thousands(r.ngauss)
        + " Gaussians)",
    )
    s += _key(
        "Quartets",
        thousands(Int(r.quartets))
        + " ("
        + thousands(r.pairs)
        + " atom pairs)",
    )
    s += _key("Block size", String(r.block_size))
    s += _key("Grid", thousands(r.blocks) + " blocks")
    s += _key("Sqrt mode", r.sqrt_mode)
    s += _key("Iterations", thousands(r.num_times) + " (after 1 untimed run)")
    s += _key("Timing", r.timing)
    if r.energy_enabled:
        s += _key("Energy", "libkernelenergy (board power)")
    else:
        s += _key("Energy", "disabled (--no-energy)")
    s += _key("Check", r.check_line)
    s += _notes(r.header_notes)
    return s


def format_results(r: Report) -> String:
    var s = (
        ljust("Routine", 12)
        + rjust("Min (s)", 14)
        + rjust("Max (s)", 14)
        + rjust("Avg (s)", 14)
        + "\n"
    )
    s += ljust("Fock build", 12)
    s += rjust(fmt_fixed(r.min_s, 8), 14)
    s += rjust(fmt_fixed(r.max_s, 8), 14)
    s += rjust(fmt_fixed(r.avg_s, 8), 14)
    s += "\n"
    s += (
        "  note: time per Fock build over "
        + thousands(len(r.times))
        + " timed iterations\n"
    )
    return s


def format_energy(r: Report) -> String:
    if not r.energy_enabled:
        return ljust("Energy", 8) + "disabled (--no-energy)\n"
    var s = (
        ljust("Energy", 8)
        + rjust("J/iter", 14)
        + rjust("Total (Wh)", 14)
        + rjust("Avg power (W)", 16)
        + rjust("Time (s)", 12)
        + rjust("Iterations", 13)
        + rjust("Samples", 10)
        + "\n"
    )
    s += ljust("board", 8)
    s += rjust(fmt_fixed(r.j_per_iter(), 6), 14)
    s += rjust(fmt_fixed(r.joules / 3600.0, 6), 14)
    s += rjust(fmt_fixed(r.avg_power_w, 2), 16)
    s += rjust(fmt_fixed(r.energy_time_s, 6), 12)
    s += rjust(thousands(r.energy_iterations), 13)
    s += rjust(thousands(r.samples), 10)
    s += "\n"
    s += _notes(r.energy_notes)
    return s


def _row(label: String, value: String) -> String:
    return ljust(label, 40) + rjust(value, 24) + "\n"


def format_correctness(r: Report) -> String:
    var p = r.precision
    var s = (
        "Correctness ("
        + p
        + " Fock build, "
        + r.sqrt_mode
        + " sqrt, untimed run)\n"
    )
    s += ljust("Quantity", 40) + rjust("Value", 24) + "\n"
    s += _row("2e- energy, " + p + " Fock, " + p + " sum", fmt_sci(r.e2_energy, 15))
    if r.checked:
        s += _row("2e- energy, " + p + " Fock, float64 sum", fmt_sci(r.e_gpu, 15))
        s += _row("2e- energy, float64 CPU reference", fmt_sci(r.e_ref, 15))
        s += _row("rel err, Fock build only", fmt_sci(r.rel_loss, 3))
        s += _row("rel err, incl. " + p + " sum", fmt_sci(r.rel_inpipe, 3))
        s += _row("max |F - F_ref|", fmt_sci(r.maxd, 3))
        s += _row("max |F - F_ref| / max |F_ref|", fmt_sci(r.maxd_rel, 3))
        s += _row("rms |F - F_ref|", fmt_sci(r.rmsd, 3))
        s += _row("max |F(i,j) - F(j,i)|", fmt_sci(r.asym, 3))
        s += _row("2e- energy, symmetrized F (diagnostic)", fmt_sci(r.e_sym, 15))
    s += _notes(r.check_notes)
    return s


def format_text(r: Report) -> String:
    """Sections 2-4 (the header is printed before the run starts)."""
    return (
        format_results(r)
        + "\n"
        + format_energy(r)
        + "\n"
        + format_correctness(r)
    )


# ===----------------------------------------------------------------------=== #
# CSV formatters
# ===----------------------------------------------------------------------=== #

comptime CSV_PREFIX = "implementation,backend,device,precision,"


def _prefix(r: Report) -> String:
    return (
        csv_field(r.implementation)
        + ","
        + csv_field(r.backend)
        + ","
        + csv_field(r.device)
        + ","
        + csv_field(r.precision)
        + ","
    )


def _shape(r: Report) -> String:
    return (
        csv_field(r.sqrt_mode)
        + ","
        + csv_field(r.input_name)
        + ","
        + String(r.natoms)
        + ","
        + String(r.ngauss)
        + ","
        + String(r.block_size)
        + ","
        + String(r.quartets)
        + ","
        + String(r.blocks)
        + ","
    )


def csv_header() -> String:
    return (
        CSV_PREFIX
        + "sqrt_mode,input,natoms,ngauss,block_size,quartets,groups,num_times,routine,"
        + "min_s,max_s,avg_s,e2_energy,rel_err_ref,energy_j_per_iter,"
        + "avg_power_w"
    )


def csv_rows(r: Report) -> String:
    var s = _prefix(r) + _shape(r)
    s += String(r.num_times) + ",Fock build,"
    s += fmt_fixed(r.min_s, 9) + ","
    s += fmt_fixed(r.max_s, 9) + ","
    s += fmt_fixed(r.avg_s, 9) + ","
    s += fmt_sci(r.e2_energy, 15) + ","
    if r.checked:
        s += fmt_sci(r.rel_inpipe, 6)
    s += ","
    if r.energy_enabled and r.energy_measured:
        s += fmt_fixed(r.j_per_iter(), 6) + "," + fmt_fixed(r.avg_power_w, 2)
    else:
        s += ","
    return s + "\n"


def csv_raw_header() -> String:
    return (
        CSV_PREFIX
        + "sqrt_mode,input,natoms,ngauss,block_size,quartets,groups,routine,iteration,time_s"
    )


def csv_raw_rows(r: Report) -> String:
    var s = String("")
    for j in range(len(r.times)):
        s += _prefix(r) + _shape(r)
        s += "Fock build," + String(j + 1) + ","
        s += fmt_fixed(r.times[j], 9) + "\n"
    return s
