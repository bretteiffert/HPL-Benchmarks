# Report struct and its three formatters (text, --csv, --csv-raw).
#
# The run fills one Report; the formatters only read from it. Nothing here
# touches the device. Stream rules (applied by main):
#   text mode -> everything on stdout;
#   CSV modes -> only CSV on stdout; header, energy, correctness on stderr.

from std.collections import List
from std.math import floor, log10
from std.utils.numerics import isnan, isinf

comptime DASH = "—"


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


struct Row(Copyable):
    var routine: String
    var bytes: Int  # bytes moved by one launch
    var one_shot: Bool
    var times: List[Float64]  # timed iterations, iteration 1..N
    var min_s: Float64
    var max_s: Float64
    var avg_s: Float64

    def __init__(out self, routine: String, bytes: Int, one_shot: Bool):
        self.routine = routine
        self.bytes = bytes
        self.one_shot = one_shot
        self.times = List[Float64]()
        self.min_s = 0.0
        self.max_s = 0.0
        self.avg_s = 0.0

    def finish(mut self) raises:
        """Compute min / max / avg over the timed iterations."""
        var n = len(self.times)
        if n == 0:
            raise Error("no timed iterations recorded for " + self.routine)
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


struct CheckRow(Copyable):
    var name: String
    var expected: Float64
    var checksum_rel_err: Float64
    var max_rel_err: Float64
    var has_max: Bool

    def __init__(
        out self,
        name: String,
        expected: Float64,
        checksum_rel_err: Float64,
        max_rel_err: Float64,
        has_max: Bool,
    ):
        self.name = name
        self.expected = expected
        self.checksum_rel_err = checksum_rel_err
        self.max_rel_err = max_rel_err
        self.has_max = has_max


struct Report(Movable):
    # header
    var implementation: String
    var backend: String
    var device: String
    var driver: String
    var precision: String
    var bytes_per_elem: Int
    var array_size: Int
    var block_size: Int
    var dot_groups: Int
    var sm_count: Int
    var sm_label: String
    var sm_known: Bool
    var num_times: Int
    var timing: String
    var energy_enabled: Bool
    var mibibytes: Bool
    var header_notes: List[String]
    # results
    var rows: List[Row]
    # energy (valid when energy_enabled and energy_measured)
    var energy_measured: Bool
    var joules: Float64
    var energy_time_s: Float64
    var avg_power_w: Float64
    var energy_iterations: Int
    var samples: Int
    var energy_notes: List[String]
    # correctness
    var checks: List[CheckRow]
    var actual_dot: Float64
    var check_notes: List[String]

    def __init__(out self):
        self.implementation = "Mojo"
        self.backend = ""
        self.device = ""
        self.driver = ""
        self.precision = ""
        self.bytes_per_elem = 0
        self.array_size = 0
        self.block_size = 0
        self.dot_groups = 0
        self.sm_count = 0
        self.sm_label = "SMs"
        self.sm_known = False
        self.num_times = 0
        self.timing = "device timestamps"
        self.energy_enabled = True
        self.mibibytes = False
        self.header_notes = List[String]()
        self.rows = List[Row]()
        self.energy_measured = False
        self.joules = 0.0
        self.energy_time_s = 0.0
        self.avg_power_w = 0.0
        self.energy_iterations = 0
        self.samples = 0
        self.energy_notes = List[String]()
        self.checks = List[CheckRow]()
        self.actual_dot = 0.0
        self.check_notes = List[String]()

    def timed_iterations(self) -> Int:
        return self.num_times - 1

    def bw_text(self, bytes: Int, seconds: Float64) -> Float64:
        """Bandwidth for the text report: GB/s, or MiB/s with --mibibytes."""
        if self.mibibytes:
            return Float64(bytes) / (1024.0 * 1024.0) / seconds
        return Float64(bytes) * 1.0e-9 / seconds

    def j_per_iter(self) -> Float64:
        return self.joules / Float64(self.energy_iterations)


def _size(r: Report, bytes: Int) -> String:
    if r.mibibytes:
        return fmt_fixed(Float64(bytes) / (1024.0 * 1024.0), 1) + " MiB"
    return fmt_fixed(Float64(bytes) * 1.0e-6, 1) + " MB"


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
    var s = "BabelStream (" + r.implementation + ")\n"
    s += _key("Backend", r.backend)
    s += _key("Device", r.device)
    s += _key("Driver", r.driver)
    s += _key("Precision", r.precision)
    var per_array = r.array_size * r.bytes_per_elem
    s += _key(
        "Array size",
        thousands(r.array_size)
        + " elements ("
        + _size(r, per_array)
        + " per array, "
        + _size(r, 3 * per_array)
        + " total)",
    )
    s += _key("Block size", String(r.block_size))
    var groups = thousands(r.dot_groups) + " (4 × "
    if r.sm_known:
        groups += thousands(r.sm_count) + " " + r.sm_label + ")"
    else:
        groups += thousands(r.sm_count) + ", " + r.sm_label + " count unknown)"
    s += _key("Dot groups", groups)
    s += _key(
        "Iterations",
        thousands(r.num_times) + " (first excluded from statistics)",
    )
    s += _key("Timing", r.timing)
    if r.energy_enabled:
        s += _key("Energy", "libkernelenergy (board power)")
    else:
        s += _key("Energy", "disabled (--no-energy)")
    s += _notes(r.header_notes)
    return s


def format_results(r: Report) -> String:
    var unit = String("MiB/s") if r.mibibytes else String("GB/s")
    var bw_w = 17 if r.mibibytes else 16
    var s = (
        ljust("Function", 12)
        + rjust("Bandwidth (" + unit + ")", bw_w)
        + rjust("Min (s)", 14)
        + rjust("Max (s)", 14)
        + rjust("Avg (s)", 14)
        + "\n"
    )
    for i in range(len(r.rows)):
        ref row = r.rows[i]
        s += ljust(row.routine, 12)
        s += rjust(fmt_fixed(r.bw_text(row.bytes, row.min_s), 3), bw_w)
        s += rjust(fmt_fixed(row.min_s, 8), 14)
        if row.one_shot:
            s += rjust(DASH, 14) + rjust(DASH, 14)
        else:
            s += rjust(fmt_fixed(row.max_s, 8), 14)
            s += rjust(fmt_fixed(row.avg_s, 8), 14)
        s += "\n"
    s += (
        "  note: bandwidth uses the minimum time over "
        + thousands(r.timed_iterations())
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


def format_correctness(r: Report) -> String:
    var s = (
        "Correctness ("
        + r.precision
        + ", "
        + thousands(r.num_times)
        + " iterations)\n"
    )
    s += (
        ljust("Array", 8)
        + rjust("Expected", 18)
        + rjust("Checksum rel err", 20)
        + rjust("Max elem rel err", 20)
        + "\n"
    )
    for i in range(len(r.checks)):
        ref c = r.checks[i]
        s += ljust(c.name, 8)
        s += rjust(fmt_sci(c.expected, 6), 18)
        s += rjust(fmt_sci(c.checksum_rel_err, 3), 20)
        if c.has_max:
            s += rjust(fmt_sci(c.max_rel_err, 3), 20)
        else:
            s += rjust(DASH, 20)
        s += "\n"
    s += "  note: actual dot = " + fmt_sci(r.actual_dot, 9) + "\n"
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


def _gbs(bytes: Int, seconds: Float64) -> String:
    return fmt_fixed(Float64(bytes) * 1.0e-9 / seconds, 4)


def csv_header() -> String:
    return (
        CSV_PREFIX
        + "array_size,block_size,dot_groups,num_times,routine,bw_gbs,"
        + "min_s,max_s,avg_s,energy_j_per_iter,avg_power_w"
    )


def csv_rows(r: Report) -> String:
    var s = String("")
    for i in range(len(r.rows)):
        ref row = r.rows[i]
        s += _prefix(r)
        s += String(r.array_size) + "," + String(r.block_size) + ","
        s += String(r.dot_groups) + "," + String(r.num_times) + ","
        s += csv_field(row.routine) + ","
        s += _gbs(row.bytes, row.min_s) + ","
        s += fmt_fixed(row.min_s, 9) + ","
        if row.one_shot:
            s += ",,"
        else:
            s += fmt_fixed(row.max_s, 9) + "," + fmt_fixed(row.avg_s, 9) + ","
        # Energy covers the timed loop only, so the one-shot Init row has none.
        if r.energy_enabled and r.energy_measured and not row.one_shot:
            s += fmt_fixed(r.j_per_iter(), 6) + "," + fmt_fixed(
                r.avg_power_w, 2
            )
        else:
            s += ","
        s += "\n"
    return s


def csv_raw_header() -> String:
    return CSV_PREFIX + "array_size,block_size,routine,iteration,time_s,bw_gbs"


def csv_raw_rows(r: Report) -> String:
    var s = String("")
    for i in range(len(r.rows)):
        ref row = r.rows[i]
        for j in range(len(row.times)):
            s += _prefix(r)
            s += String(r.array_size) + "," + String(r.block_size) + ","
            s += csv_field(row.routine) + "," + String(j + 1) + ","
            s += fmt_fixed(row.times[j], 9) + ","
            s += _gbs(row.bytes, row.times[j]) + "\n"
    return s
