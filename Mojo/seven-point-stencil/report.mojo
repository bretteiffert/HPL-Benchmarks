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
    var bytes_per_elem: Int
    var nx: Int
    var ny: Int
    var nz: Int
    var block_x: Int
    var block_y: Int
    var block_z: Int
    var grid_x: Int
    var grid_y: Int
    var grid_z: Int
    var fetch_bytes: Int
    var write_bytes: Int
    var num_times: Int
    var timing: String
    var energy_enabled: Bool
    var mibibytes: Bool
    var header_notes: List[String]
    # results (timed launches)
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
    # correctness (untimed run)
    var max_abs_err: Float64
    var mean_abs_err: Float64
    var l2_rel_err: Float64
    var num_checked: Int
    var check_notes: List[String]

    def __init__(out self):
        self.implementation = "Mojo"
        self.backend = ""
        self.device = ""
        self.driver = ""
        self.precision = ""
        self.bytes_per_elem = 0
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.block_x = 0
        self.block_y = 0
        self.block_z = 0
        self.grid_x = 0
        self.grid_y = 0
        self.grid_z = 0
        self.fetch_bytes = 0
        self.write_bytes = 0
        self.num_times = 0
        self.timing = "device timestamps"
        self.energy_enabled = True
        self.mibibytes = False
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
        self.max_abs_err = 0.0
        self.mean_abs_err = 0.0
        self.l2_rel_err = 0.0
        self.num_checked = 0
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

    def traffic(self) -> Int:
        return self.fetch_bytes + self.write_bytes

    def bw_text(self, seconds: Float64) -> Float64:
        """Bandwidth for the text report: GB/s, or MiB/s with --mibibytes."""
        if self.mibibytes:
            return Float64(self.traffic()) / (1024.0 * 1024.0) / seconds
        return Float64(self.traffic()) * 1.0e-9 / seconds

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


def _xyz(x: Int, y: Int, z: Int) -> String:
    return thousands(x) + " × " + thousands(y) + " × " + thousands(z)


# ===----------------------------------------------------------------------=== #
# Text formatter
# ===----------------------------------------------------------------------=== #


def format_header(r: Report) -> String:
    var s = "Seven-point stencil (" + r.implementation + ")\n"
    s += _key("Backend", r.backend)
    s += _key("Device", r.device)
    s += _key("Driver", r.driver)
    s += _key("Precision", r.precision)
    var points = r.nx * r.ny * r.nz
    s += _key(
        "Grid",
        _xyz(r.nx, r.ny, r.nz)
        + " ("
        + thousands(points)
        + " points, "
        + _size(r, points * r.bytes_per_elem)
        + " per array, 2 arrays)",
    )
    s += _key("Block size", _xyz(r.block_x, r.block_y, r.block_z))
    s += _key("Launch grid", _xyz(r.grid_x, r.grid_y, r.grid_z) + " blocks")
    s += _key(
        "Traffic",
        _size(r, r.fetch_bytes)
        + " read, "
        + _size(r, r.write_bytes)
        + " written per launch (theoretical)",
    )
    s += _key("Iterations", thousands(r.num_times) + " (after 1 untimed run)")
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
        ljust("Routine", 12)
        + rjust("Bandwidth (" + unit + ")", bw_w)
        + rjust("Min (s)", 14)
        + rjust("Max (s)", 14)
        + rjust("Avg (s)", 14)
        + "\n"
    )
    s += ljust("Laplacian", 12)
    s += rjust(fmt_fixed(r.bw_text(r.avg_s), 3), bw_w)
    s += rjust(fmt_fixed(r.min_s, 8), 14)
    s += rjust(fmt_fixed(r.max_s, 8), 14)
    s += rjust(fmt_fixed(r.avg_s, 8), 14)
    s += "\n"
    s += (
        "  note: bandwidth = theoretical traffic / average time over "
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


def format_correctness(r: Report) -> String:
    var s = (
        "Correctness ("
        + r.precision
        + ", untimed run; exact Laplacian = 3 at "
        + thousands(r.num_checked)
        + " interior points)\n"
    )
    s += (
        ljust("Max abs err", 16)
        + rjust("Mean abs err", 16)
        + rjust("L2 rel err", 16)
        + "\n"
    )
    s += ljust(fmt_sci(r.max_abs_err, 3), 16)
    s += rjust(fmt_sci(r.mean_abs_err, 3), 16)
    s += rjust(fmt_sci(r.l2_rel_err, 3), 16)
    s += "\n"
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
        + String(r.nx)
        + ","
        + String(r.ny)
        + ","
        + String(r.nz)
        + ","
        + String(r.block_x)
        + ","
        + String(r.block_y)
        + ","
        + String(r.block_z)
        + ","
    )


def _gbs(bytes: Int, seconds: Float64) -> String:
    return fmt_fixed(Float64(bytes) * 1.0e-9 / seconds, 4)


def csv_header() -> String:
    return (
        CSV_PREFIX
        + "nx,ny,nz,block_x,block_y,block_z,num_times,routine,bw_gbs,"
        + "min_s,max_s,avg_s,energy_j_per_iter,avg_power_w"
    )


def csv_rows(r: Report) -> String:
    var s = _prefix(r)
    s += String(r.num_times) + ",Laplacian,"
    s += _gbs(r.traffic(), r.avg_s) + ","
    s += fmt_fixed(r.min_s, 9) + ","
    s += fmt_fixed(r.max_s, 9) + ","
    s += fmt_fixed(r.avg_s, 9) + ","
    if r.energy_enabled and r.energy_measured:
        s += fmt_fixed(r.j_per_iter(), 6) + "," + fmt_fixed(r.avg_power_w, 2)
    else:
        s += ","
    return s + "\n"


def csv_raw_header() -> String:
    return (
        CSV_PREFIX
        + "nx,ny,nz,block_x,block_y,block_z,routine,iteration,time_s,bw_gbs"
    )


def csv_raw_rows(r: Report) -> String:
    var s = String("")
    for j in range(len(r.times)):
        s += _prefix(r) + "Laplacian," + String(j + 1) + ","
        s += fmt_fixed(r.times[j], 9) + ","
        s += _gbs(r.traffic(), r.times[j]) + "\n"
    return s
