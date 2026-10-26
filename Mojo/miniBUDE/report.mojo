# Report struct and its three formatters (text, --csv, --csv-raw).
#
# The run fills one Report; the formatters only read from it. Nothing here
# touches the device. Stream rules (applied by main):
#   text mode -> everything on stdout;
#   CSV modes -> only CSV on stdout; header, energy, correctness on stderr.

from std.collections import List
from std.math import floor, log10
from std.utils.numerics import isnan, isinf

from bude import Result, ULP_NBUCKETS, ulp_bucket_label


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
    var deck_dir: String
    var deck_name: String
    var natlig: Int
    var natpro: Int
    var nposes: Int
    var max_poses: Int
    var ppwis: List[Int]
    var wgsizes: List[Int]
    var iterations: Int
    var warmup: Int
    var timing: String
    var energy_enabled: Bool
    var reference_line: String
    var header_notes: List[String]
    # results, one per configuration that ran
    var results: List[Result]
    var result_notes: List[String]
    # correctness
    var rows: Int  # energies to print
    var check_notes: List[String]

    def __init__(out self):
        self.implementation = "Mojo"
        self.backend = ""
        self.device = ""
        self.driver = ""
        self.precision = ""
        self.deck_dir = ""
        self.deck_name = ""
        self.natlig = 0
        self.natpro = 0
        self.nposes = 0
        self.max_poses = 0
        self.ppwis = List[Int]()
        self.wgsizes = List[Int]()
        self.iterations = 0
        self.warmup = 0
        self.timing = "device timestamps"
        self.energy_enabled = True
        self.reference_line = ""
        self.header_notes = List[String]()
        self.results = List[Result]()
        self.result_notes = List[String]()
        self.rows = 8
        self.check_notes = List[String]()

    def best(self) -> Int:
        """Index of the configuration with the lowest average time."""
        var b = 0
        for k in range(len(self.results)):
            if self.results[k].stats.mean < self.results[b].stats.mean:
                b = k
        return b


def _key(k: String, v: String) -> String:
    return "  " + ljust(k, 13) + v + "\n"


def _notes(notes: List[String]) -> String:
    var s = String("")
    for i in range(len(notes)):
        s += "  note: " + notes[i] + "\n"
    return s


def _ints(v: List[Int]) -> String:
    var s = String("")
    for i in range(len(v)):
        if i > 0:
            s += ", "
        s += String(v[i])
    return s


def _cfg(r: Result) -> String:
    return ljust(String(r.sample.ppwi), 6) + ljust(String(r.sample.wgsize), 6)


def _same_energies(r: Report) -> Bool:
    for k in range(1, len(r.results)):
        ref a = r.results[0].sample.energies
        ref b = r.results[k].sample.energies
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
    return True


# ===----------------------------------------------------------------------=== #
# Text formatter
# ===----------------------------------------------------------------------=== #


def format_header(r: Report) -> String:
    var s = "miniBUDE (" + r.implementation + ")\n"
    s += _key("Backend", r.backend)
    s += _key("Device", r.device)
    s += _key("Driver", r.driver)
    s += _key("Precision", r.precision)
    s += _key("Deck", r.deck_dir + " (" + r.deck_name + ")")
    var poses = thousands(r.nposes) + " poses"
    if r.nposes != r.max_poses:
        poses += " (of " + thousands(r.max_poses) + ")"
    s += _key(
        "Problem",
        thousands(r.natlig)
        + " ligand atoms, "
        + thousands(r.natpro)
        + " protein atoms, "
        + poses,
    )
    s += _key("PPWI", _ints(r.ppwis))
    s += _key("Work-group", _ints(r.wgsizes))
    s += _key("Groups", thousands(r.nposes // (r.ppwis[0] * r.wgsizes[0])))
    s += _key(
        "Iterations",
        thousands(r.iterations) + " (after " + thousands(r.warmup) + " warm-up)",
    )
    s += _key("Timing", r.timing)
    if r.energy_enabled:
        s += _key("Energy", "libkernelenergy (board power)")
    else:
        s += _key("Energy", "disabled (--no-energy)")
    s += _key("Reference", r.reference_line)
    s += _notes(r.header_notes)
    return s


def format_results(r: Report) -> String:
    var s = (
        ljust("PPWI", 6)
        + ljust("WG", 6)
        + rjust("Min (s)", 14)
        + rjust("Max (s)", 14)
        + rjust("Avg (s)", 14)
        + rjust("Stddev (s)", 14)
        + "\n"
    )
    for k in range(len(r.results)):
        ref x = r.results[k]
        s += _cfg(x)
        s += rjust(fmt_fixed(x.stats.min, 8), 14)
        s += rjust(fmt_fixed(x.stats.max, 8), 14)
        s += rjust(fmt_fixed(x.stats.mean, 8), 14)
        s += rjust(fmt_fixed(x.stats.std_dev, 8), 14)
        s += "\n"
    s += "\n"
    s += (
        ljust("PPWI", 6)
        + ljust("WG", 6)
        + rjust("GFLOP/s", 12)
        + rjust("GFInst/s", 12)
        + rjust("GInteract/s", 13)
        + rjust("Launch (s)", 14)
        + rjust("Context (s)", 14)
        + "\n"
    )
    for k in range(len(r.results)):
        ref x = r.results[k]
        s += _cfg(x)
        s += rjust(fmt_fixed(x.gflops, 3), 12)
        s += rjust(fmt_fixed(x.ginsts, 3), 12)
        s += rjust(fmt_fixed(x.interactions_per_sec / 1e9, 3), 13)
        s += rjust(fmt_fixed(x.launch_overhead_s, 8), 14)
        s += rjust(fmt_fixed(x.sample.context_s, 8), 14)
        s += "\n"
    s += _notes(r.result_notes)
    if len(r.results) > 1:
        ref b = r.results[r.best()]
        s += (
            "  note: best: ppwi "
            + String(b.sample.ppwi)
            + ", wgsize "
            + String(b.sample.wgsize)
            + " (lowest avg, "
            + fmt_fixed(b.stats.mean, 8)
            + " s, "
            + fmt_fixed(b.gflops, 3)
            + " GFLOP/s)\n"
        )
    s += (
        "  note: rates use the average time; Launch = host wall time minus"
        " device time per launch; Context = buffer creation and upload\n"
    )
    return s


def format_energy(r: Report) -> String:
    if not r.energy_enabled:
        return ljust("Energy", 8) + "disabled (--no-energy)\n"
    var s = (
        ljust("PPWI", 6)
        + ljust("WG", 6)
        + rjust("J/iter", 12)
        + rjust("Total (Wh)", 14)
        + rjust("Avg power (W)", 16)
        + rjust("Time (s)", 12)
        + rjust("Iterations", 13)
        + rjust("Samples", 10)
        + "\n"
    )
    var few = 0
    for k in range(len(r.results)):
        ref x = r.results[k].sample
        s += _cfg(r.results[k])
        s += rjust(fmt_fixed(x.joules / Float64(len(x.kernel_s)), 6), 12)
        s += rjust(fmt_fixed(x.joules / 3600.0, 6), 14)
        s += rjust(fmt_fixed(x.avg_power_w, 2), 16)
        s += rjust(fmt_fixed(x.energy_time_s, 6), 12)
        s += rjust(thousands(len(x.kernel_s)), 13)
        s += rjust(thousands(x.samples), 10)
        s += "\n"
        if x.samples < 2:
            few += 1
    if few == len(r.results):
        s += (
            "  note: fewer than 2 power samples (all configurations); energy"
            " figures are unreliable (use a larger -i)\n"
        )
    elif few > 0:
        s += (
            "  note: fewer than 2 power samples for "
            + String(few)
            + " configuration(s); their energy figures are unreliable\n"
        )
    return s


def format_correctness(r: Report) -> String:
    var s = String("")
    var has_ref = len(r.results) > 0 and r.results[0].has_reference
    if has_ref:
        s += "Correctness (" + r.precision + " vs float64 reference)\n"
        s += (
            ljust("PPWI", 6)
            + ljust("WG", 6)
            + rjust("L2 rel", 12)
            + rjust("RMS abs", 12)
            + rjust("Mean signed", 13)
            + rjust("Max abs", 12)
            + rjust("Max rel", 12)
            + rjust("Max ULP", 10)
            + "\n"
        )
        for k in range(len(r.results)):
            ref m = r.results[k].vs_reference
            s += _cfg(r.results[k])
            s += rjust(fmt_sci(m.l2_rel_norm, 3), 12)
            s += rjust(fmt_sci(m.rms_abs, 3), 12)
            s += rjust(fmt_sci(m.mean_signed, 3), 13)
            s += rjust(fmt_sci(m.max_abs, 3), 12)
            s += rjust(fmt_sci(m.max_rel, 3), 12)
            s += rjust(String(m.max_ulp), 10)
            s += "\n"
        # Indices, non-finite counts and the ULP histogram, collapsed when
        # every configuration produced the same energies.
        var same = _same_energies(r)
        var n_blocks = 1 if same else len(r.results)
        for k in range(n_blocks):
            ref m = r.results[k].vs_reference
            var who = String("all configurations") if same else (
                "ppwi "
                + String(r.results[k].sample.ppwi)
                + ", wgsize "
                + String(r.results[k].sample.wgsize)
            )
            s += (
                "  note: max abs at pose "
                + String(m.max_abs_idx)
                + ", max rel at pose "
                + String(m.max_rel_idx)
                + ", max ULP at pose "
                + String(m.max_ulp_idx)
                + "; "
                + thousands(m.compared)
                + " compared, "
                + thousands(m.non_finite)
                + " non-finite ("
                + who
                + ")\n"
            )
        s += "\n" + ljust("ULP distance", 14)
        for k in range(n_blocks):
            if same:
                s += rjust("Poses", 12)
            else:
                s += rjust(
                    String(r.results[k].sample.ppwi)
                    + "/"
                    + String(r.results[k].sample.wgsize),
                    12,
                )
        s += "\n"
        for b in range(ULP_NBUCKETS):
            var seen = False
            for k in range(n_blocks):
                if r.results[k].vs_reference.ulp_histogram[b] > 0:
                    seen = True
            if not seen:
                continue
            s += ljust(ulp_bucket_label(b), 14)
            for k in range(n_blocks):
                s += rjust(thousands(r.results[k].vs_reference.ulp_histogram[b]), 12)
            s += "\n"
        if not same:
            s += "  note: ULP columns are ppwi/wgsize\n"
        s += "\n"
    else:
        s += "Correctness (" + r.precision + ")\n"
    s += _notes(r.check_notes)

    s += (
        "Deck comparison (informational; ref_energies.out is low precision,"
        " tolerance 0.025%)\n"
    )
    s += (
        ljust("PPWI", 6)
        + ljust("WG", 6)
        + rjust("L2 rel", 12)
        + rjust("Max rel", 12)
        + rjust("Beyond tolerance", 20)
        + "\n"
    )
    for k in range(len(r.results)):
        ref d = r.results[k].vs_deck
        s += _cfg(r.results[k])
        s += rjust(fmt_sci(d.l2_rel_norm, 3), 12)
        s += rjust(fmt_sci(d.max_rel, 3), 12)
        s += rjust(thousands(d.beyond_tolerance) + "/" + thousands(d.compared), 20)
        s += "\n"

    if r.rows > 0 and len(r.results) > 0:
        var same = _same_energies(r)
        var n_cols = 1 if same else len(r.results)
        var rows = min(r.rows, len(r.results[0].sample.energies))
        s += "\nEnergies (first " + String(rows) + " poses"
        s += String("; identical for all configurations)\n") if same else String(
            "; columns are ppwi/wgsize)\n"
        )
        s += ljust("Pose", 8)
        for k in range(n_cols):
            if same:
                s += rjust("Energy", 18)
            else:
                s += rjust(
                    String(r.results[k].sample.ppwi)
                    + "/"
                    + String(r.results[k].sample.wgsize),
                    18,
                )
        s += "\n"
        for i in range(rows):
            s += ljust(String(i), 8)
            for k in range(n_cols):
                s += rjust(fmt_fixed(r.results[k].sample.energies[i], 6), 18)
            s += "\n"
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


def _prefix(r: Report, x: Result) -> String:
    return (
        csv_field(r.implementation)
        + ","
        + csv_field(r.backend)
        + ","
        + csv_field(r.device)
        + ","
        + csv_field(r.precision)
        + ","
        + csv_field(r.deck_name)
        + ","
        + String(r.natlig)
        + ","
        + String(r.natpro)
        + ","
        + String(r.nposes)
        + ","
        + String(x.sample.ppwi)
        + ","
        + String(x.sample.wgsize)
        + ","
        + String(r.nposes // (x.sample.ppwi * x.sample.wgsize))
        + ","
        + String(r.warmup)
        + ","
    )


def csv_header() -> String:
    return (
        CSV_PREFIX
        + "deck,natlig,natpro,nposes,ppwi,wgsize,groups,warmup,iterations,min_s,max_s,"
        + "avg_s,stddev_s,gflops,gfinsts,ginteractions,launch_overhead_s,"
        + "context_s,energy_j_per_iter,avg_power_w,ref_l2_rel,ref_max_rel,"
        + "ref_max_ulp,deck_l2_rel,deck_beyond_tol"
    )


def csv_rows(r: Report) -> String:
    var s = String("")
    for k in range(len(r.results)):
        ref x = r.results[k]
        s += _prefix(r, x)
        s += String(len(x.sample.kernel_s)) + ","
        s += fmt_fixed(x.stats.min, 9) + ","
        s += fmt_fixed(x.stats.max, 9) + ","
        s += fmt_fixed(x.stats.mean, 9) + ","
        s += fmt_fixed(x.stats.std_dev, 9) + ","
        s += fmt_fixed(x.gflops, 4) + ","
        s += fmt_fixed(x.ginsts, 4) + ","
        s += fmt_fixed(x.interactions_per_sec / 1e9, 4) + ","
        s += fmt_fixed(x.launch_overhead_s, 9) + ","
        s += fmt_fixed(x.sample.context_s, 9) + ","
        if r.energy_enabled and x.sample.energy_measured:
            s += fmt_fixed(x.sample.joules / Float64(len(x.sample.kernel_s)), 6)
            s += "," + fmt_fixed(x.sample.avg_power_w, 2) + ","
        else:
            s += ",,"
        if x.has_reference:
            s += fmt_sci(x.vs_reference.l2_rel_norm, 6) + ","
            s += fmt_sci(x.vs_reference.max_rel, 6) + ","
            s += String(x.vs_reference.max_ulp) + ","
        else:
            s += ",,,"
        s += fmt_sci(x.vs_deck.l2_rel_norm, 6) + ","
        s += String(x.vs_deck.beyond_tolerance) + "\n"
    return s


def csv_raw_header() -> String:
    return (
        CSV_PREFIX + "deck,natlig,natpro,nposes,ppwi,wgsize,groups,warmup,iteration,time_s"
    )


def csv_raw_rows(r: Report) -> String:
    var s = String("")
    for k in range(len(r.results)):
        ref x = r.results[k]
        for j in range(len(x.sample.kernel_s)):
            s += _prefix(r, x) + String(j + 1) + ","
            s += fmt_fixed(x.sample.kernel_s[j], 9) + "\n"
    return s
