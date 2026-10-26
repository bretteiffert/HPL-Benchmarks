"""miniBUDE report: one struct filled by the run, three formatters that read
only from it (text, --csv summary, --csv-raw)."""

import math

DASH = "—"


class Report:
    def __init__(self):
        self.benchmark = "miniBUDE"
        self.implementation = "Triton"
        self.backend = ""
        self.device = ""
        self.driver = ""
        self.precision = ""          # float32 / float64
        self.deck = ""               # deck name (basename)
        self.natlig = 0
        self.natpro = 0
        self.ntypes = 0
        self.nposes = 0
        self.max_poses = 0
        self.ppwis = []
        self.wgsizes = []
        self.warp_size = 0
        self.warp_how = ""
        self.num_stages = ""
        self.iterations = 0
        self.warmup = 0
        self.rows = 8
        self.timing = "device timestamps"
        self.energy_enabled = True
        self.header_notes = []

        self.samples = []            # bude.Sample, in run order
        self.skip_notes = []         # configurations not run, and why
        self.ulp_labels = []         # histogram bucket labels


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


def best(r):
    return min(r.samples, key=lambda s: s.stats.mean)


# -- text ----------------------------------------------------------------------

def _key(name, value):
    return "  %-13s%s" % (name, value)


def text_header(r):
    if r.precision == "float32":
        prec = "float32 (checked against one untimed float64 pass)"
    else:
        prec = "float64 (the reference precision; no differential pass)"
    lines = [
        "%s (%s)" % (r.benchmark, r.implementation),
        _key("Backend", r.backend),
        _key("Device", r.device),
        _key("Driver", r.driver),
        _key("Precision", prec),
        _key("Deck", "%s (%s ligand atoms, %s protein atoms, %s types)"
             % (r.deck, count(r.natlig), count(r.natpro), count(r.ntypes))),
        _key("Poses", "%s of %s in the deck" % (count(r.nposes), count(r.max_poses))),
        _key("PPWI", ", ".join(str(p) for p in r.ppwis)),
        _key("WG size", "%s (warp size %d, from %s)"
             % (", ".join(str(w) for w in r.wgsizes), r.warp_size, r.warp_how)),
        _key("Num stages", r.num_stages),
        _key("Iterations", "%s (after %s warm-up)" % (count(r.iterations), count(r.warmup))),
        _key("Timing", r.timing),
        _key("Energy", "libkernelenergy (board power)" if r.energy_enabled
             else "disabled (--no-energy)"),
    ]
    lines += ["  note: %s" % n for n in r.header_notes]
    return "\n".join(lines) + "\n"


def text_results(r):
    fmt = "%-6s%8s%7s%8s%14s%14s%14s%14s%11s%12s%17s%15s%14s"
    lines = [fmt % ("PPWI", "WG size", "Warps", "Stages", "Min (s)", "Max (s)", "Avg (s)",
                    "Stddev (s)", "GFLOP/s", "GFInst/s", "GInteractions/s",
                    "Overhead (s)", "Context (s)")]
    for s in r.samples:
        st = s.stats
        lines.append(fmt % (s.ppwi, s.wgsize, s.num_warps, s.num_stages,
                            "%.8f" % st.min, "%.8f" % st.max, "%.8f" % st.mean,
                            "%.8f" % st.stdDev, "%.3f" % s.gflops, "%.3f" % s.gfinsts,
                            "%.3f" % s.ginteractions, "%.8f" % s.launch_overhead_s,
                            "%.8f" % s.context_s))
    lines.append("  note: GFLOP/s, GFInst/s and GInteractions/s use the average time; "
                 "Overhead is host wall time minus kernel time, averaged")
    b = best(r)
    lines.append("  note: best: ppwi %d, wgsize %d (lowest average, %.8f s)"
                 % (b.ppwi, b.wgsize, b.stats.mean))
    lines += ["  note: %s" % n for n in r.skip_notes]
    return "\n".join(lines) + "\n"


def text_energy(r):
    if not r.energy_enabled:
        return "Energy  disabled (--no-energy)\n"
    fmt = "%-8s%6s%9s%14s%14s%16s%12s%13s%10s"
    lines = [fmt % ("Energy", "PPWI", "WG size", "J/iter", "Total (Wh)", "Avg power (W)",
                    "Time (s)", "Iterations", "Samples")]
    few = []
    for s in r.samples:
        e = s.energy
        lines.append(fmt % ("board", s.ppwi, s.wgsize, "%.6f" % e.joules_per_iter,
                            "%.6f" % e.watt_hours, "%.2f" % e.avg_power_w, "%.6f" % e.seconds,
                            count(e.iterations), count(e.samples)))
        if e.samples < 2:
            few.append("ppwi %d, wgsize %d" % (s.ppwi, s.wgsize))
    if few:
        where = "all configurations" if len(few) == len(r.samples) else "; ".join(few)
        lines.append("  note: fewer than 2 power samples over the timed iterations (%s); "
                     "energy figures are unreliable (increase -i)" % where)
    return "\n".join(lines) + "\n"


def _collapse(notes_by_config, n_configs):
    """[(config label, note)] -> notes, identical ones merged."""
    out, seen = [], {}
    for label, note in notes_by_config:
        seen.setdefault(note, []).append(label)
    for note, labels in seen.items():
        where = "all configurations" if len(labels) == n_configs else "; ".join(labels)
        out.append("%s (%s)" % (note, where))
    return out


def text_correctness(r):
    diff = r.precision == "float32"
    title = ("Correctness (%s vs the float64 pass, and vs the deck's ref_energies.out)"
             % r.precision if diff else
             "Correctness (float64 vs the deck's ref_energies.out)")
    fmt = "%-6s%8s%13s%13s%13s%13s%14s%17s"
    lines = [title, fmt % ("PPWI", "WG size", "Ref L2 rel", "Ref max rel", "Ref max ULP",
                           "Deck L2 rel", "Deck max rel", "Deck beyond tol")]
    notes = []
    n = len(r.samples)
    for s in r.samples:
        ref, dk = s.vs_ref, s.vs_deck
        lines.append(fmt % (s.ppwi, s.wgsize,
                            sci(ref.l2RelNorm, 3) if ref else DASH,
                            sci(ref.maxRel, 3) if ref else DASH,
                            count(ref.maxUlp) if ref else DASH,
                            sci(dk.l2RelNorm, 3), sci(dk.maxRel, 3),
                            "%s/%s" % (count(dk.beyondTolerance), count(dk.compared))))
        label = "ppwi %d, wgsize %d" % (s.ppwi, s.wgsize)
        for m, what in ((ref, "float64 pass"), (dk, "deck")):
            if m is not None and m.nonFinite:
                notes.append((label, "%s non-finite energies skipped in the comparison "
                              "with the %s" % (count(m.nonFinite), what)))
    lines.append("  note: the deck's reference energies are stored at low precision; "
                 "'beyond tol' counts poses more than %s%% away" % "0.025")
    if not diff:
        lines.append("  note: float64 is the reference precision, so there is no "
                     "differential pass (Ref columns)")
    lines += ["  note: %s" % x for x in _collapse(notes, n)]

    if diff:
        lines.append("")
        hists = [tuple(s.vs_ref.ulpHistogram) for s in r.samples]
        if all(h == hists[0] for h in hists):
            lines.append("ULP histogram vs the float64 pass (all configurations)")
            lines.append("%-10s%14s" % ("ULPs", "Poses"))
            for label, c in zip(r.ulp_labels, hists[0]):
                if c:
                    lines.append("%-10s%14s" % (label, count(c)))
        else:
            heads = ["p%d/w%d" % (s.ppwi, s.wgsize) for s in r.samples]
            lines.append("ULP histogram vs the float64 pass (poses per configuration)")
            lines.append("%-10s" % "ULPs" + "".join("%12s" % h for h in heads))
            for i, label in enumerate(r.ulp_labels):
                if any(h[i] for h in hists):
                    lines.append("%-10s" % label + "".join("%12s" % count(h[i]) for h in hists))

    first = r.samples[0]
    rows = min(r.rows, len(first.energies))
    if rows > 0:
        lines.append("")
        lines.append("Energies (first %d pose%s, ppwi %d, wgsize %d)"
                     % (rows, "" if rows == 1 else "s", first.ppwi, first.wgsize))
        lines.append("%-10s%20s" % ("Pose", "Energy"))
        for i in range(rows):
            lines.append("%-10d%20s" % (i, sci(float(first.energies[i]), 9)))
    return "\n".join(lines) + "\n"


def text_body(r):
    return "\n" + text_results(r) + "\n" + text_energy(r) + "\n" + text_correctness(r)


def text_stderr_tail(r):
    """Notes, energy and correctness on stderr in the CSV modes."""
    notes = ["  note: best: ppwi %d, wgsize %d (lowest average, %.8f s)"
             % (best(r).ppwi, best(r).wgsize, best(r).stats.mean)]
    notes += ["  note: %s" % n for n in r.skip_notes]
    return "\n" + "\n".join(notes) + "\n\n" + text_energy(r) + "\n" + text_correctness(r)


# -- CSV -----------------------------------------------------------------------

_PREFIX = "implementation,backend,device,precision"


def _prefix(r):
    return ",".join(csv_field(x) for x in (r.implementation, r.backend, r.device, r.precision))


def csv_summary(r):
    lines = [_PREFIX + ",deck,natlig,natpro,nposes,ppwi,wgsize,iterations,min_s,max_s,avg_s,"
             "stddev_s,gflops,gfinsts,ginteractions,launch_overhead_s,context_s,"
             "energy_j_per_iter,avg_power_w,ref_l2_rel,ref_max_rel,ref_max_ulp,"
             "deck_l2_rel,deck_beyond_tol"]
    for s in r.samples:
        st, ref, dk = s.stats, s.vs_ref, s.vs_deck
        energy = ("%.9f,%.4f" % (s.energy.joules_per_iter, s.energy.avg_power_w)
                  if s.energy is not None else ",")
        refcols = ("%.6e,%.6e,%d" % (ref.l2RelNorm, ref.maxRel, ref.maxUlp)
                   if ref is not None else ",,")
        lines.append("%s,%s,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.4f,%.4f,%.4f,%.9f,%.9f,%s,%s,%.6e,%d" % (
            _prefix(r), csv_field(r.deck), r.natlig, r.natpro, r.nposes, s.ppwi, s.wgsize,
            r.iterations, st.min, st.max, st.mean, st.stdDev, s.gflops, s.gfinsts,
            s.ginteractions, s.launch_overhead_s, s.context_s, energy, refcols,
            dk.l2RelNorm, dk.beyondTolerance))
    return "\n".join(lines) + "\n"


def csv_raw(r):
    lines = [_PREFIX + ",deck,natlig,natpro,nposes,ppwi,wgsize,iteration,time_s"]
    for s in r.samples:
        head = "%s,%s,%d,%d,%d,%d,%d" % (_prefix(r), csv_field(r.deck), r.natlig, r.natpro,
                                         r.nposes, s.ppwi, s.wgsize)
        for i, t in enumerate(s.times, start=1):
            lines.append("%s,%d,%.9f" % (head, i, t))
    return "\n".join(lines) + "\n"
