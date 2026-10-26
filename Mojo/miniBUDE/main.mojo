# miniBUDE (Mojo 1.2.0 / MAX 26.7): wiring, validation against device limits,
# and the configuration sweep.
#
# The backend is fixed at compile time:
#     mojo run -D BACKEND=cuda main.mojo [args]
#     mojo run -D BACKEND=hip  main.mojo [args]
#
# float64 kernels need float64 sin/cos on the GPU. Mojo 1.2 rejects them on
# NVIDIA at compile time and has no float64 lowering on AMD, so they are only
# compiled when you add -D FP64. Without it, --double is a clear error and
# float32 runs take their float64 reference from --golden.

from std.collections import List
from std.sys import exit, stderr
from std.sys.defines import is_defined

from backend import (
    backend_label,
    check_backend,
    list_devices,
    open_device,
    query_device,
)
from bude import evaluate, fasten_dispatch
from cli import Args, help_text, parse_arguments
from deck import load_deck, read_numbers
from report import (
    Report,
    csv_header,
    csv_raw_header,
    csv_raw_rows,
    csv_rows,
    fmt_sci,
    format_correctness,
    format_energy,
    format_header,
    format_text,
    thousands,
)

comptime FP64_ENABLED = is_defined["FP64"]()


def precision_name[dtype: DType]() -> String:
    comptime if dtype == DType.float64:
        return "float64"
    else:
        return "float32"


def emit(s: String, csv_mode: Bool):
    """Report text: stdout in text mode, stderr in the CSV modes."""
    if csv_mode:
        print(s, end="", file=stderr, flush=True)
    else:
        print(s, end="", flush=True)


def write_energies(path: String, energies: List[Float64]) raises:
    var text = String("")
    for i in range(len(energies)):
        text += fmt_sci(energies[i], 16) + "\n"
    try:
        with open(path, "w") as f:
            f.write(text)
    except e:
        raise Error("cannot write " + path + ": " + String(e))


def run[dtype: DType](args: Args) raises:
    var csv_mode = args.csv or args.csv_raw

    # ---- input (before touching the device) --------------------------------
    var deck = load_deck(args.deck_dir, args.nposes)
    var n = deck.nposes()

    var golden = List[Float64]()
    var has_golden = args.golden_path.byte_length() > 0
    if has_golden:
        golden = read_numbers(args.golden_path)
        if len(golden) < n:
            raise Error(
                "golden file "
                + args.golden_path
                + " has "
                + String(len(golden))
                + " energies, fewer than the "
                + String(n)
                + " poses being computed"
            )

    # ---- device ------------------------------------------------------------
    var ctx = open_device(args.device_index)
    var info = query_device(ctx, args.device_index)

    # ---- validation and configuration list, before any launch ---------------
    var rep = Report()
    var cfg_ppwi = List[Int]()
    var cfg_wg = List[Int]()
    for pi in range(len(args.ppwis)):
        for wi in range(len(args.wgsizes)):
            var ppwi = args.ppwis[pi]
            var wg = args.wgsizes[wi]
            if info.max_threads_per_block > 0 and wg > info.max_threads_per_block:
                raise Error(
                    "work-group size "
                    + String(wg)
                    + " exceeds the device limit of "
                    + String(info.max_threads_per_block)
                    + " threads per block"
                )
            var per_block = ppwi * wg
            if n < per_block or n % per_block != 0:
                rep.result_notes.append(
                    "skipped ppwi "
                    + String(ppwi)
                    + ", wgsize "
                    + String(wg)
                    + ": "
                    + thousands(per_block)
                    + " poses per work-group do not divide "
                    + thousands(n)
                    + " poses"
                )
                continue
            var blocks = ((n + ppwi - 1) // ppwi + wg - 1) // wg
            if info.max_grid_x > 0 and blocks > info.max_grid_x:
                raise Error(
                    "ppwi "
                    + String(ppwi)
                    + ", wgsize "
                    + String(wg)
                    + " needs "
                    + String(blocks)
                    + " blocks, above the device grid limit of "
                    + String(info.max_grid_x)
                )
            cfg_ppwi.append(ppwi)
            cfg_wg.append(wg)
    if len(cfg_ppwi) == 0:
        raise Error(
            "no runnable configuration: no ppwi x wgsize divides the "
            + String(n)
            + " poses"
        )

    # ---- report header (printed before the run starts) ---------------------
    rep.backend = backend_label()
    rep.device = info.name
    rep.driver = info.driver
    rep.precision = precision_name[dtype]()
    rep.deck_dir = deck.dir
    rep.deck_name = deck.name
    rep.natlig = deck.natlig()
    rep.natpro = deck.natpro()
    rep.nposes = n
    rep.max_poses = deck.max_poses
    rep.ppwis = args.ppwis.copy()
    rep.wgsizes = args.wgsizes.copy()
    rep.iterations = args.iterations
    rep.warmup = args.warmup
    rep.energy_enabled = args.energy
    rep.rows = args.rows

    comptime if dtype == DType.float64:
        rep.reference_line = "not needed (float64 run)"
    else:
        if has_golden:
            rep.reference_line = "float64 golden file " + args.golden_path
        else:
            comptime if FP64_ENABLED:
                rep.reference_line = "float64 pass on the device (untimed)"
            else:
                rep.reference_line = "none (pass --golden FILE)"
    for i in range(len(info.notes)):
        rep.header_notes.append(info.notes[i])
    emit(format_header(rep) + "\n", csv_mode)

    # ---- float64 reference pass (only compiled with -D FP64) ---------------
    var reference = golden^
    var has_reference = has_golden
    comptime if dtype == DType.float32 and FP64_ENABLED:
        if not has_golden:
            var ref_sample = fasten_dispatch[DType.float64](
                ctx, deck, cfg_wg[0], cfg_ppwi[0], 1, 0, False, args.device_index
            )
            reference = ref_sample.energies.copy()
            has_reference = True

    # ---- the sweep ---------------------------------------------------------
    for k in range(len(cfg_ppwi)):
        var sample = fasten_dispatch[dtype](
            ctx,
            deck,
            cfg_wg[k],
            cfg_ppwi[k],
            args.iterations,
            args.warmup,
            args.energy,
            args.device_index,
        )
        rep.results.append(evaluate(deck, sample, reference, has_reference))

    if args.out_path.byte_length() > 0:
        write_energies(args.out_path, rep.results[0].sample.energies)
        rep.check_notes.append(
            "energies of ppwi "
            + String(cfg_ppwi[0])
            + ", wgsize "
            + String(cfg_wg[0])
            + " written to "
            + args.out_path
        )

    comptime if dtype == DType.float64:
        rep.check_notes.append(
            "float64 run: it is the reference, so only the deck comparison"
            " applies"
        )
    else:
        if not has_reference:
            rep.check_notes.append(
                "no float64 reference: pass --golden FILE (for example the"
                " CUDA build's `bude -o FILE` in float64), or compile with"
                " -D FP64 on a Mojo that supports float64 sin/cos on the GPU"
            )

    # ---- output ------------------------------------------------------------
    if args.csv:
        print(csv_header(), flush=True)
        print(csv_rows(rep), end="", flush=True)
        print(format_energy(rep), end="", file=stderr, flush=True)
        print("", file=stderr, flush=True)
        print(format_correctness(rep), end="", file=stderr, flush=True)
    elif args.csv_raw:
        print(csv_raw_header(), flush=True)
        print(csv_raw_rows(rep), end="", flush=True)
        print(format_energy(rep), end="", file=stderr, flush=True)
        print("", file=stderr, flush=True)
        print(format_correctness(rep), end="", file=stderr, flush=True)
    else:
        print(format_text(rep), end="", flush=True)


def run_main() raises:
    var args = parse_arguments()
    if args.help:
        print(help_text(), end="", flush=True)
        return
    if args.list_devices:
        list_devices()
        return
    if args.precision == "float64":
        comptime if FP64_ENABLED:
            run[DType.float64](args)
        else:
            raise Error(
                "--double needs float64 sin/cos on the GPU, which Mojo 1.2 does"
                " not provide (a compile-time error on NVIDIA, no lowering on"
                " AMD). The float64 path is only compiled with -D FP64"
                " (try '--float')"
            )
    else:
        run[DType.float32](args)


def main():
    check_backend()
    try:
        run_main()
    except e:
        var msg = String(e)
        if "(try '" in msg:
            print("error: " + msg, file=stderr, flush=True)
        else:
            print("error: " + msg + " (try '--help')", file=stderr, flush=True)
        exit(1)
