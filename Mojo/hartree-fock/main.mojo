# Hartree-Fock (Mojo 1.2.0 / MAX 26.7): wiring, validation against device
# limits, and the run loop.
#
# The backend is fixed at compile time:
#     mojo run -D BACKEND=cuda main.mojo INPUT [args]
#     mojo run -D BACKEND=hip  main.mojo INPUT [args]

from std.collections import List, Optional
from std.sys import exit, stderr
from std.time import perf_counter
from max.gpu.host import DeviceContext

from backend import (
    BACKEND,
    backend_label,
    check_backend,
    list_devices,
    open_device,
    query_device,
)
from cli import Args, help_text, parse_arguments
from energy import BoardEnergy
from hf import (
    Deck,
    HFDevice,
    HostArrays,
    RefArrays,
    check_against_reference,
    pair_count,
    quartet_count,
)
from report import (
    Report,
    csv_header,
    csv_raw_header,
    csv_raw_rows,
    csv_rows,
    format_correctness,
    format_energy,
    format_header,
    format_text,
    thousands,
)


def precision_name[dtype: DType]() -> String:
    comptime if dtype == DType.float64:
        return "float64"
    else:
        return "float32"


def _compute_capability(ctx: DeviceContext) -> Int:
    try:
        return ctx.compute_capability()
    except:
        return -1


def emit(s: String, csv_mode: Bool):
    """Report text: stdout in text mode, stderr in the CSV modes."""
    if csv_mode:
        print(s, end="", file=stderr, flush=True)
    else:
        print(s, end="", flush=True)


def run[dtype: DType, legacy: Bool](args: Args) raises:
    var csv_mode = args.csv or args.csv_raw

    # ---- input (before touching the device) --------------------------------
    var deck = Deck(args.input)
    var natoms = deck.natoms
    var pairs = pair_count(natoms)
    var nnnn = quartet_count(natoms)
    var blocks_u = (nnnn + 1 + UInt64(args.block_size) - 1) // UInt64(
        args.block_size
    )

    # ---- device ------------------------------------------------------------
    var ctx = open_device(args.device_index)
    var info = query_device(ctx, args.device_index)

    # ---- validation before allocating or launching -------------------------
    if (
        info.max_threads_per_block > 0
        and args.block_size > info.max_threads_per_block
    ):
        raise Error(
            "block size "
            + String(args.block_size)
            + " exceeds the device limit of "
            + String(info.max_threads_per_block)
            + " threads per block"
        )
    if info.max_grid_x > 0 and blocks_u > UInt64(info.max_grid_x):
        raise Error(
            String(natoms)
            + " atoms need "
            + String(blocks_u)
            + " blocks of "
            + String(args.block_size)
            + ", above the device grid limit of "
            + String(info.max_grid_x)
            + " (the pinned block size is 256; use a smaller input)"
        )
    comptime if BACKEND == "cuda" and dtype == DType.float64:
        # The Fock accumulation uses float64 atomic add: sm_60 and newer.
        var cc = _compute_capability(ctx)
        if cc > 0 and cc < 60:
            raise Error(
                "this GPU (compute capability "
                + String(cc // 10)
                + "."
                + String(cc % 10)
                + ") has no float64 atomic add; use --float"
            )

    var do_check = args.run_check and (
        args.force_check or natoms <= args.max_check_atoms
    )

    # ---- report header (printed before the run starts) ---------------------
    var rep = Report()
    rep.backend = backend_label()
    rep.device = info.name
    rep.driver = info.driver
    rep.precision = precision_name[dtype]()
    rep.input = args.input
    rep.input_name = deck.name
    rep.natoms = natoms
    rep.ngauss = deck.ngauss
    rep.pairs = pairs
    rep.quartets = nnnn
    rep.block_size = args.block_size
    rep.blocks = Int(blocks_u)
    rep.sqrt_mode = String("legacy-sqrtf") if legacy else String("native")
    rep.num_times = args.num_times
    rep.energy_enabled = args.energy
    if do_check:
        rep.check_line = "float64 CPU reference"
    elif not args.run_check:
        rep.check_line = "skipped (--no-check)"
    else:
        rep.check_line = (
            "skipped ("
            + thousands(natoms)
            + " atoms > --max-check-atoms "
            + thousands(args.max_check_atoms)
            + "; use --check to force)"
        )
    for i in range(len(info.notes)):
        rep.header_notes.append(info.notes[i])
    rep.header_notes.append(
        "timed runs keep adding to the Fock matrix; only the untimed run is"
        " checked"
    )
    emit(format_header(rep) + "\n", csv_mode)

    # ---- host preparation (Schwarz bounds in the run precision) ------------
    var host = HostArrays[dtype, legacy](deck)

    # ---- energy profiler ---------------------------------------------------
    var prof = Optional[BoardEnergy](None)
    if args.energy:
        try:
            prof = Optional[BoardEnergy](BoardEnergy(args.device_index))
        except e:
            host.free()
            raise Error(
                "energy profiler failed to start: "
                + String(e)
                + " (try '--no-energy')"
            )

    # ---- run ---------------------------------------------------------------
    var dev = HFDevice[dtype, legacy](ctx, host, args.block_size)

    # Untimed run: warm-up, and the Fock matrix that is checked.
    dev.run_untimed()
    dev.read_fock(host)
    rep.e2_energy = host.e2_energy()

    var t_start = Float64(0)
    if prof:
        ctx.synchronize()
        prof.value().start()
        t_start = perf_counter()
    for _ in range(args.num_times):
        rep.times.append(dev.run_timed())
    if prof:
        ctx.synchronize()
        var reading = prof.value().stop()
        var host_s = perf_counter() - t_start
        prof.value().close()
        rep.energy_measured = True
        rep.joules = reading.joules
        rep.avg_power_w = reading.avg_power_w
        rep.samples = reading.samples
        rep.energy_iterations = args.num_times
        rep.energy_time_s = reading.elapsed_s
        if not (reading.elapsed_s > 0.0):
            rep.energy_time_s = host_s
            rep.energy_notes.append(
                "profiler returned no elapsed time; Time (s) is the host clock"
            )
        if reading.samples < 2:
            rep.energy_notes.append(
                "fewer than 2 power samples; energy figures are unreliable"
                " (use a larger input or -n)"
            )
    rep.finish_times()

    # ---- correctness (untimed run vs float64 CPU reference) ----------------
    if do_check:
        var ref_arrays = RefArrays(deck)
        var chk = check_against_reference(ref_arrays, host, rep.e2_energy)
        ref_arrays.free()
        rep.checked = True
        rep.e_gpu = chk.e_gpu
        rep.e_ref = chk.e_ref
        rep.rel_loss = chk.rel_loss
        rep.rel_inpipe = chk.rel_inpipe
        rep.maxd = chk.maxd
        rep.maxd_rel = chk.maxd_rel
        rep.rmsd = chk.rmsd
        rep.asym = chk.asym
        rep.e_sym = chk.e_sym
    elif not args.run_check:
        rep.check_notes.append("float64 CPU reference skipped (--no-check)")
    else:
        rep.check_notes.append(
            "float64 CPU reference skipped: "
            + thousands(natoms)
            + " atoms > --max-check-atoms "
            + thousands(args.max_check_atoms)
            + "; use --check to force"
        )
    comptime if legacy and dtype == DType.float64:
        rep.check_notes.append(
            "--legacy-sqrt evaluates the Boys-function square roots in"
            " float32, which limits float64 accuracy"
        )
    host.free()

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
        if args.legacy_sqrt:
            run[DType.float64, True](args)
        else:
            run[DType.float64, False](args)
    else:
        if args.legacy_sqrt:
            run[DType.float32, True](args)
        else:
            run[DType.float32, False](args)


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
