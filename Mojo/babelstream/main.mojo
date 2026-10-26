# BabelStream (Mojo 1.2.0 / MAX 26.7): wiring, validation against device
# limits, and the run loop.
#
# The backend is fixed at compile time:
#     mojo run -D BACKEND=cuda main.mojo [args]
#     mojo run -D BACKEND=hip  main.mojo [args]

from std.collections import List, Optional
from std.sys import exit, size_of, stderr
from std.time import perf_counter

from backend import (
    backend_label,
    check_backend,
    compute_unit_label,
    list_devices,
    open_device,
    query_device,
)
from cli import Args, help_text, parse_arguments
from energy import BoardEnergy
from report import (
    CheckRow,
    Report,
    Row,
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
from stream import MojoStream, check_solution


def precision_name[dtype: DType]() -> String:
    comptime if dtype == DType.float64:
        return "float64"
    elif dtype == DType.float32:
        return "float32"
    else:
        return "float16"


def min_normal[dtype: DType]() -> Float64:
    comptime if dtype == DType.float64:
        return 2.2250738585072014e-308
    elif dtype == DType.float32:
        return 1.1754943508222875e-38
    else:
        return 6.103515625e-05


def max_finite[dtype: DType]() -> Float64:
    comptime if dtype == DType.float64:
        return 1.7976931348623157e308
    elif dtype == DType.float32:
        return 3.4028234663852886e38
    else:
        return 65504.0


def max_finite_text[dtype: DType]() -> String:
    comptime if dtype == DType.float16:
        return "65,504"
    else:
        return fmt_sci(max_finite[dtype](), 3)


def emit(s: String, csv_mode: Bool):
    """Report text: stdout in text mode, stderr in the CSV modes."""
    if csv_mode:
        print(s, end="", file=stderr, flush=True)
    else:
        print(s, end="", flush=True)


def run_bs[dtype: DType, tbsize: Int](args: Args) raises:
    var csv_mode = args.csv or args.csv_raw
    var n = args.array_size

    # ---- device ------------------------------------------------------------
    var ctx = open_device(args.device_index)
    var info = query_device(ctx, args.device_index)
    if not info.sm_known:
        raise Error(
            compute_unit_label()
            + " count is unavailable; the pinned Dot configuration requires 4 × SM/CU groups"
        )

    # ---- validation before allocating or launching -------------------------
    if info.max_threads_per_block > 0 and tbsize > info.max_threads_per_block:
        raise Error(
            "block size "
            + String(tbsize)
            + " exceeds the device limit of "
            + String(info.max_threads_per_block)
            + " threads per block"
        )
    if info.warp_size > 0 and tbsize % info.warp_size != 0:
        raise Error(
            "block size "
            + String(tbsize)
            + " must be a multiple of the warp size ("
            + String(info.warp_size)
            + ")"
        )
    if n < tbsize or n % tbsize != 0:
        raise Error(
            "array size "
            + String(n)
            + " must be a multiple of the block size "
            + String(tbsize)
            + " (the kernels have no bounds check)"
        )
    var dot_groups = 4 * info.sm_count
    if info.max_grid_x > 0 and n // tbsize > info.max_grid_x:
        raise Error(
            "array size "
            + String(n)
            + " needs "
            + String(n // tbsize)
            + " blocks, above the device grid limit of "
            + String(info.max_grid_x)
        )
    if info.max_grid_x > 0 and dot_groups > info.max_grid_x:
        raise Error(
            "dot needs "
            + String(dot_groups)
            + " groups, above the device grid limit of "
            + String(info.max_grid_x)
        )

    # ---- report header (printed before the run starts) ---------------------
    var rep = Report()
    rep.backend = backend_label()
    rep.device = info.name
    rep.driver = info.driver
    rep.precision = precision_name[dtype]()
    rep.bytes_per_elem = size_of[dtype]()
    rep.array_size = n
    rep.block_size = tbsize
    rep.dot_groups = dot_groups
    rep.sm_count = info.sm_count
    rep.sm_label = compute_unit_label()
    rep.sm_known = info.sm_known
    rep.num_times = args.num_times
    rep.timing = "device timestamps"
    rep.energy_enabled = args.energy
    rep.mibibytes = args.mibibytes
    for i in range(len(info.notes)):
        rep.header_notes.append(info.notes[i])
    rep.header_notes.append(
        "Dot time includes copying the partial sums to the host"
    )
    emit(format_header(rep) + "\n", csv_mode)

    var bpe = rep.bytes_per_elem
    rep.rows.append(Row("Init", 3 * bpe * n, True))
    rep.rows.append(Row("Copy", 2 * bpe * n, False))
    rep.rows.append(Row("Mul", 2 * bpe * n, False))
    rep.rows.append(Row("Add", 3 * bpe * n, False))
    rep.rows.append(Row("Triad", 3 * bpe * n, False))
    rep.rows.append(Row("Dot", 2 * bpe * n, False))

    # ---- energy profiler ---------------------------------------------------
    var prof = Optional[BoardEnergy](None)
    if args.energy:
        try:
            prof = Optional[BoardEnergy](BoardEnergy(args.device_index))
        except e:
            raise Error(
                "energy profiler failed to start: "
                + String(e)
                + " (try '--no-energy')"
            )

    # ---- run ---------------------------------------------------------------
    var bench = MojoStream[dtype, tbsize](ctx, n, dot_groups)

    bench.init_arrays()
    rep.rows[0].times.append(bench.get_time_taken())

    var dot_sum = Float64(0)
    var t_start = Float64(0)
    for it in range(args.num_times):
        # Iteration 0 is the warm-up: excluded from stats, CSV and energy.
        if it == 1 and prof:
            ctx.synchronize()
            prof.value().start()
            t_start = perf_counter()
        var timed = it > 0
        bench.copy()
        if timed:
            rep.rows[1].times.append(bench.get_time_taken())
        bench.mul()
        if timed:
            rep.rows[2].times.append(bench.get_time_taken())
        bench.add()
        if timed:
            rep.rows[3].times.append(bench.get_time_taken())
        bench.triad()
        if timed:
            rep.rows[4].times.append(bench.get_time_taken())
        dot_sum = bench.dot()
        if timed:
            rep.rows[5].times.append(bench.get_time_taken())

    if prof:
        ctx.synchronize()
        var reading = prof.value().stop()
        var host_s = perf_counter() - t_start
        prof.value().close()
        rep.energy_measured = True
        rep.joules = reading.joules
        rep.avg_power_w = reading.avg_power_w
        rep.samples = reading.samples
        rep.energy_iterations = args.num_times - 1
        rep.energy_time_s = reading.elapsed_s
        if not (reading.elapsed_s > 0.0):
            rep.energy_time_s = host_s
            rep.energy_notes.append(
                "profiler returned no elapsed time; Time (s) is the host clock"
            )
        if reading.samples < 2:
            rep.energy_notes.append(
                "fewer than 2 power samples; energy figures are unreliable"
                " (use a larger -s or -n)"
            )

    for i in range(len(rep.rows)):
        rep.rows[i].finish()

    # ---- correctness -------------------------------------------------------
    var ha = List[Scalar[dtype]](unsafe_uninit_length=n)
    var hb = List[Scalar[dtype]](unsafe_uninit_length=n)
    var hc = List[Scalar[dtype]](unsafe_uninit_length=n)
    bench.read_arrays(ha, hb, hc)
    var chk = check_solution[dtype](args.num_times, ha, hb, hc, dot_sum, n)

    rep.checks.append(
        CheckRow("a", chk.gold.a, chk.chk_a, chk.max_a, True)
    )
    rep.checks.append(
        CheckRow("b", chk.gold.b, chk.chk_b, chk.max_b, True)
    )
    rep.checks.append(
        CheckRow("c", chk.gold.c, chk.chk_c, chk.max_c, True)
    )
    rep.checks.append(
        CheckRow("dot", chk.expected_dot, chk.dot_err, 0.0, False)
    )
    rep.actual_dot = chk.actual_dot

    var tiny = min_normal[dtype]()
    var prec = rep.precision
    var under = String("")
    if chk.gold.a64 != 0.0 and abs(chk.gold.a64) < tiny:
        under += "a"
    if chk.gold.b64 != 0.0 and abs(chk.gold.b64) < tiny:
        under += String(", b") if under.byte_length() > 0 else String("b")
    if chk.gold.c64 != 0.0 and abs(chk.gold.c64) < tiny:
        under += String(", c") if under.byte_length() > 0 else String("c")
    if under.byte_length() > 0:
        rep.check_notes.append(
            "expected "
            + under
            + " underflow "
            + prec
            + " (smallest normal "
            + fmt_sci(tiny, 3)
            + ") after "
            + thousands(args.num_times)
            + " iterations, so their errors are not meaningful"
        )
    var product = abs(chk.gold.a64 * chk.gold.b64)
    if product != 0.0 and product < tiny:
        rep.check_notes.append(
            "expected element product "
            + fmt_sci(product, 3)
            + " underflows "
            + prec
            + ", so the dot relative error is not meaningful"
        )
    var dot64 = chk.gold.a64 * chk.gold.b64 * Float64(n)
    if abs(dot64) > max_finite[dtype]():
        rep.check_notes.append(
            "expected dot "
            + fmt_sci(dot64, 3)
            + " overflows "
            + prec
            + " (max "
            + max_finite_text[dtype]()
            + "), so the dot relative error is not meaningful"
        )

    # ---- output ------------------------------------------------------------
    if args.csv:
        print(csv_header())
        print(csv_rows(rep), end="")
        print(format_energy(rep), end="", file=stderr)
        print("", file=stderr)
        print(format_correctness(rep), end="", file=stderr)
    elif args.csv_raw:
        print(csv_raw_header())
        print(csv_raw_rows(rep), end="")
        print(format_energy(rep), end="", file=stderr)
        print("", file=stderr)
        print(format_correctness(rep), end="", file=stderr)
    else:
        print(format_text(rep), end="", flush=True)


def run[dtype: DType](args: Args) raises:
    run_bs[dtype, 256](args)


def run_main() raises:
    var args = parse_arguments()
    if args.help:
        print(help_text(), end="", flush=True)
        return
    if args.list_devices:
        list_devices()
        return
    if args.precision == "float64":
        run[DType.float64](args)
    elif args.precision == "float32":
        run[DType.float32](args)
    else:
        run[DType.float16](args)


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
