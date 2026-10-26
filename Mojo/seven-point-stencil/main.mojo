# Seven-point stencil (Mojo 1.2.0 / MAX 26.7): wiring, validation against
# device limits, and the run loop.
#
# The backend is fixed at compile time:
#     mojo run -D BACKEND=cuda main.mojo [args]
#     mojo run -D BACKEND=hip  main.mojo [args]

from max.gpu.host import DeviceAttribute, DeviceContext
from std.collections import List, Optional
from std.sys import exit, size_of, stderr
from std.time import perf_counter

from backend import (
    backend_label,
    check_backend,
    list_devices,
    open_device,
    query_device,
)
from cli import Args, help_text, parse_arguments
from energy import BoardEnergy
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
from stencil import (
    TBSize,
    check_correctness,
    laplacian,
    test_function,
    theoretical_fetch_bytes,
    theoretical_write_bytes,
    time_one_iteration,
)


def precision_name[dtype: DType]() -> String:
    comptime if dtype == DType.float64:
        return "float64"
    elif dtype == DType.float32:
        return "float32"
    else:
        return "float16"


def _attr(ctx: DeviceContext, attr: DeviceAttribute) -> Int:
    try:
        return ctx.get_attribute(attr)
    except:
        return -1


def _check_limit(value: Int, limit: Int, what: String) raises:
    if limit > 0 and value > limit:
        raise Error(
            what + " " + String(value) + " exceeds the device limit of " + String(limit)
        )


def emit(s: String, csv_mode: Bool):
    """Report text: stdout in text mode, stderr in the CSV modes."""
    if csv_mode:
        print(s, end="", file=stderr, flush=True)
    else:
        print(s, end="", flush=True)


def run[dtype: DType](args: Args) raises:
    var csv_mode = args.csv or args.csv_raw
    var nx = args.nx
    var ny = args.ny
    var nz = args.nz
    var bx = args.block_x
    var by = args.block_y
    var bz = args.block_z
    var gx = (nx - 1) // bx + 1
    var gy = (ny - 1) // by + 1
    var gz = (nz - 1) // bz + 1

    # ---- precision range (before touching the device) ----------------------
    comptime if dtype == DType.float16:
        # invhxyz2 = -2 * sum((n - 1)^2) must stay below the float16 maximum.
        var coeff = 2 * ((nx - 1) * (nx - 1) + (ny - 1) * (ny - 1) + (nz - 1) * (nz - 1))
        if coeff > 65504:
            raise Error(
                "--half: the stencil coefficient 2 * sum((side - 1)^2) = "
                + thousands(coeff)
                + " overflows float16 (max 65,504); use at most 105 points"
                " per side for a cube, or --float"
            )

    # ---- device ------------------------------------------------------------
    var ctx = open_device(args.device_index)
    var info = query_device(ctx, args.device_index)

    # ---- validation before allocating or launching -------------------------
    _check_limit(bx * by * bz, info.max_threads_per_block, "threads per block")
    _check_limit(TBSize, info.max_threads_per_block, "initialization block size")
    _check_limit(bx, _attr(ctx, DeviceAttribute.MAX_BLOCK_DIM_X), "block x size")
    _check_limit(by, _attr(ctx, DeviceAttribute.MAX_BLOCK_DIM_Y), "block y size")
    _check_limit(bz, _attr(ctx, DeviceAttribute.MAX_BLOCK_DIM_Z), "block z size")
    var max_gy = _attr(ctx, DeviceAttribute.MAX_GRID_DIM_Y)
    var max_gz = _attr(ctx, DeviceAttribute.MAX_GRID_DIM_Z)
    _check_limit(gx, info.max_grid_x, "grid x blocks")
    _check_limit(gy, max_gy, "grid y blocks")
    _check_limit(gz, max_gz, "grid z blocks")
    # The initialization launch uses a (·, ny, nz) grid of 1024-thread blocks.
    _check_limit(ny, max_gy, "initialization grid y (ny)")
    _check_limit(nz, max_gz, "initialization grid z (nz)")

    # ---- report header (printed before the run starts) ---------------------
    var rep = Report()
    rep.backend = backend_label()
    rep.device = info.name
    rep.driver = info.driver
    rep.precision = precision_name[dtype]()
    rep.bytes_per_elem = size_of[dtype]()
    rep.nx = nx
    rep.ny = ny
    rep.nz = nz
    rep.block_x = bx
    rep.block_y = by
    rep.block_z = bz
    rep.grid_x = gx
    rep.grid_y = gy
    rep.grid_z = gz
    rep.fetch_bytes = theoretical_fetch_bytes[dtype](nx, ny, nz)
    rep.write_bytes = theoretical_write_bytes[dtype](nx, ny, nz)
    rep.num_times = args.num_times
    rep.energy_enabled = args.energy
    rep.mibibytes = args.mibibytes
    for i in range(len(info.notes)):
        rep.header_notes.append(info.notes[i])
    emit(format_header(rep) + "\n", csv_mode)

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
    var n = nx * ny * nz
    var d_u = ctx.enqueue_create_buffer[dtype](n)
    var d_f = ctx.enqueue_create_buffer[dtype](n)

    var hx = Scalar[dtype](1) / Scalar[dtype](nx - 1)
    var hy = Scalar[dtype](1) / Scalar[dtype](ny - 1)
    var hz = Scalar[dtype](1) / Scalar[dtype](nz - 1)

    test_function[dtype](ctx, d_u, nx, ny, nz, hx, hy, hz)

    # Untimed run: warm-up, and the result that is checked.
    laplacian[dtype](ctx, d_f, d_u, nx, ny, nz, bx, by, bz, hx, hy, hz)
    ctx.synchronize()
    var corr = check_correctness[dtype](ctx, d_f, nx, ny, nz)

    var t_start = Float64(0)
    if prof:
        ctx.synchronize()
        prof.value().start()
        t_start = perf_counter()
    for _ in range(args.num_times):
        rep.times.append(
            time_one_iteration[dtype](
                ctx, d_f, d_u, nx, ny, nz, bx, by, bz, hx, hy, hz
            )
        )
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
                " (use a larger grid or -n)"
            )
    rep.finish_times()

    # ---- correctness -------------------------------------------------------
    rep.max_abs_err = corr.max_abs_err
    rep.mean_abs_err = corr.mean_abs_err
    rep.l2_rel_err = corr.l2_rel_err
    rep.num_checked = corr.num_checked
    comptime if dtype != DType.float64:
        rep.check_notes.append(
            "the stencil divides differences of nearby values by h^2 = 1 / (n - 1)^2,"
            " so "
            + rep.precision
            + " cancellation error grows with the grid size; large errors here"
            " reflect the precision, not a wrong result"
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
