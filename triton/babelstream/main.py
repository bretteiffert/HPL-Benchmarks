#!/usr/bin/env python3
"""BabelStream (Triton): wiring, validation against device limits, run loop."""

import os
import sys

import backend
import cli
import report

INT32_MAX = 2 ** 31 - 1
UINT32_MAX = 2 ** 32 - 1
def fail(what, hint="--help"):
    print("error: %s (try '%s')" % (what, hint), file=sys.stderr)
    sys.exit(1)


def validate(opts, be, props, dot_groups, max_block, num_warps):
    bs = opts.block_size
    if bs & (bs - 1):
        fail("block size %s is not a power of two (Triton's tl.arange requires one)"
             % report.count(bs))
    if bs > max_block:
        fail("block size %s exceeds Triton's maximum tensor size of %s elements"
             % (report.count(bs), report.count(max_block)))
    if opts.array_size % bs:
        fail("array size %s is not a multiple of the block size %s "
             "(the elementwise kernels have no bounds check)"
             % (report.count(opts.array_size), report.count(bs)))
    # Offsets are int32 in every kernel; the dot loop steps one stride past the end.
    largest_offset = opts.array_size - 1 + bs * dot_groups
    if largest_offset > INT32_MAX:
        fail("array size %s is too large: kernel offsets are 32-bit, so array size "
             "plus the dot stride (%s) must stay below %s"
             % (report.count(opts.array_size), report.count(bs * dot_groups),
                report.count(INT32_MAX + 1)))
    if be == backend.HIP:
        warp = int(getattr(props, "warp_size", 64) or 64)
        work_items = (opts.array_size // bs) * num_warps * warp
        if work_items > UINT32_MAX:
            fail("grid of %s programs x %d threads exceeds HIP's limit of %s work-items "
                 "per launch; use a larger --block-size"
                  % (report.count(opts.array_size // bs), num_warps * warp,
                    report.count(UINT32_MAX)))


def main(argv):
    try:
        opts = cli.parse(argv)
    except cli.UsageError as e:
        fail(str(e))
    if opts.help:
        sys.stdout.write(cli.help_text(os.path.basename(argv[0])))
        return 0

    try:
        torch = backend.load_torch()
        be = backend.select(torch)
        expected_backend = os.environ.get("HPL_BACKEND")
        if expected_backend and be != expected_backend:
            raise backend.BackendError(
                "requested %s but the selected pixi environment loaded a %s PyTorch build"
                % (expected_backend, be)
            )
    except backend.BackendError as e:
        fail(str(e))
    try:
        import triton.language as tl
        import stream as stream_mod
    except ImportError as e:
        fail("Triton is not installed in this environment (%s)" % e)
    from energy import BoardEnergy, EnergyError

    n_dev = backend.device_count(torch)
    if opts.list:
        if n_dev == 0:
            fail("no %s devices found" % be)
        for i, name in backend.list_devices(torch, be):
            print("%d: %s" % (i, name))
        return 0
    if n_dev == 0:
        fail("no %s devices found" % be)
    if opts.device >= n_dev:
        fail("device index %d out of range (%d device%s found; see --list)"
             % (opts.device, n_dev, "" if n_dev == 1 else "s"))

    info = backend.describe(torch, be, opts.device)
    props = torch.cuda.get_device_properties(opts.device)

    r = report.Report()
    r.backend = be
    r.device = info.name
    r.driver = info.driver
    r.precision = opts.precision
    r.itemsize = {"float16": 2, "float32": 4, "float64": 8}[opts.precision]
    r.array_size = opts.array_size
    r.block_size = opts.block_size
    r.num_times = opts.num_times
    r.energy_enabled = opts.energy
    r.mibibytes = opts.mibibytes
    r.header_notes = list(info.notes)
    if not info.compute_units:
        fail("compute-unit count is unavailable; the pinned Dot configuration requires 4 x SM/CU groups")
    r.dot_groups = 4 * info.compute_units
    r.dot_groups_detail = "(4 × %s %s)" % (report.count(info.compute_units), info.unit_label)
    fmad = os.environ.get("FMAD")
    r.fmad = fmad if fmad is not None else "unset"
    if fmad is not None:
        r.header_notes.append("FMAD has no effect; Triton fuses multiply-add by default "
                              "(TRITON_DEFAULT_FP_FUSION=0 turns that off)")

    warp = 32 if be == backend.CUDA else 64
    num_warps = opts.block_size // warp
    if num_warps not in (4, 8):
        fail("the pinned 256-element Triton program requires 8 CUDA warps or 4 HIP waves")
    validate(opts, be, props, r.dot_groups, tl.TRITON_MAX_TENSOR_NUMEL, num_warps)

    csv_mode = opts.csv or opts.csv_raw
    info_out = sys.stderr if csv_mode else sys.stdout
    info_out.write(report.text_header(r))
    info_out.flush()

    dtype = {"float16": torch.float16, "float32": torch.float32,
             "float64": torch.float64}[opts.precision]
    try:
        s = stream_mod.TritonStream(opts.array_size, opts.block_size, r.dot_groups,
                                    opts.device, dtype, num_warps)
    except torch.cuda.OutOfMemoryError:
        fail("not enough device memory for three arrays of %.1f MB"
             % (opts.array_size * r.itemsize * 1.0e-6))

    meter = None
    if opts.energy:
        try:
            meter = BoardEnergy(opts.device, s.stream_ptr)
        except EnergyError as e:
            fail("energy profiler failed to start: %s" % e, "--no-energy")

    routines = [s.copy, s.mul, s.add, s.triad]
    dot_sum = None
    try:
        r.init_time = s.init_arrays()
        for it in range(opts.num_times):
            # Iteration 0 is warm-up: excluded from statistics, CSV and energy.
            if it == 1 and meter is not None:
                meter.start()
            times = [f() for f in routines]
            dot_time, dot_sum = s.dot()
            if it > 0:
                for name, t in zip(report.ROUTINES, times + [dot_time]):
                    r.times[name].append(t)
    except stream_mod.TimerError as e:
        fail(str(e))
    if meter is not None:
        r.energy = meter.stop(r.timed_iterations)

    a, b, c = s.read_arrays()
    r.correctness = stream_mod.check_solution(opts.num_times, a, b, c, dot_sum,
                                              opts.array_size)

    if csv_mode:
        sys.stdout.write(report.csv_raw(r) if opts.csv_raw else report.csv_summary(r))
        sys.stdout.flush()
        sys.stderr.write(report.text_stderr_tail(r))
    else:
        sys.stdout.write(report.text_body(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
