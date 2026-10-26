#!/usr/bin/env python3
"""Hartree-Fock (Triton): wiring, validation against device limits, run loop."""

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


def validate(opts, be, props, grid, max_block, num_warps):
    bs = opts.block_size
    if bs & (bs - 1):
        fail("block size %s is not a power of two (Triton's tl.arange requires one)"
             % report.count(bs))
    if bs > max_block:
        fail("block size %s exceeds Triton's maximum tensor size of %s elements"
             % (report.count(bs), report.count(max_block)))
    if grid > INT32_MAX:
        fail("grid of %s programs exceeds the maximum grid size %s; the pinned block "
             "size is 256, so use a smaller input"
             % (report.count(grid), report.count(INT32_MAX)))
    if be == backend.HIP:
        warp = int(getattr(props, "warp_size", 64) or 64)
        work_items = grid * num_warps * warp
        if work_items > UINT32_MAX:
            fail("grid of %s programs x %d threads exceeds HIP's limit of %s work-items "
                 "per launch; the pinned block size is 256, so use a smaller input"
                  % (report.count(grid), num_warps * warp, report.count(UINT32_MAX)))
    # The kernel accumulates the Fock matrix with float atomic adds.
    if be == backend.CUDA and opts.precision == "float64":
        major = int(getattr(props, "major", 0) or 0)
        minor = int(getattr(props, "minor", 0) or 0)
        if major and major < 6:
            fail("float64 atomic add needs compute capability 6.0 or newer; this device "
                 "is %d.%d (use --float)" % (major, minor))


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
    except backend.BackendError as e:
        fail(str(e))
    try:
        import triton.language as tl
        import hf
    except ImportError as e:
        fail("Triton is not installed in this environment, or lacks pow/erf (%s)" % e)
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

    try:
        prob = hf.Problem(opts.input, opts.precision, opts.legacy_sqrt)
    except hf.InputError as e:
        fail(str(e))
    grid = prob.grid(opts.block_size)
    num_warps = 8 if be == backend.CUDA else 4
    validate(opts, be, props, grid, tl.TRITON_MAX_TENSOR_NUMEL, num_warps)

    run_check = opts.run_check and (opts.force_check or prob.natoms <= opts.max_check_atoms)

    r = report.Report()
    r.backend = be
    r.device = info.name
    r.driver = info.driver
    r.precision = opts.precision
    r.sqrt_mode = "legacy-sqrtf" if opts.legacy_sqrt else "native"
    r.input = prob.input_name
    r.natoms = prob.natoms
    r.ngauss = prob.ngauss
    r.atom_pairs = prob.nn
    r.quartets = prob.nnnn
    r.block_size = opts.block_size
    r.grid = grid
    r.num_warps = num_warps
    r.num_times = opts.num_times
    r.energy_enabled = opts.energy
    r.header_notes = list(info.notes)
    if not opts.run_check:
        r.check = "skipped (--no-check)"
    elif not run_check:
        r.check = ("skipped (%s atoms > --max-check-atoms %s; --check forces it)"
                   % (report.count(prob.natoms), report.count(opts.max_check_atoms)))
    else:
        r.check = "float64 CPU reference on the untimed run's Fock matrix"

    csv_mode = opts.csv or opts.csv_raw
    info_out = sys.stderr if csv_mode else sys.stdout
    info_out.write(report.text_header(r))
    info_out.flush()

    try:
        run = hf.FockBuild(prob, opts.block_size, opts.device, num_warps)
    except torch.cuda.OutOfMemoryError:
        fail("not enough device memory for the %s-atom input" % report.count(prob.natoms))

    meter = None
    if opts.energy:
        try:
            meter = BoardEnergy(opts.device, run.stream_ptr)
        except EnergyError as e:
            fail("energy profiler failed to start: %s" % e, "--no-energy")

    try:
        h_fock = run.untimed_run()
        if meter is not None:
            meter.start()
        for it in range(1, opts.num_times + 1):
            r.times.append(run.timed_run(it))
    except hf.TimerError as e:
        fail(str(e))
    if meter is not None:
        r.energy = meter.stop(opts.num_times)

    c = hf.check(prob, h_fock, run_check)
    if not opts.run_check:
        c.notes.append("CPU reference check skipped (--no-check)")
    elif not run_check:
        c.notes.append("CPU reference check skipped: %s atoms > --max-check-atoms %s "
                       "(use --check to force)"
                       % (report.count(prob.natoms), report.count(opts.max_check_atoms)))
    if opts.legacy_sqrt and run_check:
        c.notes.append("--legacy-sqrt evaluates two square roots in float32 while the "
                       "reference uses float64, so relative errors up to about 1e-7 are expected")
    r.correctness = c

    if csv_mode:
        sys.stdout.write(report.csv_raw(r) if opts.csv_raw else report.csv_summary(r))
        sys.stdout.flush()
        sys.stderr.write(report.text_stderr_tail(r))
    else:
        sys.stdout.write(report.text_body(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
