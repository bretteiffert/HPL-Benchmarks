#!/usr/bin/env python3
"""Seven-point stencil (Triton): wiring, validation against device limits, run loop."""

import os
import sys

import backend
import cli
import report

INT32_MAX = 2 ** 31 - 1
UINT32_MAX = 2 ** 32 - 1
GRID_YZ_MAX = 65535        # CUDA's limit on grid y and z (Triton passes the grid through)
FP16_MAX = 65504.0
TRITON_NUM_WARPS = 4       # Triton's default launch (num_warps=4) on NVIDIA and AMD


def fail(what, hint="--help"):
    print("error: %s (try '%s')" % (what, hint), file=sys.stderr)
    sys.exit(1)


def _cdiv(a, b):
    return (a + b - 1) // b


def validate(opts, be, props, max_numel, test_tb):
    nx, ny, nz = opts.grid
    bx, by, bz = opts.block
    c = report.count
    for name, b in (("x", bx), ("y", by), ("z", bz)):
        if b & (b - 1):
            fail("block size %s along %s is not a power of two (Triton's tl.arange requires one)"
                 % (c(b), name))
    if bx * by * bz > max_numel:
        fail("block %s x %s x %s = %s points exceeds Triton's maximum tensor size of %s"
             % (c(bx), c(by), c(bz), c(bx * by * bz), c(max_numel)))
    if opts.precision == "float16":
        coeff = 2.0 * ((nx - 1) ** 2 + (ny - 1) ** 2 + (nz - 1) ** 2)
        if coeff > FP16_MAX:
            fail("float16 cannot hold the centre coefficient 2*((X-1)^2+(Y-1)^2+(Z-1)^2) = %s "
                 "(max 65,504); use a grid of at most 105 per side or --float"
                 % c(int(coeff)))
    # Both kernels form point offsets in 32-bit integers before widening them, and
    # padded programs compute offsets past the grid.
    padded = max(_cdiv(nx, bx) * bx * _cdiv(ny, by) * by * _cdiv(nz, bz) * bz,
                 _cdiv(nx, test_tb) * test_tb * ny * nz)
    if padded > INT32_MAX:
        fail("grid %s x %s x %s is too large: the kernels compute 32-bit point offsets, so "
             "the grid rounded up to whole blocks (%s points) must stay below %s"
             % (c(nx), c(ny), c(nz), c(padded), c(INT32_MAX + 1)))
    gy, gz = max(_cdiv(ny, by), ny), max(_cdiv(nz, bz), nz)
    if gy > GRID_YZ_MAX or gz > GRID_YZ_MAX:
        fail("grid needs %s x %s programs along y and z, above the limit of %s"
             % (c(gy), c(gz), c(GRID_YZ_MAX)))
    if be == backend.HIP:
        warp = int(getattr(props, "warp_size", 64) or 64)
        gx = max(_cdiv(nx, bx), _cdiv(nx, test_tb))
        if gx * TRITON_NUM_WARPS * warp > UINT32_MAX:
            fail("%s programs along x x %d threads exceeds HIP's limit of %s work-items; "
                 "use a larger --block-size along x"
                 % (c(gx), TRITON_NUM_WARPS * warp, c(UINT32_MAX)))


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
        import stencil
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
    validate(opts, be, props, tl.TRITON_MAX_TENSOR_NUMEL, stencil.TEST_FUNCTION_TB)

    nx, ny, nz = opts.grid
    bx, by, bz = opts.block
    r = report.Report()
    r.backend = be
    r.device = info.name
    r.driver = info.driver
    r.precision = opts.precision
    r.itemsize = {"float16": 2, "float32": 4, "float64": 8}[opts.precision]
    r.nx, r.ny, r.nz = nx, ny, nz
    r.bx, r.by, r.bz = bx, by, bz
    r.grid = stencil.launch_grid(nx, ny, nz, bx, by, bz)
    r.fetch_bytes, r.write_bytes = stencil.traffic_bytes(nx, ny, nz, r.itemsize)
    r.num_times = opts.num_times
    r.energy_enabled = opts.energy
    r.header_notes = list(info.notes)

    csv_mode = opts.csv or opts.csv_raw
    info_out = sys.stderr if csv_mode else sys.stdout
    info_out.write(report.text_header(r))
    info_out.flush()

    try:
        st = stencil.Stencil(nx, ny, nz, opts.block, opts.precision, opts.device)
    except torch.cuda.OutOfMemoryError:
        fail("not enough device memory for two arrays of %.1f MB"
             % (nx * ny * nz * r.itemsize * 1.0e-6))

    meter = None
    if opts.energy:
        try:
            meter = BoardEnergy(opts.device, st.stream_ptr)
        except EnergyError as e:
            fail("energy profiler failed to start: %s" % e, "--no-energy")

    try:
        st.untimed_run()
        r.correctness = stencil.check(st)   # every run rewrites the same values
        if opts.precision == "float16":
            r.correctness.notes.append(
                "float16 keeps about 3 significant digits and the stencil subtracts nearly "
                "equal neighbours, so errors around 1e-1 are expected")
        if meter is not None:
            meter.start()
        for it in range(1, opts.num_times + 1):
            r.times.append(st.timed_run(it))
    except stencil.TimerError as e:
        fail(str(e))
    if meter is not None:
        r.energy = meter.stop(opts.num_times)

    if csv_mode:
        sys.stdout.write(report.csv_raw(r) if opts.csv_raw else report.csv_summary(r))
        sys.stdout.flush()
        sys.stderr.write(report.text_stderr_tail(r))
    else:
        sys.stdout.write(report.text_body(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
