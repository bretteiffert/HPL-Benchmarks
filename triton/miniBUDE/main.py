#!/usr/bin/env python3
"""miniBUDE (Triton): wiring, validation against device limits, run loop."""

import os
import sys

import backend
import cli
import report

INT32_MAX = 2 ** 31 - 1
MAX_THREADS_PER_WORKGROUP = 1024   # CUDA and HIP


def fail(what, hint="--help"):
    print("error: %s (try '%s')" % (what, hint), file=sys.stderr)
    sys.exit(1)


def select_device(torch, be, selector):
    devices = backend.list_devices(torch, be)
    if not devices:
        fail("no %s devices found" % be)
    if selector == "":
        return 0
    try:
        idx = int(selector, 10)
    except ValueError:
        for i, name in devices:
            if selector in name:
                return i
        fail("no device name contains '%s' (see --list)" % selector)
    if not 0 <= idx < len(devices):
        fail("device index %d out of range (%d device%s found; see --list)"
             % (idx, len(devices), "" if len(devices) == 1 else "s"))
    return idx


def plan(opts, nposes, warp_size, max_block):
    """[(ppwi, wgsize)] to run, and a note for each combination skipped."""
    configs, skipped = [], []
    for ppwi in opts.ppwis:
        for wg in opts.wgsizes:
            block = ppwi * wg
            why = None
            if wg % warp_size:
                why = "wgsize %d is not a multiple of the warp size %d" % (wg, warp_size)
            elif (wg // warp_size) & (wg // warp_size - 1):
                why = ("num_warps %d (wgsize / warp size) is not a power of two, which "
                       "Triton requires" % (wg // warp_size))
            elif wg > MAX_THREADS_PER_WORKGROUP:
                why = "wgsize %d exceeds %s threads per work-group" % (
                    wg, report.count(MAX_THREADS_PER_WORKGROUP))
            elif block > max_block:
                why = "ppwi x wgsize %s exceeds Triton's maximum tensor size %s" % (
                    report.count(block), report.count(max_block))
            elif nposes < block:
                why = "fewer poses (%s) than ppwi x wgsize (%s)" % (
                    report.count(nposes), report.count(block))
            elif nposes % block:
                why = "%s poses do not divide by ppwi x wgsize (%s)" % (
                    report.count(nposes), report.count(block))
            elif nposes - 1 + block > INT32_MAX:
                why = "pose offsets would overflow 32 bits"
            if why:
                skipped.append("skipped ppwi %d, wgsize %d: %s" % (ppwi, wg, why))
            else:
                configs.append((ppwi, wg))
    return configs, skipped


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
        import bude
        import deck as deck_mod
    except ImportError as e:
        fail("Triton is not installed in this environment (%s)" % e)
    from energy import BoardEnergy, EnergyError

    if opts.list:
        devices = backend.list_devices(torch, be)
        if not devices:
            fail("no %s devices found" % be)
        for i, name in devices:
            print("%d: %s" % (i, name))
        return 0

    idx = select_device(torch, be, opts.device)
    torch.cuda.set_device(idx)
    device = torch.device("cuda", idx)
    info = backend.describe(torch, be, idx)
    warp_size, warp_how = bude.resolve_warp_size()

    r = report.Report()
    r.header_notes = list(info.notes)
    if warp_how == "guessed":
        r.header_notes.append("warp size could not be queried; assumed %d for %s"
                              % (warp_size, be))
    if not opts.wgsizes:
        opts.wgsizes = [warp_size]
        r.header_notes.append("wgsize defaults to one warp; the reference default of 1 "
                              "is not expressible in Triton")

    deck_dir = opts.deck or os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), cli.DEFAULT_DECK))
    try:
        dk = deck_mod.load(deck_dir, opts.nposes)
    except deck_mod.DeckError as e:
        fail(str(e))

    configs, r.skip_notes = plan(opts, dk.nposes, warp_size, tl.TRITON_MAX_TENSOR_NUMEL)
    if not configs:
        fail("no runnable (ppwi, wgsize) combination: %s" % "; ".join(
            n.replace("skipped ", "") for n in r.skip_notes))

    r.backend = be
    r.device = info.name
    r.driver = info.driver
    r.precision = opts.precision
    r.deck = dk.name
    r.natlig, r.natpro, r.ntypes = dk.natlig, dk.natpro, dk.ntypes
    r.nposes, r.max_poses = dk.nposes, dk.max_poses
    r.ppwis, r.wgsizes = opts.ppwis, opts.wgsizes
    r.warp_size, r.warp_how = warp_size, warp_how
    r.num_stages = ("tuned per configuration over %s (untimed)"
                    % ", ".join(map(str, bude.NUM_STAGES_CANDIDATES))
                    if bude.AUTOTUNE_NUM_STAGES else "pinned at %d" % bude.NUM_STAGES_DEFAULT)
    r.iterations, r.warmup, r.rows = opts.iterations, opts.warmup, opts.rows
    r.energy_enabled = opts.energy
    r.ulp_labels = [bude.ulp_bucket_label(i) for i in range(bude.ULP_NBUCKETS)]

    csv_mode = opts.csv or opts.csv_raw
    info_out = sys.stderr if csv_mode else sys.stdout
    info_out.write(report.text_header(r))
    info_out.flush()

    import numpy as np
    dtype = np.float32 if opts.precision == "float32" else np.float64
    stream_ptr = torch.cuda.current_stream(device).cuda_stream
    meter_factory = (lambda: BoardEnergy(idx, stream_ptr)) if opts.energy else None
    if meter_factory is not None:
        try:
            meter_factory().close()   # fail before any launch if the profiler is unavailable
        except EnergyError as e:
            fail("energy profiler failed to start: %s" % e, "--no-energy")
    timer = bude.Timer()

    try:
        reference = None
        if dtype == np.float32:
            for ppwi, wg in configs:
                try:
                    reference = bude.reference_pass(dk, ppwi, wg, warp_size, device)
                    break
                except bude.ConfigUnavailable:
                    continue
            if reference is None:
                fail("the float64 reference pass did not fit the device in any configuration")

        for ppwi, wg in configs:
            try:
                buf = bude.Buffers(dk, dtype, device)
                s = bude.run_config(buf, ppwi, wg, warp_size, opts.warmup, opts.iterations,
                                    timer, meter_factory)
            except (bude.ConfigUnavailable, bude.OutOfResources) as e:
                r.skip_notes.append("skipped ppwi %d, wgsize %d: exceeded device resources (%s)"
                                    % (ppwi, wg, str(e).replace("\n", " ")[:160]))
                continue
            finally:
                buf = None
            bude.evaluate(dk, s, reference)
            r.samples.append(s)
    except bude.TimerError as e:
        fail(str(e))
    except EnergyError as e:
        fail("energy profiler failed to start: %s" % e, "--no-energy")
    except torch.cuda.OutOfMemoryError:
        fail("not enough device memory for the %s-pose deck" % report.count(dk.nposes))

    if not r.samples:
        fail("no configuration could run on this device: %s" % "; ".join(r.skip_notes))

    if opts.output:
        try:
            with open(opts.output, "w") as f:
                for e in r.samples[0].energies:
                    f.write("%.17g\n" % e)
        except OSError as e:
            fail("cannot write energies to '%s': %s" % (opts.output, e.strerror or e))

    if csv_mode:
        sys.stdout.write(report.csv_raw(r) if opts.csv_raw else report.csv_summary(r))
        sys.stdout.flush()
        sys.stderr.write(report.text_stderr_tail(r))
    else:
        sys.stdout.write(report.text_body(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
