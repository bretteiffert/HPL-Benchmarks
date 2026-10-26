"""Python wrapper around the GPU energy-profiler C API (libkernelenergy.so).

Thin ctypes binding to the ``energy_*`` functions declared in
``kernel_energy_capi.h``.  The C profiler samples instantaneous board power on a
background thread between :meth:`EnergyProfiler.start` / :meth:`stop` and
integrates it against wall-clock time to report joules; elapsed time is measured
with GPU events on the given stream.

Build the shared library first::

    dslbench/build_energy.sh            # --cuda (NVML) auto on NVIDIA, --hip on ROCm

Manual use mirroring the C sequence::

    from dslbench.energy import EnergyProfiler
    with EnergyProfiler(gpu_id=0) as prof:
        prof.start()
        for _ in range(iters):
            launch_kernel(...)
        prof.stop()
        print(prof.joules, prof.elapsed_ms, prof.avg_power_w)

Or let :meth:`profile` run the warmup + timed loop and return a metrics dict::

    prof = EnergyProfiler(gpu_id=0)
    m = prof.profile(lambda: launch_kernel(...), iters=100, ops=2*M*N*K, sync=cp.cuda.runtime.deviceSynchronize)
    print(m["joules_per_iter"], m["avg_power_w"], m["tops_per_watt"])
"""

import ctypes
import os

__all__ = ["EnergyProfiler", "load_lib"]

_DEFAULT_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkernelenergy.so")

# One cached handle per resolved library path.
_LIB_CACHE = {}


def load_lib(lib_path=None):
    """Load libkernelenergy.so once and declare the C prototypes.

    Set ``DSLBENCH_ENERGY_LIB`` to override the path.  Raises a helpful
    ``OSError`` (pointing at build_energy.sh) if the library is missing.
    """
    path = lib_path or os.environ.get("DSLBENCH_ENERGY_LIB") or _DEFAULT_LIB
    path = os.path.abspath(path)
    if path in _LIB_CACHE:
        return _LIB_CACHE[path]

    if not os.path.exists(path):
        raise OSError(
            f"energy library not found: {path}\n"
            "Build it first:  dslbench/build_energy.sh   (--cuda or --hip)"
        )
    lib = ctypes.CDLL(path)

    lib.energy_create.argtypes = [ctypes.c_int, ctypes.c_void_p]
    lib.energy_create.restype = ctypes.c_void_p
    lib.energy_destroy.argtypes = [ctypes.c_void_p]
    lib.energy_destroy.restype = None
    lib.energy_start.argtypes = [ctypes.c_void_p]
    lib.energy_start.restype = None
    lib.energy_stop.argtypes = [ctypes.c_void_p]
    lib.energy_stop.restype = None
    lib.energy_joules.argtypes = [ctypes.c_void_p]
    lib.energy_joules.restype = ctypes.c_double
    lib.energy_elapsed_ms.argtypes = [ctypes.c_void_p]
    lib.energy_elapsed_ms.restype = ctypes.c_double
    lib.energy_avg_power_w.argtypes = [ctypes.c_void_p]
    lib.energy_avg_power_w.restype = ctypes.c_double
    lib.energy_sample_count.argtypes = [ctypes.c_void_p]
    lib.energy_sample_count.restype = ctypes.c_int
    lib.energy_display.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
    lib.energy_display.restype = None

    _LIB_CACHE[path] = lib
    return lib


class EnergyProfiler:
    """Handle to a C ``EnergyProfiler`` for one GPU.

    Parameters
    ----------
    gpu_id : int
        GPU / SMI device index to sample (see the ``SQUID_*_DEV`` env-var
        overrides in ``kernel.energy.h`` for visibility-mask corner cases).
    stream : int
        GPU stream to time on, as an integer pointer value; ``0`` is the default
        stream.  For CuPy pass ``cp.cuda.get_current_stream().ptr``.
    lib_path : str, optional
        Path to libkernelenergy.so (default: alongside this file, or
        ``$DSLBENCH_ENERGY_LIB``).
    """

    def __init__(self, gpu_id=0, stream=0, lib_path=None):
        self._lib = load_lib(lib_path)
        self._h = self._lib.energy_create(int(gpu_id), ctypes.c_void_p(int(stream)))
        if not self._h:
            raise RuntimeError("energy_create returned NULL (allocation failed)")

    # -- lifecycle ------------------------------------------------------------
    def start(self):
        """Sync the device and begin timing + power sampling."""
        self._check()
        self._lib.energy_start(self._h)

    def stop(self):
        """Record the end event, drain the stream, and integrate energy."""
        self._check()
        self._lib.energy_stop(self._h)

    def close(self):
        """Release the underlying C handle (idempotent)."""
        if getattr(self, "_h", None):
            self._lib.energy_destroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        self.close()

    # -- results (valid after stop(), over the whole timed window) ------------
    @property
    def joules(self):
        self._check()
        return float(self._lib.energy_joules(self._h))

    @property
    def elapsed_ms(self):
        self._check()
        return float(self._lib.energy_elapsed_ms(self._h))

    @property
    def avg_power_w(self):
        self._check()
        return float(self._lib.energy_avg_power_w(self._h))

    @property
    def sample_count(self):
        self._check()
        return int(self._lib.energy_sample_count(self._h))

    def display(self, ops=0, iterations=1):
        """Print the C profiler's human-readable report."""
        self._check()
        self._lib.energy_display(self._h, float(ops), int(iterations))

    def metrics(self, ops=0, iterations=1):
        """Return per-iteration metrics as a dict (no printing).

        ``ops`` is the total op count over the timed window; per-iteration and
        TOPS/TOPS-per-watt figures divide by ``iterations`` to match
        :meth:`display`.
        """
        self._check()
        iters = max(int(iterations), 1)
        j = self.joules
        ms = self.elapsed_ms
        power = self.avg_power_w
        per_it_s = (ms / iters) * 1e-3
        tops = (ops / (per_it_s * 1e12)) if per_it_s > 0 else 0.0
        return {
            "joules": j,
            "joules_per_iter": j / iters,
            "elapsed_ms": ms,
            "ms_per_iter": ms / iters,
            "avg_power_w": power,
            "tops": tops,
            "tops_per_watt": (tops / power) if power > 0 else 0.0,
            "sample_count": self.sample_count,
            "iterations": iters,
            "ops": ops,
        }

    # -- convenience ----------------------------------------------------------
    def profile(self, fn, iters, ops=0, warmup=3, sync=None):
        """Run ``fn`` ``warmup``+``iters`` times under the profiler and return
        :meth:`metrics`.

        Mirrors the ``ENERGY_PROFILE`` C macro: ``warmup`` untimed calls, then a
        timed loop of ``iters`` calls between start/stop.  ``sync`` (e.g.
        ``cupy.cuda.runtime.deviceSynchronize``) is called once before starting so
        the warmup is fully drained; start()/stop() also sync the stream.
        """
        for _ in range(max(warmup, 0)):
            fn()
        if sync is not None:
            sync()
        self.start()
        for _ in range(iters):
            fn()
        self.stop()
        return self.metrics(ops=ops, iterations=iters)

    # -- internal -------------------------------------------------------------
    def _check(self):
        if not getattr(self, "_h", None):
            raise RuntimeError("EnergyProfiler handle is closed")


if __name__ == "__main__":
    # Smoke test: profile an empty window (no kernels) — should return ~0 J and a
    # nonzero idle power reading from the SMI sampler.
    import time

    with EnergyProfiler(gpu_id=int(os.environ.get("DSLBENCH_GPU", "0"))) as p:
        p.start()
        time.sleep(0.2)
        p.stop()
        print(f"idle window: {p.elapsed_ms:.2f} ms (GPU-event), "
              f"{p.sample_count} power samples, "
              f"{p.avg_power_w:.1f} W avg, {p.joules:.4f} J")
