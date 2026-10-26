"""Board-power measurement through the shared wrapper ../../dependencies/energy.py.

That wrapper loads libkernelenergy.so from the same dependencies folder (or from
$DSLBENCH_ENERGY_LIB). Shared by every Triton benchmark (keep this file identical
across them).
"""

import importlib.util
import os

DEPENDENCIES_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "dependencies"))


class EnergyError(Exception):
    pass


class EnergyResult:
    def __init__(self, joules, seconds, avg_power_w, samples, iterations):
        self.joules = joules
        self.seconds = seconds
        self.avg_power_w = avg_power_w
        self.samples = samples
        self.iterations = iterations

    @property
    def joules_per_iter(self):
        return self.joules / self.iterations

    @property
    def watt_hours(self):
        return self.joules / 3600.0


def _load_wrapper():
    path = os.path.join(DEPENDENCIES_DIR, "energy.py")
    if not os.path.isfile(path):
        raise EnergyError("energy wrapper not found: %s" % path)
    # Imported by path under its own name so this module cannot shadow it.
    spec = importlib.util.spec_from_file_location("dependencies_energy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BoardEnergy:
    """Samples board power on device ``device_index`` between start() and stop()."""

    def __init__(self, device_index, stream_ptr):
        try:
            wrapper = _load_wrapper()
            self._prof = wrapper.EnergyProfiler(gpu_id=device_index, stream=stream_ptr)
        except EnergyError:
            raise
        except (OSError, RuntimeError, AttributeError) as e:
            raise EnergyError(str(e).splitlines()[0])

    def start(self):
        self._prof.start()

    def stop(self, iterations):
        self._prof.stop()
        result = EnergyResult(self._prof.joules, self._prof.elapsed_ms * 1.0e-3,
                              self._prof.avg_power_w, self._prof.sample_count,
                              iterations)
        self.close()
        return result

    def close(self):
        prof = getattr(self, "_prof", None)
        if prof is not None:
            prof.close()
            self._prof = None
