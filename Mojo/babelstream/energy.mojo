# Board-power measurement through the Python energy library (energy.py, which
# wraps libkernelenergy.so with ctypes). energy.py is shipped unchanged.
#
# Library lookup, first hit wins:
#   $DSLBENCH_ENERGY_LIB                         (handled by energy.py)
#   <project>/../../dependencies/libkernelenergy.so
#   energy.py's default (libkernelenergy.so next to energy.py)
#
# The profiler is created on stream 0. MAX launches on its own stream, so the
# caller synchronizes the device context right before start() and right
# before stop(): the profiler's window (its GPU events and its power-sampling
# thread) then spans exactly the timed iterations whichever stream it records
# on.

from std.os import getenv
from std.os.path import dirname, exists, join
from std.python import Python, PythonObject

from backend import add_python_paths, project_dir


def _lib_path() -> String:
    if getenv("DSLBENCH_ENERGY_LIB").byte_length() > 0:
        return ""  # energy.py reads the variable itself
    var deps = join(
        dirname(dirname(project_dir())), "dependencies", "libkernelenergy.so"
    )
    if exists(deps):
        return deps
    return ""


struct EnergyReading(Copyable):
    var joules: Float64
    var elapsed_s: Float64
    var avg_power_w: Float64
    var samples: Int

    def __init__(
        out self,
        joules: Float64,
        elapsed_s: Float64,
        avg_power_w: Float64,
        samples: Int,
    ):
        self.joules = joules
        self.elapsed_s = elapsed_s
        self.avg_power_w = avg_power_w
        self.samples = samples


struct BoardEnergy(Movable):
    var prof: PythonObject

    def __init__(out self, device_index: Int) raises:
        """Load energy.py and create a profiler for `device_index`.

        Raises with the library's message on failure; main turns that into
        an error suggesting --no-energy."""
        add_python_paths()
        var mod = Python.import_module("energy")
        var lib = _lib_path()
        if lib.byte_length() > 0:
            self.prof = mod.EnergyProfiler(
                gpu_id=PythonObject(Int64(device_index)),
                stream=PythonObject(Int64(0)),
                lib_path=PythonObject(lib),
            )
        else:
            self.prof = mod.EnergyProfiler(
                gpu_id=PythonObject(Int64(device_index)),
                stream=PythonObject(Int64(0)),
            )

    def start(mut self) raises:
        _ = self.prof.start()

    def stop(mut self) raises -> EnergyReading:
        _ = self.prof.stop()
        var r = EnergyReading(
            Float64(py=self.prof.joules),
            Float64(py=self.prof.elapsed_ms) * 1.0e-3,
            Float64(py=self.prof.avg_power_w),
            Int(Int64(py=self.prof.sample_count)),
        )
        return r^

    def close(mut self) raises:
        _ = self.prof.close()
