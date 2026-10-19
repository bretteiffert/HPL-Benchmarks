// ─────────────────────────────────────────────────────────────────────────────
// C API for the GPU EnergyProfiler (kernel.energy.h).
//
// A flat, C-callable surface over the C++ EnergyProfiler class so it can be
// driven from Python (ctypes), Fortran, or any FFI.  The profiler samples
// instantaneous board power on a background thread between energy_start() and
// energy_stop() and integrates it against wall-clock time to yield joules;
// elapsed time comes from GPU events on the given stream.
//
// Backend is chosen at compile time: NVIDIA/NVML by default, AMD/rocm_smi when
// built with hipcc (__HIPCC__).  See build_energy.sh.
//
// Typical use (all handle methods are no-ops on a NULL handle):
//     void* p = energy_create(gpu_id, 0 /*default stream*/);
//     energy_start(p);
//     for (int i = 0; i < iters; i++) launch_kernel(...);
//     energy_stop(p);                       // blocks until the stream drains
//     double j  = energy_joules(p);         // total over the timed window
//     double ms = energy_elapsed_ms(p);
//     energy_destroy(p);
// ─────────────────────────────────────────────────────────────────────────────
#ifndef KERNEL_ENERGY_CAPI_H
#define KERNEL_ENERGY_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

// Create a profiler bound to GPU `gpu_id`, sampling power on `stream`
// (pass 0 for the default stream; on CUDA this is a CUstream, on HIP a
// hipStream_t — both are pointers).  Returns an opaque handle, or NULL on
// allocation failure.  NVML/rsmi is initialised here so energy_start() has no
// init latency.  The sampled SMI device can be overridden via the
// SQUID_NVML_DEV / SQUID_SMI_DEV env vars (see kernel.energy.h).
void*  energy_create(int gpu_id, void* stream);

// Destroy a handle created by energy_create() (safe on NULL).
void   energy_destroy(void* handle);

// Begin timing: syncs the device, records the start event, and launches the
// power-sampling thread.  Call after any warmup so init cost is excluded.
void   energy_start(void* handle);

// End timing: records the end event, blocks until the stream drains, stops the
// sampler, and integrates energy.  Results below are valid after this returns.
void   energy_stop(void* handle);

// Results for the whole timed window (all iterations run between start/stop).
double energy_joules(void* handle);        // integrated energy, joules
double energy_elapsed_ms(void* handle);    // GPU elapsed time, milliseconds
double energy_avg_power_w(void* handle);   // joules / seconds, watts
int    energy_sample_count(void* handle);  // number of power samples taken

// Print the human-readable report (energy/time/power/TOPS/TOPS-per-watt),
// dividing totals by `iterations`.  `ops` is the total op count over the window.
void   energy_display(void* handle, double ops, int iterations);

#ifdef __cplusplus
}
#endif

#endif // KERNEL_ENERGY_CAPI_H
