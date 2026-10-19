// C API implementation for the GPU EnergyProfiler.  See kernel_energy_capi.h.
//
// kernel.energy.h is header-style (#pragma once, all-inline class) and picks
// the NVML or rocm_smi backend from __HIPCC__, so we just include it and forward
// each C entry point to the matching EnergyProfiler method.
#include "kernel.energy.h"
#include "kernel_energy_capi.h"

static inline EnergyProfiler* as_prof(void* h) {
    return reinterpret_cast<EnergyProfiler*>(h);
}

extern "C" {

void* energy_create(int gpu_id, void* stream) {
#if defined(__HIPCC__)
    return new (std::nothrow) EnergyProfiler((hipStream_t)stream, gpu_id);
#else
    return new (std::nothrow) EnergyProfiler((CUstream)stream, gpu_id);
#endif
}

void energy_destroy(void* h)        { delete as_prof(h); }
void energy_start(void* h)          { if (h) as_prof(h)->start(); }
void energy_stop(void* h)           { if (h) as_prof(h)->stop(); }

double energy_joules(void* h)       { return h ? as_prof(h)->get_energy_joules() : 0.0; }
double energy_elapsed_ms(void* h)   { return h ? as_prof(h)->get_elapsed_ms()    : 0.0; }
double energy_avg_power_w(void* h)  { return h ? as_prof(h)->get_avg_power_w()    : 0.0; }
int    energy_sample_count(void* h) { return h ? as_prof(h)->get_sample_count()  : 0;   }

void energy_display(void* h, double ops, int iterations) {
    if (h) as_prof(h)->display(ops, iterations);
}

} // extern "C"
