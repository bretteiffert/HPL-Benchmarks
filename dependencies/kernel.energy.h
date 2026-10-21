#pragma once

#include <iostream>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdlib>

#if defined(__HIPCC__)
// ─────────────────────────────────────────────────────────────────────────────
// HIP / ROCm energy profiler (AMD Instinct, e.g. MI300A) using rocm_smi.
// Mirrors the NVML profiler below: a background pthread samples instantaneous
// power while the timed kernel loop runs on the given HIP stream.
// ─────────────────────────────────────────────────────────────────────────────
#include <hip/hip_runtime.h>
#include <rocm_smi/rocm_smi.h>
#else
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvml.h>
#endif

#define ENABLE_ENERGY

#ifdef ENABLE_ENERGY
#define ENERGY_PROFILE(ITER, OPS, devno, stream, FN)  printf("********* Energy Profiling function:%s iter:%d\n", __func__, ITER); fflush(stdout); { FN; } { FN; } { FN; } { EnergyProfiler profiler(stream, devno); profiler.start(); for(int i=0; i<ITER; i++) { FN; } profiler.stop(); profiler.display(OPS, ITER); }
#else
#define ENERGY_PROFILE(ITER, OPS, devno, stream, FN)  FN
#endif

#if defined(__HIPCC__)
// ─────────────────────────────────────────────────────────────────────────────
// ROCm SMI implementation
// ─────────────────────────────────────────────────────────────────────────────
class EnergyProfiler
{
private:
    static constexpr int MAX_SAMPLES = 100000;
    static constexpr int SAMPLE_INTERVAL_US = 5000;

    uint32_t     dv_ind;          // rocm_smi (absolute) device index to sample
    pthread_t    sampler_thread;
    volatile bool running;

    double       power_samples[MAX_SAMPLES];
    double       timestamps[MAX_SAMPLES];
    int          sample_count;
    hipEvent_t   start_event;
    hipEvent_t   end_event;
    hipStream_t  stream;
    double       energy;
    float        elapsed_ms;

public:
    // rsmi_init in the constructor so start() has zero init overhead.
    // The rocm_smi index is *absolute* (independent of HIP_VISIBLE_DEVICES); it
    // defaults to gpu_id but can be overridden with SQUID_SMI_DEV so the sampled
    // GPU matches the one HIP actually runs on under a visibility mask.
    EnergyProfiler(hipStream_t stream=0, int gpu_id=0) : stream(stream)
    {
        running = false;
        sample_count = 0;
        dv_ind = (uint32_t)gpu_id;
        if(const char* e = getenv("SQUID_SMI_DEV")) dv_ind = (uint32_t)atoi(e);
        hipEventCreate(&start_event);
        hipEventCreate(&end_event);
        rsmi_init(0);
    }

    ~EnergyProfiler()
    {
        hipEventDestroy(start_event);
        hipEventDestroy(end_event);
        if(running)
            stop_dummy();
        rsmi_shut_down();
    }

private:
    static double current_time()
    {
        struct timeval tv;
        gettimeofday(&tv,nullptr);
        return tv.tv_sec + tv.tv_usec*1e-6;
    }

    static void* sampler_trampoline(void* arg)
    {
        return ((EnergyProfiler*)arg)->power_sampler();
    }

    // Query instantaneous power (µW) via rsmi_dev_power_get, falling back to the
    // current-socket and average-power APIs for older / partial device support.
    double read_power_watts()
    {
        uint64_t power_uW = 0;
        RSMI_POWER_TYPE ptype = RSMI_INVALID_POWER;
        rsmi_status_t st = rsmi_dev_power_get(dv_ind, &power_uW, &ptype);
        if(st != RSMI_STATUS_SUCCESS || power_uW == 0)
            st = rsmi_dev_current_socket_power_get(dv_ind, &power_uW);
        if(st != RSMI_STATUS_SUCCESS || power_uW == 0)
            rsmi_dev_power_ave_get(dv_ind, 0, &power_uW);
        return power_uW / 1.0e6;   // µW → W
    }

    void* power_sampler()
    {
        while(running)
        {
            double power_W = read_power_watts();
            if(sample_count < MAX_SAMPLES)
            {
                power_samples[sample_count] = power_W;
                timestamps[sample_count]    = current_time();
                sample_count++;
            }
            usleep(SAMPLE_INTERVAL_US);
        }
        return nullptr;
    }

    void stop_dummy() { stop(); }

public:
    void start()
    {
        hipDeviceSynchronize();          // ensure warmup is complete before timing
        hipEventRecord(start_event, stream);

        sample_count = 0;
        running      = true;
        pthread_create(&sampler_thread, nullptr, sampler_trampoline, this);
    }

    void stop()
    {
        hipEventRecord(end_event, stream);
        hipStreamSynchronize(stream);    // blocks until all kernels + end_event finish

        running = false;
        pthread_join(sampler_thread, nullptr);

        hipEventElapsedTime(&elapsed_ms, start_event, end_event);

        energy = 0.0;
        for(int i = 1; i < sample_count; i++)
        {
            double dt = timestamps[i] - timestamps[i-1];
            energy += power_samples[i] * dt;
        }
    }

    void display(double ops=0, int iterations=1) {
        double per_it_energy    = energy / iterations;               // J per iteration
        double per_it_elapsed_s = (elapsed_ms / iterations) * 1e-3;  // s per iteration

        double tops = ((double)ops) / (per_it_elapsed_s * 1e12);
        double avg_power_W    = (per_it_elapsed_s > 0.0)
                                ? per_it_energy / per_it_elapsed_s : 0.0;
        double tops_per_watt  = (avg_power_W > 0.0) ? tops / avg_power_W : 0.0;

        printf("Energy consumption  : %.4f J\n", per_it_energy);
        printf("Time                : %.4f ms\n", per_it_elapsed_s * 1e3);
        printf("Total ops           : %.2f OPS\n", ops);
        printf("Avg power           : %.2f W\n", avg_power_W);
        printf("Throughput          : %.2f TOPS\n", tops);
        printf("Throughput/Watt     : %.4f TOPS/W\n", tops_per_watt);
    }

    // Accessors for the C API (values are valid after stop()); energy/elapsed
    // cover the whole timed window (all ITER iterations).
    double get_energy_joules() const { return energy; }
    double get_elapsed_ms()    const { return elapsed_ms; }
    int    get_sample_count()  const { return sample_count; }
    double get_avg_power_w()   const {
        double s = elapsed_ms * 1e-3;
        return s > 0.0 ? energy / s : 0.0;
    }
};

#else
// ─────────────────────────────────────────────────────────────────────────────
// NVML implementation (NVIDIA)
// ─────────────────────────────────────────────────────────────────────────────
class EnergyProfiler
{
private:

    static constexpr int MAX_SAMPLES = 100000;
    static constexpr int SAMPLE_INTERVAL_US = 5000;

    nvmlDevice_t device;
    pthread_t sampler_thread;
    volatile bool running;

    double power_samples[MAX_SAMPLES];
    double timestamps[MAX_SAMPLES];
    int sample_count;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    CUstream stream;
    int gpu_id;
    double energy;
    float elapsed_ms;
public:

    // nvmlInit here so start() has zero NVML-init overhead
    EnergyProfiler(CUstream stream=0, int gpu_id=0) : stream(stream), gpu_id(gpu_id)
    {
        running = false;
        sample_count = 0;
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
        nvmlInit();
        // Bind NVML to the GPU that CUDA is actually running on.  NVML device
        // enumeration ignores CUDA_VISIBLE_DEVICES, so nvmlDeviceGetHandleByIndex
        // (gpu_id) can sample the wrong physical GPU under a visibility mask (and
        // if that GPU's power API returns NOT_SUPPORTED the reading is garbage).
        // Preference: SQUID_NVML_DEV env (explicit index) > current CUDA device's
        // PCI bus id (mask-independent) > gpu_id index as a last resort.
        device = nullptr;
        bool bound = false;
        if(const char* e = getenv("SQUID_NVML_DEV"))
            bound = (nvmlDeviceGetHandleByIndex((unsigned)atoi(e), &device) == NVML_SUCCESS);
        if(!bound) {
            int cudaDev = 0; char pci[64];
            if(cudaGetDevice(&cudaDev) == cudaSuccess &&
               cudaDeviceGetPCIBusId(pci, sizeof(pci), cudaDev) == cudaSuccess &&
               nvmlDeviceGetHandleByPciBusId(pci, &device) == NVML_SUCCESS)
                bound = true;
        }
        if(!bound)
            nvmlDeviceGetHandleByIndex((unsigned)gpu_id, &device);
    }

    ~EnergyProfiler()
    {
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
        if(running)
            stop_dummy();
        nvmlShutdown();
    }

private:

    /* ---------------------------
       time helper
       --------------------------- */
    static double current_time()
    {
        struct timeval tv;
        gettimeofday(&tv,nullptr);
        return tv.tv_sec + tv.tv_usec*1e-6;
    }

    /* ---------------------------
       thread trampoline
       --------------------------- */
    static void* sampler_trampoline(void* arg)
    {
        return ((EnergyProfiler*)arg)->power_sampler();
    }

    /* ---------------------------
       power sampling thread: runs concurrently with GPU kernels
       --------------------------- */
    // Read instantaneous board power in Watts.  nvmlDeviceGetPowerUsage returns
    // milliwatts; on devices where it is NOT_SUPPORTED, fall back to the
    // NVML_FI_DEV_POWER_INSTANT field (also milliwatts).  Returns 0.0 only if
    // both fail — never an uninitialized value.
    double read_power_watts()
    {
        unsigned int power_mW = 0;
        nvmlReturn_t st = nvmlDeviceGetPowerUsage(device, &power_mW);
        if(st != NVML_SUCCESS || power_mW == 0)
        {
            nvmlFieldValue_t fv; fv.fieldId = NVML_FI_DEV_POWER_INSTANT; fv.scopeId = 0;
            if(nvmlDeviceGetFieldValues(device, 1, &fv) == NVML_SUCCESS &&
               fv.nvmlReturn == NVML_SUCCESS &&
               fv.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)
                power_mW = fv.value.uiVal;
        }
        return power_mW / 1000.0;   // mW → W
    }

    void* power_sampler()
    {
        while(running)
        {
            double power_W = read_power_watts();

            if(sample_count < MAX_SAMPLES)
            {
                power_samples[sample_count] = power_W;
                timestamps[sample_count]    = current_time();
                sample_count++;
            }

            usleep(SAMPLE_INTERVAL_US);
        }

        return nullptr;
    }

    /* used only in destructor safety */
    void stop_dummy() { stop(); }

public:

    // start(): sync GPU, then immediately launch sampler before any kernel submissions.
    // NVML is already initialised in the constructor, so no init latency here.
    void start()
    {
        cudaDeviceSynchronize();        // ensure warmup is complete before timing
        cudaEventRecord(start_event, stream);

        sample_count = 0;
        running      = true;
        pthread_create(&sampler_thread, nullptr, sampler_trampoline, this);
    }

    // stop(): record GPU end, wait for completion, then stop the sampler.
    void stop()
    {
        cudaEventRecord(end_event, stream);
        cudaStreamSynchronize(stream);  // blocks until all kernels + end_event finish

        running = false;
        pthread_join(sampler_thread, nullptr);

        cudaEventElapsedTime(&elapsed_ms, start_event, end_event);

        energy = 0.0;
        for(int i = 1; i < sample_count; i++)
        {
            double dt = timestamps[i] - timestamps[i-1];
            energy += power_samples[i] * dt;
        }
    }

    void display(double ops=0, int iterations=1) {
        double per_it_energy    = energy / iterations;               // J per iteration
        double per_it_elapsed_s = (elapsed_ms / iterations) * 1e-3; // s per iteration

        // TOPS = ops / (time_s * 1e12)
        double tops = ((double)ops) / (per_it_elapsed_s * 1e12);

        // Power = Energy / Time;  TOPS/W = TOPS / Power
        double avg_power_W    = (per_it_elapsed_s > 0.0)
                                ? per_it_energy / per_it_elapsed_s : 0.0;
        double tops_per_watt  = (avg_power_W > 0.0) ? tops / avg_power_W : 0.0;

        printf("Energy consumption  : %.4f J\n", per_it_energy);
        printf("Time                : %.4f ms\n", per_it_elapsed_s * 1e3);
        printf("Total ops           : %.2f OPS\n", ops);
        printf("Avg power           : %.2f W\n", avg_power_W);
        printf("Throughput          : %.2f TOPS\n", tops);
        printf("Throughput/Watt     : %.4f TOPS/W\n", tops_per_watt);
    }

    // Accessors for the C API (values are valid after stop()); energy/elapsed
    // cover the whole timed window (all ITER iterations).
    double get_energy_joules() const { return energy; }
    double get_elapsed_ms()    const { return elapsed_ms; }
    int    get_sample_count()  const { return sample_count; }
    double get_avg_power_w()   const {
        double s = elapsed_ms * 1e-3;
        return s > 0.0 ? energy / s : 0.0;
    }
};
#endif
