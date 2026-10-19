// miniBUDE on CUDA: device buffers, launch geometry, event-timed launches
// with paired host timestamps, and the untimed fp64 reference pass. This is
// the host half of the original fasten.hpp CudaBude class (allocation modes,
// context timing, launch geometry, timing loop, energy window), reorganised
// as functions. Nothing here prints; failures throw.

#include "bude.h"
#include "deck.h"
#include "kernels.cuh"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void checkError(const cudaError_t err, const char* what)
{
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorName(err) + ": " +
                             cudaGetErrorString(err));
}

template <typename U> U* allocate(std::size_t size)
{
  U* data = nullptr;
#if defined(MANAGED)
  checkError(cudaMallocManaged(&data, sizeof(U) * size), "cudaMallocManaged");
#elif defined(PAGEFAULT)
  data = (U*)std::malloc(sizeof(U) * size);
  if (data == nullptr) throw std::runtime_error("malloc failed");
#else
  checkError(cudaMalloc(&data, sizeof(U) * size), "cudaMalloc");
#endif
  return data;
}

template <typename U> U* allocate(const std::vector<U>& xs)
{
  U* data = allocate<U>(xs.size());
  checkError(cudaMemcpy(data, xs.data(), xs.size() * sizeof(U), cudaMemcpyHostToDevice), "cudaMemcpy");
  return data;
}

template <typename U> void deallocate(U* data)
{
  if (data == nullptr) return;
#if defined(PAGEFAULT)
  std::free(data);
#else
  (void)cudaFree(data);   // best effort: cleanup never throws
#endif
}

using Clock = std::chrono::high_resolution_clock;

double seconds_between(Clock::time_point a, Clock::time_point b)
{
  return std::chrono::duration<double>(b - a).count();
}

// Device buffers of one configuration; freed on scope exit.
template <typename T> struct Buffers
{
  AtomT<T>* protein = nullptr;
  AtomT<T>* ligand = nullptr;
  T* transforms[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  FFParamsT<T>* forcefield = nullptr;
  T* results = nullptr;

  explicit Buffers(const Deck& d)
  {
    try
    {
      protein = allocate(convertAtoms<T>(d.protein));
      ligand = allocate(convertAtoms<T>(d.ligand));
      for (int i = 0; i < 6; i++) transforms[i] = allocate(convertPoses<T>(d.poses[i]));
      forcefield = allocate(convertForcefield<T>(d.forcefield));
      results = allocate<T>(d.nposes());
      checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
    catch (...)
    {
      release();
      throw;
    }
  }
  ~Buffers() { release(); }
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  void release()
  {
    deallocate(protein);
    deallocate(ligand);
    for (auto& t : transforms) deallocate(t);
    deallocate(forcefield);
    deallocate(results);
    protein = nullptr; ligand = nullptr; forcefield = nullptr; results = nullptr;
    for (auto& t : transforms) t = nullptr;
  }

  std::vector<double> energies(std::size_t n) const
  {
    std::vector<T> raw(n);
    checkError(cudaMemcpy(raw.data(), results, raw.size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
    return std::vector<double>(raw.begin(), raw.end());
  }
};

struct Geometry
{
  std::size_t global, local, shared;
};

template <typename T> Geometry geometry(const Deck& d, std::size_t ppwi, std::size_t wgsize)
{
  std::size_t global = std::size_t(std::ceil(double(d.nposes()) / double(ppwi)));
  global = std::size_t(std::ceil(double(global) / double(wgsize)));
#ifdef USE_SHARED
  const std::size_t shared = d.ntypes() * sizeof(FFParamsT<T>);
#else
  const std::size_t shared = 0;
#endif
  return {global, wgsize, shared};
}

template <typename T, std::size_t PPWI>
void launch(const Deck& d, const Buffers<T>& b, const Geometry& g)
{
  fasten_main<T, PPWI><<<g.global, g.local, g.shared>>>(
      int(d.natlig()), int(d.natpro()), int(d.ntypes()), b.protein, b.ligand,
      b.transforms[0], b.transforms[1], b.transforms[2], b.transforms[3], b.transforms[4], b.transforms[5],
      b.results, b.forcefield, int(d.nposes()));
  checkError(cudaGetLastError(), "fasten_main launch");
}

struct Events
{
  cudaEvent_t start = nullptr, stop = nullptr;
  Events()
  {
    checkError(cudaEventCreate(&start), "cudaEventCreate");
    if (cudaEventCreate(&stop) != cudaSuccess)
    {
      (void)cudaEventDestroy(start);
      throw std::runtime_error("CUDA cudaEventCreate failed");
    }
  }
  ~Events()
  {
    (void)cudaEventDestroy(start);
    (void)cudaEventDestroy(stop);
  }
};

template <typename T, std::size_t PPWI>
Sample run_typed(const Deck& d, std::size_t wgsize, int device, std::size_t warmup, std::size_t iterations,
                 EnergyMeter& meter)
{
  checkError(cudaSetDevice(device), "cudaSetDevice");

  Sample sample;
  sample.ppwi = PPWI;
  sample.wgsize = wgsize;

  const auto contextStart = Clock::now();
  Buffers<T> buffers(d);
  const auto contextEnd = Clock::now();
  sample.context_s = seconds_between(contextStart, contextEnd);

  const Geometry g = geometry<T>(d, PPWI, wgsize);
  Events ev;

  for (std::size_t i = 0; i < warmup + iterations; ++i)
  {
    if (i == warmup) meter.start();

    const auto wallStart = Clock::now();
    checkError(cudaEventRecord(ev.start), "cudaEventRecord");
    launch<T, PPWI>(d, buffers, g);
    checkError(cudaEventRecord(ev.stop), "cudaEventRecord");
    checkError(cudaEventSynchronize(ev.stop), "cudaEventSynchronize");
    const auto wallEnd = Clock::now();

    float deviceMs = 0.0f;
    checkError(cudaEventElapsedTime(&deviceMs, ev.start, ev.stop), "cudaEventElapsedTime");
    if (!(deviceMs > 0.0f))
      throw std::runtime_error("device timer returned no measurement for ppwi " + std::to_string(PPWI) +
                               ", wgsize " + std::to_string(wgsize));

    if (i >= warmup)
    {
      sample.kernel_s.push_back(double(deviceMs) * 1e-3);
      sample.wall_s.push_back(seconds_between(wallStart, wallEnd));
    }
  }
  meter.stop();
  sample.energy = meter.result(static_cast<unsigned long>(iterations));

  sample.energies = buffers.energies(d.nposes());
  return sample;
}

template <typename T, std::size_t PPWI>
std::vector<double> reference_typed(const Deck& d, std::size_t wgsize, int device)
{
  checkError(cudaSetDevice(device), "cudaSetDevice");
  Buffers<T> buffers(d);
  launch<T, PPWI>(d, buffers, geometry<T>(d, PPWI, wgsize));
  checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  return buffers.energies(d.nposes());
}

// Run-time ppwi -> compile-time PPWI.
template <typename T, typename F> auto dispatch(std::size_t ppwi, F&& f)
{
  switch (ppwi)
  {
    case 1:   return f(std::integral_constant<std::size_t, 1>{});
    case 2:   return f(std::integral_constant<std::size_t, 2>{});
    case 4:   return f(std::integral_constant<std::size_t, 4>{});
    case 8:   return f(std::integral_constant<std::size_t, 8>{});
    case 16:  return f(std::integral_constant<std::size_t, 16>{});
    case 32:  return f(std::integral_constant<std::size_t, 32>{});
    case 64:  return f(std::integral_constant<std::size_t, 64>{});
    case 128: return f(std::integral_constant<std::size_t, 128>{});
  }
  throw std::runtime_error("ppwi " + std::to_string(ppwi) + " is not compiled in");
}

} // namespace

Sample run_config(Precision p, const Deck& deck, std::size_t ppwi, std::size_t wgsize, int device,
                  std::size_t warmup, std::size_t iterations, EnergyMeter& meter)
{
  if (p == Precision::Double)
    return dispatch<double>(ppwi, [&](auto n) {
      return run_typed<double, decltype(n)::value>(deck, wgsize, device, warmup, iterations, meter);
    });
  return dispatch<float>(ppwi, [&](auto n) {
    return run_typed<float, decltype(n)::value>(deck, wgsize, device, warmup, iterations, meter);
  });
}

std::vector<double> reference_pass(const Deck& deck, std::size_t ppwi, std::size_t wgsize, int device)
{
  return dispatch<double>(ppwi, [&](auto n) { return reference_typed<double, decltype(n)::value>(deck, wgsize, device); });
}

std::string kernel_arch_note(int device)
{
  // The original compatible() check: the version kernel reports the
  // __CUDA_ARCH__ the kernels were built for.
  checkError(cudaSetDevice(device), "cudaSetDevice");
  int* data = nullptr;
  checkError(cudaMalloc(&data, sizeof(int)), "cudaMalloc");
  version<<<1, 1>>>(data);
  int rawKernelCC = -2;
  const cudaError_t launchErr = cudaGetLastError();
  if (launchErr == cudaSuccess)
    (void)cudaMemcpy(&rawKernelCC, data, sizeof(rawKernelCC), cudaMemcpyDeviceToHost);
  (void)cudaDeviceSynchronize();
  (void)cudaFree(data);

  cudaDeviceProp props{};
  checkError(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");
  const int deviceCC = props.major * 10 + props.minor;

  std::string reason;
  switch (rawKernelCC)
  {
    case 0:  reason = "the check kernel returned a bad result"; break;
    case -1: reason = "__CUDA_ARCH__ was not defined"; break;
    case -2: reason = "the check kernel failed to run"; break;
    default:
      if (rawKernelCC / 10 == deviceCC) return "";
      return "kernels built for sm_" + std::to_string(rawKernelCC / 10) + " but the device is sm_" +
             std::to_string(deviceCC) + "; results may be incorrect, rebuild with make ARCH=" +
             std::to_string(deviceCC);
  }
  return "could not confirm the kernels' architecture (" + reason + "); the device is sm_" +
         std::to_string(deviceCC);
}

std::size_t shared_mem_per_block(int device)
{
  cudaDeviceProp props{};
  checkError(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");
  return props.sharedMemPerBlock;
}

bool shared_forcefield()
{
#ifdef USE_SHARED
  return true;
#else
  return false;
#endif
}

const char* allocation_mode()
{
#if defined(MANAGED)
  return "managed (cudaMallocManaged)";
#elif defined(PAGEFAULT)
  return "pagefault (host malloc)";
#else
  return "device (cudaMalloc)";
#endif
}

bool restrict_pointers()
{
#ifdef NO_RESTRICT
  return false;
#else
  return true;
#endif
}
