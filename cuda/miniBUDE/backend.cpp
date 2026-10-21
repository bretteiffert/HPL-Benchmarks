// CUDA backend: device list and device facts.
//
// Lookup order for every fact: override env var (GPU_NAME, GPU_DRIVER) ->
// CUDA runtime -> vendor library loaded at run time (libcuda, NVML) ->
// placeholder plus a note.

#include "backend.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

std::string env_or_empty(const char* name)
{
  const char* v = std::getenv(name);
  return (v != nullptr) ? std::string(v) : std::string();
}

void check(cudaError_t err, const char* what)
{
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(err));
}

// ---- vendor libraries, loaded at run time --------------------------------

// libcuda (driver API). CUdevice is an int; CUresult 0 is success.
struct LibCuda
{
  void* handle = nullptr;
  int (*cuInit)(unsigned int) = nullptr;
  int (*cuDeviceGet)(int*, int) = nullptr;
  int (*cuDeviceGetName)(char*, int, int) = nullptr;
  int (*cuDeviceGetAttribute)(int*, int, int) = nullptr;
  int (*cuDriverGetVersion)(int*) = nullptr;

  LibCuda()
  {
    handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) handle = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) return;
    cuInit               = reinterpret_cast<int (*)(unsigned int)>(dlsym(handle, "cuInit"));
    cuDeviceGet          = reinterpret_cast<int (*)(int*, int)>(dlsym(handle, "cuDeviceGet"));
    cuDeviceGetName      = reinterpret_cast<int (*)(char*, int, int)>(dlsym(handle, "cuDeviceGetName"));
    cuDeviceGetAttribute = reinterpret_cast<int (*)(int*, int, int)>(dlsym(handle, "cuDeviceGetAttribute"));
    cuDriverGetVersion   = reinterpret_cast<int (*)(int*)>(dlsym(handle, "cuDriverGetVersion"));
    if (cuInit == nullptr || cuInit(0) != 0) { dlclose(handle); handle = nullptr; }
  }
  ~LibCuda() { if (handle != nullptr) dlclose(handle); }
  LibCuda(const LibCuda&) = delete;
  LibCuda& operator=(const LibCuda&) = delete;

  bool device(int index, int& dev) const
  {
    return handle != nullptr && cuDeviceGet != nullptr && cuDeviceGet(&dev, index) == 0;
  }

  std::string name(int index) const
  {
    int dev = 0;
    char buf[256] = {0};
    if (!device(index, dev) || cuDeviceGetName == nullptr) return {};
    if (cuDeviceGetName(buf, static_cast<int>(sizeof(buf)), dev) != 0) return {};
    return std::string(buf);
  }

  int sm_count(int index) const
  {
    const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    int dev = 0, value = 0;
    if (!device(index, dev) || cuDeviceGetAttribute == nullptr) return 0;
    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) != 0) return 0;
    return value;
  }

  int driver_version() const
  {
    int v = 0;
    if (handle == nullptr || cuDriverGetVersion == nullptr || cuDriverGetVersion(&v) != 0) return 0;
    return v;
  }
};

// NVML: display-driver version string such as "550.54.15".
std::string nvml_driver_version()
{
  void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (h == nullptr) h = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
  if (h == nullptr) return {};

  auto init  = reinterpret_cast<int (*)()>(dlsym(h, "nvmlInit_v2"));
  auto get   = reinterpret_cast<int (*)(char*, unsigned int)>(dlsym(h, "nvmlSystemGetDriverVersion"));
  auto close = reinterpret_cast<int (*)()>(dlsym(h, "nvmlShutdown"));

  std::string out;
  if (init != nullptr && get != nullptr && init() == 0)
  {
    char buf[96] = {0};
    if (get(buf, static_cast<unsigned int>(sizeof(buf))) == 0) out = buf;
    if (close != nullptr) close();
  }
  dlclose(h);
  return out;
}

// 12040 -> "12.4"
std::string cuda_version_string(int v)
{
  return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

} // namespace

const char* backend_name() { return "CUDA"; }

const char* compute_unit_label() { return "SMs"; }

int device_count()
{
  int count = 0;
  check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  return count;
}

std::vector<std::pair<int, std::string>> list_devices()
{
  std::vector<std::pair<int, std::string>> out;
  const int count = device_count();
  for (int i = 0; i < count; i++)
  {
    cudaDeviceProp props;
    std::string name;
    if (cudaGetDeviceProperties(&props, i) == cudaSuccess) name = props.name;
    if (name.empty())
    {
      LibCuda lib;
      name = lib.name(i);
    }
    if (name.empty()) name = std::string("CUDA device ") + std::to_string(i);
    out.emplace_back(i, name);
  }
  return out;
}

DeviceInfo query_device(int index)
{
  DeviceInfo info;
  info.index = index;

  check(cudaSetDevice(index), "cudaSetDevice");

  cudaDeviceProp props;
  check(cudaGetDeviceProperties(&props, index), "cudaGetDeviceProperties");

  info.max_threads_per_block = props.maxThreadsPerBlock;
  info.max_grid_x = props.maxGridSize[0];
  info.warp_size = props.warpSize;
  info.half_supported = (props.major * 10 + props.minor) >= 53;
  info.compute_units = props.multiProcessorCount;

  // Name
  info.name = env_or_empty("GPU_NAME");
  if (info.name.empty()) info.name = props.name;

  // libcuda is only loaded when the runtime left a gap.
  std::unique_ptr<LibCuda> libcuda;
  auto lib = [&libcuda]() -> const LibCuda& {
    if (!libcuda) libcuda.reset(new LibCuda());
    return *libcuda;
  };

  if (info.name.empty()) info.name = lib().name(index);
  if (info.name.empty())
  {
    info.name = std::string("CUDA device ") + std::to_string(index);
    info.notes.push_back("device name unavailable from the CUDA runtime and libcuda; set GPU_NAME to override");
  }

  // SM count
  if (info.compute_units <= 0) info.compute_units = lib().sm_count(index);

  // Driver
  info.driver = env_or_empty("GPU_DRIVER");
  if (info.driver.empty())
  {
    int cuda_driver = 0;
    if (cudaDriverGetVersion(&cuda_driver) != cudaSuccess || cuda_driver <= 0)
      cuda_driver = lib().driver_version();

    const std::string display = nvml_driver_version();
    if (!display.empty() && cuda_driver > 0)
      info.driver = display + " (CUDA " + cuda_version_string(cuda_driver) + ")";
    else if (!display.empty())
      info.driver = display;
    else if (cuda_driver > 0)
      info.driver = "CUDA " + cuda_version_string(cuda_driver);
    else
      info.driver = "unknown";
  }

  return info;
}
