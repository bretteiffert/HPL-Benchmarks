// HIP backend: device list and device facts.
//
// Lookup order for every fact: override env var (GPU_NAME, GPU_DRIVER) ->
// HIP runtime -> vendor library loaded at run time (libamdhip64) ->
// placeholder plus a note.

#include "backend.h"

#include <hip/hip_runtime.h>
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

void check(hipError_t err, const char* what)
{
  if (err != hipSuccess)
    throw std::runtime_error(std::string("HIP ") + what + ": " + hipGetErrorString(err));
}

// ---- vendor library, loaded at run time ----------------------------------

// libamdhip64 C API. hipDevice_t is an int; hipError_t 0 is success.
struct LibHip
{
  void* handle = nullptr;
  int (*hipDeviceGetName_)(char*, int, int) = nullptr;
  int (*hipDeviceGetAttribute_)(int*, int, int) = nullptr;
  int (*hipDriverGetVersion_)(int*) = nullptr;

  LibHip()
  {
    handle = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) handle = dlopen("libamdhip64.so.6", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) return;
    hipDeviceGetName_      = reinterpret_cast<int (*)(char*, int, int)>(dlsym(handle, "hipDeviceGetName"));
    hipDeviceGetAttribute_ = reinterpret_cast<int (*)(int*, int, int)>(dlsym(handle, "hipDeviceGetAttribute"));
    hipDriverGetVersion_   = reinterpret_cast<int (*)(int*)>(dlsym(handle, "hipDriverGetVersion"));
  }
  ~LibHip() { if (handle != nullptr) dlclose(handle); }
  LibHip(const LibHip&) = delete;
  LibHip& operator=(const LibHip&) = delete;

  std::string name(int index) const
  {
    char buf[256] = {0};
    if (hipDeviceGetName_ == nullptr) return {};
    if (hipDeviceGetName_(buf, static_cast<int>(sizeof(buf)), index) != 0) return {};
    return std::string(buf);
  }

  int cu_count(int index) const
  {
    const int hipDeviceAttributeMultiprocessorCount_ = 63;   // hip_runtime_api.h, ROCm 6.x
    int value = 0;
    if (hipDeviceGetAttribute_ == nullptr) return 0;
    if (hipDeviceGetAttribute_(&value, hipDeviceAttributeMultiprocessorCount_, index) != 0) return 0;
    return value;
  }

  int driver_version() const
  {
    int v = 0;
    if (hipDriverGetVersion_ == nullptr || hipDriverGetVersion_(&v) != 0) return 0;
    return v;
  }
};

// 60342134 -> "6.3.42134" (major * 10^7 + minor * 10^5 + patch)
std::string hip_version_string(int v)
{
  return std::to_string(v / 10000000) + "." + std::to_string((v / 100000) % 100) + "." +
         std::to_string(v % 100000);
}

} // namespace

const char* backend_name() { return "HIP"; }

const char* compute_unit_label() { return "CUs"; }

int device_count()
{
  int count = 0;
  check(hipGetDeviceCount(&count), "hipGetDeviceCount");
  return count;
}

std::vector<std::pair<int, std::string>> list_devices()
{
  std::vector<std::pair<int, std::string>> out;
  const int count = device_count();
  for (int i = 0; i < count; i++)
  {
    hipDeviceProp_t props;
    std::string name;
    if (hipGetDeviceProperties(&props, i) == hipSuccess) name = props.name;
    if (name.empty())
    {
      LibHip lib;
      name = lib.name(i);
    }
    if (name.empty()) name = std::string("HIP device ") + std::to_string(i);
    out.emplace_back(i, name);
  }
  return out;
}

DeviceInfo query_device(int index)
{
  DeviceInfo info;
  info.index = index;

  check(hipSetDevice(index), "hipSetDevice");

  hipDeviceProp_t props;
  check(hipGetDeviceProperties(&props, index), "hipGetDeviceProperties");

  info.max_threads_per_block = props.maxThreadsPerBlock;
  info.max_grid_x = props.maxGridSize[0];
  info.warp_size = props.warpSize;
  info.half_supported = true;   // fp16 arithmetic is native on every ROCm-supported GPU
  info.compute_units = props.multiProcessorCount;

  // libamdhip64 is only loaded when the runtime left a gap.
  std::unique_ptr<LibHip> libhip;
  auto lib = [&libhip]() -> const LibHip& {
    if (!libhip) libhip.reset(new LibHip());
    return *libhip;
  };

  // Name
  info.name = env_or_empty("GPU_NAME");
  if (info.name.empty()) info.name = props.name;
  if (info.name.empty()) info.name = lib().name(index);
  if (info.name.empty())
  {
    info.name = std::string("HIP device ") + std::to_string(index);
    info.notes.push_back("device name unavailable from the HIP runtime and libamdhip64; set GPU_NAME to override");
  }

  // CU count
  if (info.compute_units <= 0) info.compute_units = lib().cu_count(index);

  // Driver
  info.driver = env_or_empty("GPU_DRIVER");
  if (info.driver.empty())
  {
    int v = 0;
    if (hipDriverGetVersion(&v) != hipSuccess || v <= 0) v = lib().driver_version();
    info.driver = (v > 0) ? "HIP " + hip_version_string(v) : "unknown";
  }

  return info;
}
