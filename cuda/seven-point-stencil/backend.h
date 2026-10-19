// CUDA device list and device-fact lookup (name, driver, SM count).
// Shared unchanged by every CUDA benchmark in this suite.

#pragma once

#include <string>
#include <utility>
#include <vector>

struct DeviceInfo
{
  int index = 0;
  std::string name;            // GPU_NAME > runtime > vendor library > placeholder
  std::string driver;          // GPU_DRIVER > runtime/vendor library > "unknown"
  int compute_units = 0;       // SMs; 0 when unknown
  int max_threads_per_block = 0;
  int max_grid_x = 0;
  int warp_size = 0;
  bool half_supported = false; // native fp16 arithmetic on this device
  std::vector<std::string> notes;
};

// "CUDA"
const char* backend_name();

// "SMs"
const char* compute_unit_label();

// Number of visible devices. Throws std::runtime_error on runtime failure.
int device_count();

// Index and name of every visible device.
std::vector<std::pair<int, std::string>> list_devices();

// Make the device current and collect its facts. Throws on runtime failure.
DeviceInfo query_device(int index);
