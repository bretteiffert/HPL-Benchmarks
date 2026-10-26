// Board-power measurement through libkernelenergy (../../dependencies).

#include "energy.h"

#include "kernel_energy_capi.h"

#include <stdexcept>
#include <string>

EnergyMeter::EnergyMeter(int device, bool enabled)
{
  if (!enabled) return;
  handle_ = energy_create(device, nullptr);
  if (handle_ == nullptr)
    throw std::runtime_error("could not start the board-power profiler on device " +
                             std::to_string(device) + " (run with --no-energy to skip it)");
}

EnergyMeter::~EnergyMeter()
{
  if (handle_ != nullptr)
  {
    if (running_) energy_stop(handle_);
    energy_destroy(handle_);
  }
}

void EnergyMeter::start()
{
  t0_ = std::chrono::steady_clock::now();
  if (handle_ != nullptr) energy_start(handle_);
  running_ = true;
}

void EnergyMeter::stop()
{
  if (!running_) return;
  if (handle_ != nullptr) energy_stop(handle_);
  seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  running_ = false;
}

EnergyResult EnergyMeter::result(unsigned long iterations) const
{
  EnergyResult r;
  r.enabled = (handle_ != nullptr);
  r.seconds = seconds_;
  r.iterations = iterations;
  if (!r.enabled) return r;

  r.joules = energy_joules(handle_);
  r.samples = energy_sample_count(handle_);
  r.joules_per_iter = (iterations > 0) ? r.joules / static_cast<double>(iterations) : 0.0;
  r.total_wh = r.joules / 3600.0;
  r.avg_power_w = (seconds_ > 0.0) ? r.joules / seconds_ : 0.0;
  return r;
}
