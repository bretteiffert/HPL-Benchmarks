// Board-power measurement through libkernelenergy. Shared unchanged by every
// benchmark in this suite. Sampling covers only the span between start() and
// stop(), which callers place around the timed iterations.

#pragma once

#include <chrono>

struct EnergyResult
{
  bool enabled = false;
  double joules = 0.0;
  double joules_per_iter = 0.0;
  double total_wh = 0.0;
  double avg_power_w = 0.0;
  double seconds = 0.0;      // wall time between start() and stop()
  unsigned long iterations = 0;
  int samples = 0;
};

class EnergyMeter
{
  public:
    // enabled == false gives a meter that does nothing (--no-energy).
    // Throws std::runtime_error if the profiler cannot be created.
    EnergyMeter(int device, bool enabled);
    ~EnergyMeter();

    EnergyMeter(const EnergyMeter&) = delete;
    EnergyMeter& operator=(const EnergyMeter&) = delete;

    bool enabled() const { return handle_ != nullptr; }

    void start();
    void stop();

    // Totals for the sampled span, divided over `iterations`.
    EnergyResult result(unsigned long iterations) const;

  private:
    void* handle_ = nullptr;
    bool running_ = false;
    std::chrono::steady_clock::time_point t0_;
    double seconds_ = 0.0;
};
