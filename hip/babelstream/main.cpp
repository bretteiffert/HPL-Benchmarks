// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

// BabelStream driver: parses the command line, validates the configuration
// against the device limits before allocating anything, runs the five
// kernels num_times times with device-event timing (the first iteration is
// warm-up and is excluded from statistics, CSV and energy), and hands one
// report struct to the formatters.

#include <climits>
#include <iostream>
#include <memory>
#include <string>

#include "backend.h"
#include "cli.h"
#include "energy.h"
#include "report.h"
#include "stream.h"

namespace {

bool is_power_of_two(long long v) { return v > 0 && (v & (v - 1)) == 0; }

// Throws CliError naming the limit that a setting breaks.
void validate(const Options& o, const DeviceInfo& dev, int dot_groups)
{
  const std::string device = "device " + std::to_string(dev.index) + " (" + dev.name + ")";

  if (o.precision == Precision::Half && !dev.half_supported)
    throw CliError("float16 is not supported on " + device +
                   " (needs native fp16 arithmetic); use --float or --double");

  if (!is_power_of_two(o.block_size))
    throw CliError("block size " + std::to_string(o.block_size) +
                   " must be a power of two (the Dot reduction halves the block each step)");

  if (o.block_size > kMaxBlockSize)
    throw CliError("block size " + std::to_string(o.block_size) + " exceeds the Dot reduction's " +
                   std::to_string(kMaxBlockSize) + " shared-memory slots");

  if (dev.max_threads_per_block > 0 && o.block_size > dev.max_threads_per_block)
    throw CliError("block size " + std::to_string(o.block_size) + " exceeds the " +
                   std::to_string(dev.max_threads_per_block) + " threads per block allowed on " + device);

  if (o.array_size > INT_MAX)
    throw CliError("array size " + std::to_string(o.array_size) +
                   " does not fit the kernels' 32-bit index (maximum " + std::to_string(INT_MAX) + ")");

  if (o.array_size % o.block_size != 0)
    throw CliError("array size " + std::to_string(o.array_size) + " is not a multiple of the block size " +
                   std::to_string(o.block_size) + " (the kernels have no bounds check)");

  const long long dot_stride = static_cast<long long>(o.block_size) * dot_groups;
  if (o.array_size + dot_stride > INT_MAX)
    throw CliError("array size " + std::to_string(o.array_size) + " plus the Dot grid stride " +
                   std::to_string(dot_stride) + " overflows the kernels' 32-bit index (maximum " +
                   std::to_string(INT_MAX) + ")");

  const long long grid = o.array_size / o.block_size;
  if (dev.max_grid_x > 0 && grid > dev.max_grid_x)
    throw CliError("grid of " + std::to_string(grid) + " blocks exceeds the device maximum of " +
                   std::to_string(dev.max_grid_x) + "; use a larger --block-size");
}

int run(const Options& o)
{
  const bool csv_mode = (o.output != OutputMode::Text);
  std::ostream& text_out = csv_mode ? std::cerr : std::cout;

  const int count = device_count();
  if (count == 0)
    throw std::runtime_error(std::string("no ") + backend_name() + " devices found");
  if (o.device >= count)
    throw CliError("device index " + std::to_string(o.device) + " is out of range (" +
                   std::to_string(count) + (count == 1 ? " device" : " devices") + " found; see --list)");

  const DeviceInfo dev = query_device(o.device);

  Report r;
  r.implementation = backend_name();
  r.backend = backend_name();
  r.device = dev.name;
  r.driver = dev.driver;
  r.precision = o.precision;
  r.array_size = o.array_size;
  r.block_size = o.block_size;
  r.compute_units = dev.compute_units;
  r.compute_unit_label = compute_unit_label();
  r.num_times = o.num_times;
  r.fmad = fmad_setting();
  r.timing = "device timestamps";
  r.mibibytes = o.mibibytes;
  r.header_notes = dev.notes;

  if (dev.compute_units <= 0)
    throw std::runtime_error("compute-unit count is unavailable; the pinned Dot configuration requires 4 x CU groups");
  r.dot_groups = 4 * dev.compute_units;

  validate(o, dev, r.dot_groups);

  EnergyMeter meter(o.device, o.energy);
  r.energy.enabled = meter.enabled();

  print_text_header(r, text_out);

  StreamConfig cfg;
  cfg.array_size = static_cast<int>(o.array_size);
  cfg.block_size = o.block_size;
  cfg.dot_groups = r.dot_groups;
  std::unique_ptr<Stream> stream = make_stream(o.precision, cfg);

  const double elem = static_cast<double>(precision_bytes(o.precision));
  const double n = static_cast<double>(o.array_size);

  RoutineResult init  {"Init",  3 * elem * n, true,  {}};
  RoutineResult copy  {"Copy",  2 * elem * n, false, {}};
  RoutineResult mul   {"Mul",   2 * elem * n, false, {}};
  RoutineResult add   {"Add",   3 * elem * n, false, {}};
  RoutineResult triad {"Triad", 3 * elem * n, false, {}};
  RoutineResult dot   {"Dot",   2 * elem * n, false, {}};

  init.times.push_back(stream->init());

  for (unsigned int k = 0; k < o.num_times; k++)
  {
    // Iteration 0 is warm-up: excluded from statistics, CSV and energy.
    const bool timed = (k > 0);
    if (k == 1) meter.start();

    const double t_copy  = stream->copy();
    const double t_mul   = stream->mul();
    const double t_add   = stream->add();
    const double t_triad = stream->triad();
    const double t_dot   = stream->dot();

    if (timed)
    {
      copy.times.push_back(t_copy);
      mul.times.push_back(t_mul);
      add.times.push_back(t_add);
      triad.times.push_back(t_triad);
      dot.times.push_back(t_dot);
    }
  }
  meter.stop();

  r.routines = {init, copy, mul, add, triad, dot};
  r.energy = meter.result(o.num_times - 1);
  r.correctness = stream->check(o.num_times);

  switch (o.output)
  {
    case OutputMode::Text:
      print_text_results(r, std::cout);
      break;
    case OutputMode::Csv:
      print_csv(r, std::cout);
      break;
    case OutputMode::CsvRaw:
      print_csv_raw(r, std::cout);
      break;
  }
  print_text_energy(r, text_out);
  print_text_correctness(r, text_out);

  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  Options o;
  try
  {
    o = parse_cli(argc, argv);

    if (o.help)
    {
      std::cout << help_text(argv[0]);
      return 0;
    }

    if (o.list)
    {
      const auto devices = list_devices();
      if (devices.empty())
      {
        std::cerr << "error: no " << backend_name() << " devices found" << std::endl;
        return 1;
      }
      for (const auto& d : devices)
        std::cout << d.first << ": " << d.second << "\n";
      return 0;
    }

    return run(o);
  }
  catch (const CliError& e)
  {
    std::cerr << "error: " << e.what() << " (try '--help')" << std::endl;
    return 1;
  }
  catch (const std::exception& e)
  {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
