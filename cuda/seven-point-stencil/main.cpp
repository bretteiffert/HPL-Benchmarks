// Seven-point stencil driver: parses the command line, validates the grid
// and block shape against the device limits before allocating anything,
// runs the untimed first Laplacian and the correctness check, then num_times
// event-timed Laplacians inside one energy measurement, and hands one report
// struct to the formatters.

#include <climits>
#include <iostream>
#include <memory>
#include <string>

#include "backend.h"
#include "cli.h"
#include "energy.h"
#include "report.h"
#include "stencil.h"

namespace {

std::string d3(const Dims& d)
{
  return std::to_string(d.x) + "," + std::to_string(d.y) + "," + std::to_string(d.z);
}

// Throws CliError naming the limit that a setting breaks.
void validate(const Options& o, const DeviceInfo& dev, const LaunchLimits& lim)
{
  const std::string device = "device " + std::to_string(dev.index) + " (" + dev.name + ")";
  const Dims& n = o.grid;
  const Dims& b = o.block;

  if (n.x < 3 || n.y < 3 || n.z < 3)
    throw CliError("grid " + d3(n) + " has no interior points; every side needs at least 3 points");

  if ((unsigned long long)n.x * n.y * n.z > (unsigned long long)INT_MAX)
    throw CliError("grid " + d3(n) + " has " + std::to_string((unsigned long long)n.x * n.y * n.z) +
                   " points; the kernels index with 32-bit int, so the maximum is " + std::to_string(INT_MAX));

  if (o.precision == Precision::Half)
  {
    if (!dev.half_supported)
      throw CliError("float16 is not supported on " + device + " (needs native fp16 arithmetic); use --float or --double");
    const double centre = 2.0 * (double(n.x - 1) * (n.x - 1) + double(n.y - 1) * (n.y - 1) + double(n.z - 1) * (n.z - 1));
    if (centre > 65504.0)
      throw CliError("float16 cannot hold the centre coefficient 2*((nx-1)^2+(ny-1)^2+(nz-1)^2) = " +
                     std::to_string((long long)centre) + " for grid " + d3(n) +
                     " (float16 maximum 65,504; at most 105 points per side for a cube); use a smaller grid or --float");
  }

  const char* axis[3] = {"x", "y", "z"};
  const std::size_t bv[3] = {b.x, b.y, b.z};
  const std::size_t bmax[3] = {lim.max_block.x, lim.max_block.y, lim.max_block.z};
  for (int d = 0; d < 3; d++)
    if (bmax[d] > 0 && bv[d] > bmax[d])
      throw CliError(std::string("block size ") + axis[d] + " = " + std::to_string(bv[d]) + " exceeds the " +
                     std::to_string(bmax[d]) + " allowed in " + axis[d] + " on " + device);
  if (dev.max_threads_per_block > 0 && b.count() > (std::size_t)dev.max_threads_per_block)
    throw CliError("block " + d3(b) + " has " + std::to_string(b.count()) + " threads, more than the " +
                   std::to_string(dev.max_threads_per_block) + " per block allowed on " + device);

  const Dims g = launch_grid(n, b);
  const std::size_t gv[3] = {g.x, g.y, g.z};
  const std::size_t gmax[3] = {lim.max_grid.x, lim.max_grid.y, lim.max_grid.z};
  for (int d = 0; d < 3; d++)
    if (gmax[d] > 0 && gv[d] > gmax[d])
      throw CliError(std::string("launch grid ") + axis[d] + " = " + std::to_string(gv[d]) + " blocks exceeds the " +
                     std::to_string(gmax[d]) + " allowed on " + device + "; use a larger block in " + axis[d]);

  // The test-field launch uses fixed 1024-thread blocks over x and one block
  // row per (y, z), as in the original.
  if ((dev.max_threads_per_block > 0 && kSetupBlock > (std::size_t)dev.max_threads_per_block) ||
      (lim.max_block.x > 0 && kSetupBlock > lim.max_block.x))
    throw CliError("the test-field launch needs blocks of " + std::to_string(kSetupBlock) +
                   " threads, which " + device + " does not allow");
  if ((lim.max_grid.y > 0 && n.y > lim.max_grid.y) || (lim.max_grid.z > 0 && n.z > lim.max_grid.z))
    throw CliError("the test-field launch needs ny = " + std::to_string(n.y) + " and nz = " + std::to_string(n.z) +
                   " blocks in y and z, more than the " + std::to_string(lim.max_grid.y) + " x " +
                   std::to_string(lim.max_grid.z) + " allowed on " + device);
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
  const LaunchLimits lim = launch_limits(o.device);
  validate(o, dev, lim);

  const std::size_t elem = precision_bytes(o.precision);

  Report r;
  r.implementation = backend_name();
  r.backend = backend_name();
  r.device = dev.name;
  r.driver = dev.driver;
  r.precision = o.precision;
  r.grid = o.grid;
  r.block = o.block;
  r.launch = launch_grid(o.grid, o.block);
  r.fetch_bytes = double(fetch_bytes(o.grid, elem));
  r.write_bytes = double(write_bytes(o.grid, elem));
  r.num_times = o.num_times;
  r.timing = "device timestamps (device synchronized before each run; caches are not flushed)";
  r.header_notes = dev.notes;
  r.laplacian.name = "Laplacian";
  r.laplacian.bytes = r.fetch_bytes + r.write_bytes;

  EnergyMeter meter(o.device, o.energy);
  r.energy.enabled = meter.enabled();

  print_text_header(r, text_out);

  std::unique_ptr<Stencil> st = make_stencil(o.precision, o.grid, o.block);

  // Untimed first run, then the check (outside the measured window).
  st->first_run();
  r.correctness = st->check();

  meter.start();
  for (int it = 0; it < o.num_times; it++)
    r.laplacian.times.push_back(st->timed_run());
  meter.stop();
  r.energy = meter.result(static_cast<unsigned long>(o.num_times));

  switch (o.output)
  {
    case OutputMode::Text:   print_text_results(r, std::cout); break;
    case OutputMode::Csv:    print_csv(r, std::cout); break;
    case OutputMode::CsvRaw: print_csv_raw(r, std::cout); break;
  }
  print_text_energy(r, text_out);
  print_text_correctness(r, text_out);
  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    const Options o = parse_cli(argc, argv);

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
      for (const auto& d : devices) std::cout << d.first << ": " << d.second << "\n";
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
