// miniBUDE driver: parses the command line, selects the device, reads the
// deck, validates every configuration against the device limits before
// allocating anything (non-divisible ones are skipped with a note), runs one
// untimed fp64 pass as the reference for float32 runs, then runs each
// (ppwi, wgsize) configuration with its own energy window, and hands one
// report struct to the formatters. The figures of merit and the metrics are
// computed exactly as in the original evaluate().

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

#include "backend.h"
#include "bude.h"
#include "cli.h"
#include "deck.h"
#include "energy.h"
#include "metrics.h"
#include "report.h"

namespace {

int select_device(const std::string& needle)
{
  const auto devices = list_devices();
  if (devices.empty())
    throw std::runtime_error(std::string("no ") + backend_name() + " devices found");

  bool numeric = !needle.empty() && needle.find_first_not_of("0123456789") == std::string::npos;
  if (numeric)
  {
    const long long idx = std::stoll(needle);
    if (idx >= static_cast<long long>(devices.size()))
      throw CliError("device index " + needle + " is out of range (" + std::to_string(devices.size()) +
                     (devices.size() == 1 ? " device" : " devices") + " found; see --list)");
    return static_cast<int>(idx);
  }
  for (const auto& d : devices)
    if (d.second.find(needle) != std::string::npos) return d.first;
  throw CliError("no device name contains '" + needle + "' (see --list)");
}

struct Config
{
  std::size_t ppwi, wgsize;
};

ConfigResult evaluate(const Deck& d, Sample s, const std::vector<double>* reference, Precision p)
{
  ConfigResult r;
  const auto& t = s.kernel_s;
  r.min_s = *std::min_element(t.begin(), t.end());
  r.max_s = *std::max_element(t.begin(), t.end());
  const double sum = std::accumulate(t.begin(), t.end(), 0.0);
  r.avg_s = sum / double(t.size());
  double var = 0.0;
  for (double x : t) var += std::pow(x - r.avg_s, 2.0);
  r.stddev_s = std::sqrt(var / double(t.size()));

  const double runtime = r.avg_s;
  const std::size_t ppwi = s.ppwi;
  std::size_t ops_per_wg = ppwi * 27 + d.natlig() * (2 + ppwi * 18 + d.natpro() * (10 + ppwi * 30)) + ppwi;
  double total_ops = double(ops_per_wg) * (double(d.nposes()) / double(ppwi));
  r.gflops = total_ops / runtime / 1e9;

  std::size_t total_finsts = 25 * d.natpro() * d.natlig() * d.nposes();
  r.gfinsts = double(total_finsts) / runtime / 1e9;

  std::size_t interactions = d.nposes() * d.natlig() * d.natpro();
  r.ginteractions = double(interactions) / runtime / 1e9;

  double acc = 0.0;
  for (std::size_t i = 0; i < t.size(); ++i) acc += s.wall_s[i] - t[i];
  r.launch_overhead_s = acc / double(t.size());

  r.has_ref = reference != nullptr;
  if (r.has_ref) r.vs_ref = computeMetrics(s.energies, *reference, precision_bits(p));
  r.vs_deck = compareToDeck(s.energies, d.refEnergies);

  r.sample = std::move(s);
  return r;
}

std::string build_string()
{
  std::string s = shared_forcefield() ? "forcefield in shared memory" : "forcefield in global memory";
  s += ", ";
  s += allocation_mode();
  s += " allocation, ";
  s += restrict_pointers() ? "__restrict__ on" : "__restrict__ off (NO_RESTRICT)";
  return s;
}

int run(const Options& o)
{
  const bool csv_mode = (o.output != OutputMode::Text);
  std::ostream& text_out = csv_mode ? std::cerr : std::cout;

  const int device = select_device(o.device);
  const DeviceInfo dev = query_device(device);
  const std::string devname = "device " + std::to_string(dev.index) + " (" + dev.name + ")";

  const Deck deck = read_deck(o.deck, o.poses);

  // ---- validation, before any allocation
  if (deck.nposes() > static_cast<std::size_t>(INT_MAX))
    throw CliError(std::to_string(deck.nposes()) + " poses exceed the kernel's 32-bit pose index (maximum " +
                   std::to_string(INT_MAX) + "); use -n");

  for (auto w : o.wgsizes)
    if (dev.max_threads_per_block > 0 && w > static_cast<std::size_t>(dev.max_threads_per_block))
      throw CliError("work-group size " + std::to_string(w) + " exceeds the " +
                     std::to_string(dev.max_threads_per_block) + " threads per block allowed on " + devname);

  if (shared_forcefield())
  {
    const std::size_t need = deck.ntypes() * (o.precision == Precision::Double ? sizeof(FFParamsT<double>)
                                                                               : sizeof(FFParamsT<float>));
    const std::size_t have = shared_mem_per_block(device);
    if (need > have)
      throw CliError("the forcefield needs " + std::to_string(need) + " bytes of shared memory (USE_SHARED) but " +
                     devname + " allows " + std::to_string(have) + " per block");
  }

  std::vector<Config> configs;
  std::vector<std::string> skipped;
  for (auto ppwi : o.ppwis)
    for (auto wg : o.wgsizes)
    {
      const std::size_t per_group = ppwi * wg;
      const std::string label = "ppwi " + std::to_string(ppwi) + ", wgsize " + std::to_string(wg) + ": ";
      if (deck.nposes() < per_group)
      {
        skipped.push_back(label + format_thousands((long long)deck.nposes()) + " poses is fewer than ppwi x wgsize = " +
                          format_thousands((long long)per_group));
        continue;
      }
      if (deck.nposes() % per_group != 0)
      {
        skipped.push_back(label + format_thousands((long long)deck.nposes()) +
                          " poses is not a multiple of ppwi x wgsize = " + format_thousands((long long)per_group));
        continue;
      }
      const std::size_t grid = deck.nposes() / per_group;
      if (dev.max_grid_x > 0 && grid > static_cast<std::size_t>(dev.max_grid_x))
        throw CliError("ppwi " + std::to_string(ppwi) + ", wgsize " + std::to_string(wg) + " needs " +
                       std::to_string(grid) + " blocks, more than the device maximum of " +
                       std::to_string(dev.max_grid_x));
      configs.push_back({ppwi, wg});
    }

  if (configs.empty())
  {
    std::string why;
    for (const auto& s : skipped) why += "; " + s;
    throw CliError("no runnable configuration" + why);
  }

  // ---- report header, printed before the run
  Report r;
  r.implementation = backend_name();
  r.backend = backend_name();
  r.device = dev.name;
  r.driver = dev.driver;
  r.precision = o.precision;
  r.deck = deck.name;
  r.natlig = deck.natlig();
  r.natpro = deck.natpro();
  r.ntypes = deck.ntypes();
  r.nposes = deck.nposes();
  r.max_poses = deck.maxPoses;
  r.ppwis = o.ppwis;
  r.wgsizes = o.wgsizes;
  r.iterations = o.iterations;
  r.warmup = o.warmup;
  r.reference = o.precision == Precision::Float
                  ? "one untimed fp64 pass (ppwi " + std::to_string(configs[0].ppwi) + ", wgsize " +
                      std::to_string(configs[0].wgsize) + ")"
                  : "none (float64 is the reference precision)";
  r.build = build_string();
  r.timing = "device timestamps (host wall time per launch for the overhead column)";
  r.energy_enabled = o.energy;
  r.rows = o.rows;
  r.header_notes = dev.notes;
  const std::string arch = kernel_arch_note(device);
  if (!arch.empty()) r.header_notes.push_back(arch);
  r.skipped = skipped;

  // -o: make sure the file can be written before running anything.
  if (!o.out.empty())
  {
    std::ofstream probe(o.out, std::ios::out | std::ios::trunc);
    if (!probe) throw CliError("cannot write the energies file '" + o.out + "' (-o)");
  }

  // First energy window created before the header so a profiler failure
  // stops the run before anything is printed or allocated.
  std::unique_ptr<EnergyMeter> meter(new EnergyMeter(device, o.energy));

  print_text_header(r, text_out);

  // ---- fp64 reference pass for float32 runs
  std::vector<double> golden;
  if (o.precision == Precision::Float)
    golden = reference_pass(deck, configs[0].ppwi, configs[0].wgsize, device);

  // ---- timed configurations
  for (std::size_t k = 0; k < configs.size(); k++)
  {
    if (k > 0) meter.reset(new EnergyMeter(device, o.energy));
    Sample s = run_config(o.precision, deck, configs[k].ppwi, configs[k].wgsize, device, o.warmup, o.iterations,
                          *meter);
    r.results.push_back(evaluate(deck, std::move(s), golden.empty() ? nullptr : &golden, o.precision));
  }
  meter.reset();

  r.best = 0;
  for (std::size_t k = 1; k < r.results.size(); k++)
    if (r.results[k].avg_s < r.results[r.best].avg_s) r.best = k;

  // ---- -o: energies of the first configuration, at round-trip precision
  if (!o.out.empty())
  {
    std::ofstream out(o.out, std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write the energies file '" + o.out + "'");
    out << std::setprecision(17);
    for (auto e : r.results[0].sample.energies) out << e << "\n";
    if (!out) throw std::runtime_error("failed writing the energies file '" + o.out + "'");
  }

  switch (o.output)
  {
    case OutputMode::Text:   print_text_results(r, std::cout); break;
    case OutputMode::Csv:    print_csv(r, std::cout); print_text_best(r, text_out); break;
    case OutputMode::CsvRaw: print_csv_raw(r, std::cout); print_text_best(r, text_out); break;
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
  catch (const DeckError& e)
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
