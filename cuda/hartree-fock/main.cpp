// Hartree-Fock driver: parses the command line, reads the input file,
// validates the configuration against the device limits before allocating
// anything, runs one untimed Fock build (checksum and correctness source)
// and then num_times event-timed builds inside one energy measurement, runs
// the fp64 CPU reference check outside the measured window, and hands one
// report struct to the formatters.

#include <iostream>
#include <memory>
#include <string>

#include "backend.h"
#include "cli.h"
#include "deck.h"
#include "energy.h"
#include "hf.h"
#include "report.h"

namespace {

// Throws CliError naming the limit that a setting breaks.
void validate(const Options& o, const DeviceInfo& dev, const Deck& deck)
{
  const std::string device = "device " + std::to_string(dev.index) + " (" + dev.name + ")";

  if (o.precision == Precision::Double && !fp64_atomic_add_supported(dev.index))
    throw CliError("float64 needs atomicAdd on double, which " + device +
                   " does not support; use --float");

  if (dev.max_threads_per_block > 0 && o.block_size > dev.max_threads_per_block)
    throw CliError("block size " + std::to_string(o.block_size) + " exceeds the " +
                   std::to_string(dev.max_threads_per_block) + " threads per block allowed on " + device);

  // quartet_count uses 64-bit (nn * nn + nn) / 2: safe up to 65,535 atoms.
  if (deck.natoms > 65535u)
    throw CliError("input '" + deck.path + "' has " + std::to_string(deck.natoms) +
                   " atoms; the quartet index fits 64 bits only up to 65,535 atoms");

  const uint64_t grid = grid_blocks(deck.natoms, static_cast<uint32_t>(o.block_size));
  const uint64_t max_grid = dev.max_grid_x > 0 ? static_cast<uint64_t>(dev.max_grid_x) : 2147483647ULL;
  if (grid > max_grid)
    throw CliError("grid of " + std::to_string(grid) + " blocks for " + std::to_string(deck.natoms) +
                   " atoms exceeds the device maximum of " + std::to_string(max_grid) +
                   "; use a larger --block-size");
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
  const Deck deck = read_deck(o.input);

  validate(o, dev, deck);

  Report r;
  r.implementation = backend_name();
  r.backend = backend_name();
  r.device = dev.name;
  r.driver = dev.driver;
  r.precision = o.precision;
  r.sqrt_mode = o.sqrt_mode;
  r.input = deck.name;
  r.natoms = deck.natoms;
  r.ngauss = deck.ngauss;
  r.pairs = pair_count(deck.natoms);
  r.quartets = quartet_count(deck.natoms);
  r.block_size = static_cast<uint32_t>(o.block_size);
  r.grid = grid_blocks(deck.natoms, r.block_size);
  r.num_times = o.num_times;
  r.timing = "device timestamps";
  r.header_notes = dev.notes;
  r.fock.name = "Fock build";

  EnergyMeter meter(o.device, o.energy);
  r.energy.enabled = meter.enabled();

  print_text_header(r, text_out);

  std::unique_ptr<HartreeFock> hf = make_hartree_fock(o.precision, o.sqrt_mode, deck, r.block_size);

  // Untimed first build: fills the Fock matrix used by the checksum and check.
  hf->first_build();
  r.checksum = hf->checksum();

  meter.start();
  for (int it = 0; it < o.num_times; it++)
    r.fock.times.push_back(hf->timed_build());
  meter.stop();
  r.energy = meter.result(static_cast<unsigned long>(o.num_times));

  // fp64 CPU reference, outside the measured window.
  if (!o.check)
  {
    r.check_skipped = "--no-check";
  }
  else if (!o.force_check && static_cast<long long>(deck.natoms) > o.max_check_atoms)
  {
    r.check_skipped = std::to_string(deck.natoms) + " atoms > --max-check-atoms " +
                      std::to_string(o.max_check_atoms) + "; use --check to force it";
  }
  else
  {
    r.check = hf->check();
    r.checked = true;
  }

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
