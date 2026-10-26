// Command-line parsing and --help for miniBUDE. Benchmark flags keep the
// upstream names (-i, -n, -p, -w, --deck, -o, -r, -d).

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "bude.h"
#include "deck.h"

enum class OutputMode { Text, Csv, CsvRaw };

struct Options
{
  std::string deck = DATA_DIR;
  Precision precision = Precision::Float;
  std::size_t iterations = 8;          // timed launches per configuration
  std::size_t warmup = 2;              // untimed launches before them
  std::size_t poses = 0;               // 0 = all poses in the deck
  std::vector<std::size_t> ppwis = {1};
  std::vector<std::size_t> wgsizes = {256};
  std::string device = "0";            // index, or a substring of the device name
  std::string out;                     // -o: energies of the first configuration
  std::size_t rows = 8;                // energies shown in the text report
  OutputMode output = OutputMode::Text;
  bool energy = true;
  bool list = false;
  bool help = false;
};

// A user error. main prints "error: <what> (try '--help')" and exits 1.
struct CliError : std::runtime_error
{
  explicit CliError(const std::string& what) : std::runtime_error(what) {}
};

Options parse_cli(int argc, char* argv[]);

std::string help_text(const char* program);
