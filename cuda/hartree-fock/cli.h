// Command-line parsing and --help for Hartree-Fock.

#pragma once

#include <stdexcept>
#include <string>

#include "hf.h"

enum class OutputMode { Text, Csv, CsvRaw };

struct Options
{
  std::string input;                    // positional input file
  Precision precision = Precision::Double;
  SqrtMode sqrt_mode = SqrtMode::Native;
  int num_times = 10;                   // timed builds, after 1 untimed build
  int device = 0;
  int block_size = 256;
  OutputMode output = OutputMode::Text;
  bool energy = true;
  bool check = true;                    // --no-check turns it off
  bool force_check = false;             // --check: ignore --max-check-atoms
  int max_check_atoms = 128;
  bool list = false;
  bool help = false;
};

// A user error. main prints "error: <what> (try '--help')" and exits 1.
struct CliError : std::runtime_error
{
  explicit CliError(const std::string& what) : std::runtime_error(what) {}
};

// Throws CliError. --help and --list are reported through Options.
Options parse_cli(int argc, char* argv[]);

std::string help_text(const char* program);
