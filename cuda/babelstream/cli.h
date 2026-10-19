// Command-line parsing and --help for BabelStream.

#pragma once

#include <stdexcept>
#include <string>

#include "stream.h"

enum class OutputMode { Text, Csv, CsvRaw };

struct Options
{
  Precision precision = Precision::Double;
  long long array_size = 33554432;     // 2^25 elements
  unsigned int num_times = 1000;       // first iteration is warm-up
  int device = 0;
  int block_size = 1024;
  bool mibibytes = false;              // text report only; CSV is always GB/s
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

// Throws CliError. --help and --list are reported through Options so that
// main decides what to print.
Options parse_cli(int argc, char* argv[]);

std::string help_text(const char* program);
