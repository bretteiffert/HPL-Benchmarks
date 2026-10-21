// Command-line parsing and --help for the seven-point stencil.

#pragma once

#include <stdexcept>
#include <string>

#include "stencil.h"

enum class OutputMode { Text, Csv, CsvRaw };

struct Options
{
  Dims grid = {1024, 1024, 1024};      // -s L or --grid X,Y,Z
  Dims block = {1024, 1, 1};           // --block-size X[,Y[,Z]]
  Precision precision = Precision::Double;
  int num_times = 100;                 // timed runs after 1 untimed run
  int device = 0;
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
