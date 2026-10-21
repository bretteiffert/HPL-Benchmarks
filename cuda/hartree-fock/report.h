// Hartree-Fock report: one struct filled by the run, three formatters
// (text, --csv, --csv-raw) that read only from it.
//
// Stream rules (applied by main): text mode writes everything to stdout; the
// CSV modes write only CSV to stdout and the text header, energy and
// correctness sections to stderr.

#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "energy.h"
#include "hf.h"

struct RoutineResult
{
  std::string name;              // "Fock build"
  std::vector<double> times;     // seconds, timed builds only (untimed first build excluded)

  double min_s() const;
  double max_s() const;
  double avg_s() const;
};

struct Report
{
  std::string benchmark = "Hartree-Fock";
  std::string implementation;    // port name, "CUDA"
  std::string backend;
  std::string device;
  std::string driver;
  Precision precision = Precision::Double;
  SqrtMode sqrt_mode = SqrtMode::Native;

  std::string input;             // file name without directories
  uint32_t natoms = 0;
  uint32_t ngauss = 0;
  uint64_t pairs = 0;            // unique atom pairs
  uint64_t quartets = 0;         // unique pair quartets = kernel threads doing work

  uint32_t block_size = 0;
  uint64_t grid = 0;
  int num_times = 0;             // timed builds
  std::string timing;

  std::vector<std::string> header_notes;

  RoutineResult fock;
  double checksum = 0.0;         // 2e- energy of the first build, run-precision sum
  EnergyResult energy;

  bool checked = false;
  std::string check_skipped;     // reason, when not checked
  FockCheck check;
};

void print_text_header(const Report& r, std::ostream& out);
void print_text_results(const Report& r, std::ostream& out);
void print_text_energy(const Report& r, std::ostream& out);
void print_text_correctness(const Report& r, std::ostream& out);

void print_csv(const Report& r, std::ostream& out, bool with_header = true);
void print_csv_raw(const Report& r, std::ostream& out, bool with_header = true);

std::string format_thousands(long long v);
std::string format_fixed(double v, int decimals);
std::string format_sci(double v, int decimals);   // 1.230000e-1, no exponent padding
std::string csv_field(const std::string& s);
