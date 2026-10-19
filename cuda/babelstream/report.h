// BabelStream report: one struct filled by the run, three formatters
// (text, --csv, --csv-raw) that read only from it.
//
// Stream rules (applied by main): text mode writes everything to stdout; the
// CSV modes write only CSV to stdout and the text header, energy and
// correctness sections to stderr.

#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "energy.h"
#include "stream.h"

struct RoutineResult
{
  std::string name;              // Init, Copy, Mul, Add, Triad, Dot
  double bytes = 0.0;            // bytes moved per call
  bool one_shot = false;         // Init: a single timed call, no max/avg
  std::vector<double> times;     // seconds, timed iterations only (warm-up excluded)

  double min_s() const;
  double max_s() const;
  double avg_s() const;
};

struct Report
{
  std::string benchmark = "BabelStream";
  std::string implementation;    // port name, "CUDA"
  std::string backend;           // "CUDA"
  std::string device;
  std::string driver;
  Precision precision = Precision::Double;

  long long array_size = 0;
  int block_size = 0;
  int dot_groups = 0;
  int compute_units = 0;         // 0 when the count was unavailable
  std::string compute_unit_label;  // "SMs" or "CUs"

  unsigned int num_times = 0;    // including the warm-up iteration
  int fmad = 1;
  std::string timing;            // e.g. "device timestamps"
  bool mibibytes = false;

  std::vector<std::string> header_notes;

  std::vector<RoutineResult> routines;
  EnergyResult energy;
  Correctness correctness;
};

// Text sections. print_text_header prints before the run starts.
void print_text_header(const Report& r, std::ostream& out);
void print_text_results(const Report& r, std::ostream& out);
void print_text_energy(const Report& r, std::ostream& out);
void print_text_correctness(const Report& r, std::ostream& out);

// CSV formatters. The header line is printed when `with_header` is true.
void print_csv(const Report& r, std::ostream& out, bool with_header = true);
void print_csv_raw(const Report& r, std::ostream& out, bool with_header = true);

// Helpers shared by the formatters (exposed for tests).
std::string format_thousands(long long v);
std::string format_fixed(double v, int decimals);
std::string format_sci(double v, int decimals);   // 1.230000e-1, no exponent padding
std::string csv_field(const std::string& s);      // quotes when needed
