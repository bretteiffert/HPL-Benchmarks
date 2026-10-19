// Seven-point stencil report: one struct filled by the run, three
// formatters (text, --csv, --csv-raw) that read only from it.
//
// Stream rules (applied by main): text mode writes everything to stdout; the
// CSV modes write only CSV to stdout and the text header, energy and
// correctness sections to stderr.

#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "energy.h"
#include "stencil.h"

struct RoutineResult
{
  std::string name;              // "Laplacian"
  double bytes = 0.0;            // theoretical traffic per run
  std::vector<double> times;     // seconds, timed runs only

  double min_s() const;
  double max_s() const;
  double avg_s() const;
};

struct Report
{
  std::string benchmark = "Seven-point stencil";
  std::string implementation;    // port name, "HIP"
  std::string backend;
  std::string device;
  std::string driver;
  Precision precision = Precision::Double;

  Dims grid, block, launch;
  double fetch_bytes = 0.0, write_bytes = 0.0;
  int num_times = 0;
  std::string timing;

  std::vector<std::string> header_notes;

  RoutineResult laplacian;
  EnergyResult energy;
  Correctness correctness;
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
