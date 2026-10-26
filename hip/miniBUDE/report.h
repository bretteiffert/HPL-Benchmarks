// miniBUDE report: one struct filled by the run, three formatters (text,
// --csv, --csv-raw) that read only from it.
//
// Stream rules (applied by main): text mode writes everything to stdout; the
// CSV modes write only CSV to stdout and the text header, energy and
// correctness sections to stderr.

#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "bude.h"
#include "metrics.h"

struct ConfigResult
{
  Sample sample;                   // timings exclude warm-up launches
  double min_s = 0, max_s = 0, avg_s = 0, stddev_s = 0;   // population stddev, as upstream
  double gflops = 0, gfinsts = 0, ginteractions = 0;      // per second, 1e9 units
  double launch_overhead_s = 0;    // mean (wall - device) per timed launch
  bool has_ref = false;            // float32 runs: compared with the fp64 pass
  Metrics vs_ref;
  Metrics vs_deck;
};

struct Report
{
  std::string benchmark = "miniBUDE";
  std::string implementation;
  std::string backend;
  std::string device;
  std::string driver;
  Precision precision = Precision::Float;

  std::string deck;                // deck name, e.g. "bm1"
  std::size_t natlig = 0, natpro = 0, ntypes = 0, nposes = 0, max_poses = 0;
  std::vector<std::size_t> ppwis, wgsizes;
  std::size_t iterations = 0, warmup = 0;
  std::string reference;           // how the reference was obtained
  std::string build;               // build-time options
  std::string timing;
  bool energy_enabled = false;
  std::size_t rows = 8;

  std::vector<std::string> header_notes;
  std::vector<std::string> skipped;   // one line per skipped configuration

  std::vector<ConfigResult> results;
  std::size_t best = 0;               // index of the lowest avg_s
};

void print_text_header(const Report& r, std::ostream& out);
void print_text_results(const Report& r, std::ostream& out);   // table + best note
void print_text_best(const Report& r, std::ostream& out);      // best note alone (CSV modes)
void print_text_energy(const Report& r, std::ostream& out);
void print_text_correctness(const Report& r, std::ostream& out);

void print_csv(const Report& r, std::ostream& out, bool with_header = true);
void print_csv_raw(const Report& r, std::ostream& out, bool with_header = true);

std::string format_thousands(long long v);
std::string format_fixed(double v, int decimals);
std::string format_sci(double v, int decimals);   // 1.230000e-1, no exponent padding
std::string csv_field(const std::string& s);
