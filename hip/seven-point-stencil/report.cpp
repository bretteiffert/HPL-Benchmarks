// Seven-point stencil report formatters.

#include "report.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>

// ---- statistics ------------------------------------------------------------

double RoutineResult::min_s() const
{
  return times.empty() ? 0.0 : *std::min_element(times.begin(), times.end());
}

double RoutineResult::max_s() const
{
  return times.empty() ? 0.0 : *std::max_element(times.begin(), times.end());
}

double RoutineResult::avg_s() const
{
  return times.empty() ? 0.0
                       : std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
}

// ---- formatting helpers ----------------------------------------------------

std::string format_thousands(long long v)
{
  const bool neg = v < 0;
  std::string digits = std::to_string(neg ? -v : v);
  std::string out;
  int n = 0;
  for (auto it = digits.rbegin(); it != digits.rend(); ++it)
  {
    if (n > 0 && n % 3 == 0) out.push_back(',');
    out.push_back(*it);
    n++;
  }
  if (neg) out.push_back('-');
  std::reverse(out.begin(), out.end());
  return out;
}

std::string format_fixed(double v, int decimals)
{
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
  return buf;
}

std::string format_sci(double v, int decimals)
{
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*e", decimals, v);
  std::string s(buf);
  const std::size_t e = s.find('e');
  if (e == std::string::npos) return s;
  const int exponent = std::atoi(s.c_str() + e + 1);
  return s.substr(0, e) + "e" + std::to_string(exponent);
}

std::string csv_field(const std::string& s)
{
  if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
  std::string out = "\"";
  for (char ch : s)
  {
    if (ch == '"') out += "\"\"";
    else out.push_back(ch);
  }
  out += "\"";
  return out;
}

namespace {

const char* TIMES = "\xC3\x97";   // "×"

std::string pad_left(const std::string& s, std::size_t width)
{
  return s.size() >= width ? s : std::string(width - s.size(), ' ') + s;
}

std::string pad_right(const std::string& s, std::size_t width)
{
  return s.size() >= width ? s : s + std::string(width - s.size(), ' ');
}

void key(std::ostream& out, const std::string& k, const std::string& v)
{
  out << "  " << pad_right(k, 13) << v << "\n";
}

std::string dims(const Dims& d)
{
  return format_thousands((long long)d.x) + " " + TIMES + " " + format_thousands((long long)d.y) + " " + TIMES +
         " " + format_thousands((long long)d.z);
}

std::string mb(double bytes) { return format_fixed(bytes * 1e-6, 1) + " MB"; }

std::string csv_prefix(const Report& r)
{
  return csv_field(r.implementation) + "," + csv_field(r.backend) + "," + csv_field(r.device) + "," +
         precision_name(r.precision) + "," + std::to_string(r.grid.x) + "," + std::to_string(r.grid.y) + "," +
         std::to_string(r.grid.z) + "," + std::to_string(r.block.x) + "," + std::to_string(r.block.y) + "," +
         std::to_string(r.block.z) + ",";
}

} // namespace

// ---- text ------------------------------------------------------------------

void print_text_header(const Report& r, std::ostream& out)
{
  const std::size_t elem = precision_bytes(r.precision);
  out << r.benchmark << " (" << r.implementation << ")\n";
  key(out, "Backend", r.backend);
  key(out, "Device", r.device);
  key(out, "Driver", r.driver);
  key(out, "Precision", precision_name(r.precision));
  key(out, "Grid", dims(r.grid) + " (" + format_thousands((long long)r.grid.count()) + " points, " +
                   mb(double(r.grid.count()) * elem) + " per array)");
  key(out, "Block size", dims(r.block) + " (" + format_thousands((long long)r.block.count()) + " threads)");
  key(out, "Launch grid", dims(r.launch) + " blocks");
  key(out, "Traffic", mb(r.fetch_bytes) + " read + " + mb(r.write_bytes) + " written per run (theoretical)");
  key(out, "Iterations", format_thousands(r.num_times) + " (after 1 untimed run)");
  key(out, "Timing", r.timing);
  key(out, "Energy", r.energy.enabled ? "libkernelenergy (board power)" : "disabled (--no-energy)");
  for (const auto& n : r.header_notes) out << "  note: " << n << "\n";
  out << std::flush;
}

void print_text_results(const Report& r, std::ostream& out)
{
  const RoutineResult& rt = r.laplacian;
  out << "\n"
      << pad_right("Routine", 12) << pad_left("Bandwidth (GB/s)", 18)
      << pad_left("Min (s)", 14) << pad_left("Max (s)", 14) << pad_left("Avg (s)", 14) << "\n";
  out << pad_right(rt.name, 12) << pad_left(format_fixed(1e-9 * rt.bytes / rt.min_s(), 3), 18)
      << pad_left(format_fixed(rt.min_s(), 8), 14) << pad_left(format_fixed(rt.max_s(), 8), 14)
      << pad_left(format_fixed(rt.avg_s(), 8), 14) << "\n";
  out << "  note: bandwidth uses the minimum time; from the average time it is "
      << format_fixed(1e-9 * rt.bytes / rt.avg_s(), 3) << " GB/s\n";
  out << std::flush;
}

void print_text_energy(const Report& r, std::ostream& out)
{
  out << "\n";
  if (!r.energy.enabled)
  {
    out << "Energy  disabled (--no-energy)\n" << std::flush;
    return;
  }
  const EnergyResult& e = r.energy;
  out << pad_right("Energy", 8) << pad_left("J/iter", 14) << pad_left("Total (Wh)", 14)
      << pad_left("Avg power (W)", 16) << pad_left("Time (s)", 12)
      << pad_left("Iterations", 13) << pad_left("Samples", 10) << "\n";
  out << pad_right("board", 8)
      << pad_left(format_fixed(e.joules_per_iter, 6), 14)
      << pad_left(format_fixed(e.total_wh, 6), 14)
      << pad_left(format_fixed(e.avg_power_w, 2), 16)
      << pad_left(format_fixed(e.seconds, 6), 12)
      << pad_left(format_thousands(static_cast<long long>(e.iterations)), 13)
      << pad_left(format_thousands(e.samples), 10) << "\n";
  if (e.samples < 2)
    out << "  note: fewer than two power samples; the energy figures may be unreliable\n";
  out << std::flush;
}

void print_text_correctness(const Report& r, std::ostream& out)
{
  const Correctness& c = r.correctness;
  const std::size_t lw = 24, vw = 18;
  out << "\n"
      << "Correctness (" << precision_name(r.precision) << ", interior points vs the exact Laplacian 3)\n";
  out << pad_right("Quantity", lw) << pad_left("Value", vw) << "\n";
  out << pad_right("Points checked", lw) << pad_left(format_thousands((long long)c.num_checked), vw) << "\n";
  out << pad_right("Max abs error", lw) << pad_left(format_sci(c.max_abs_err, 6), vw) << "\n";
  out << pad_right("Mean abs error", lw) << pad_left(format_sci(c.mean_abs_err, 6), vw) << "\n";
  out << pad_right("Relative L2 error", lw) << pad_left(format_sci(c.l2_rel_err, 6), vw) << "\n";
  out << "  note: checked after the untimed first run; each timed run recomputes the same values\n";
  if (r.precision != Precision::Double)
    out << "  note: the stencil cancels terms of size about 6/h^2 to produce 3, so the error grows with the grid "
           "in " << precision_name(r.precision) << "\n";
  out << std::flush;
}

// ---- CSV -------------------------------------------------------------------

void print_csv(const Report& r, std::ostream& out, bool with_header)
{
  if (with_header)
    out << "implementation,backend,device,precision,nx,ny,nz,block_x,block_y,block_z,num_times,routine,"
           "bw_gbs,min_s,max_s,avg_s,energy_j_per_iter,avg_power_w\n";
  const RoutineResult& rt = r.laplacian;
  out << csv_prefix(r) << r.num_times << "," << csv_field(rt.name) << ","
      << format_fixed(1e-9 * rt.bytes / rt.min_s(), 4) << ","
      << format_fixed(rt.min_s(), 9) << "," << format_fixed(rt.max_s(), 9) << "," << format_fixed(rt.avg_s(), 9) << ",";
  if (r.energy.enabled)
    out << format_fixed(r.energy.joules_per_iter, 6) << "," << format_fixed(r.energy.avg_power_w, 3);
  else
    out << ",";
  out << "\n" << std::flush;
}

void print_csv_raw(const Report& r, std::ostream& out, bool with_header)
{
  if (with_header)
    out << "implementation,backend,device,precision,nx,ny,nz,block_x,block_y,block_z,routine,iteration,time_s,bw_gbs\n";
  const RoutineResult& rt = r.laplacian;
  const std::string prefix = csv_prefix(r) + csv_field(rt.name) + ",";
  for (std::size_t j = 0; j < rt.times.size(); j++)
    out << prefix << (j + 1) << "," << format_fixed(rt.times[j], 9) << ","
        << format_fixed(1e-9 * rt.bytes / rt.times[j], 4) << "\n";
  out << std::flush;
}
