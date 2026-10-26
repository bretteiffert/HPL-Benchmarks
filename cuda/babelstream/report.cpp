// BabelStream report formatters.

#include "report.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
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

const char* EM_DASH = "\xE2\x80\x94";   // "—", one display column

std::string pad_left(const std::string& s, std::size_t width)
{
  return s.size() >= width ? s : std::string(width - s.size(), ' ') + s;
}

std::string pad_right(const std::string& s, std::size_t width)
{
  return s.size() >= width ? s : s + std::string(width - s.size(), ' ');
}

// Right-align a single em dash in a column of `width` display columns.
std::string dash_cell(std::size_t width)
{
  return std::string(width - 1, ' ') + EM_DASH;
}

double bandwidth_scale(bool mibibytes)
{
  return mibibytes ? std::pow(2.0, -20.0) : 1.0E-9;
}

void key(std::ostream& out, const std::string& k, const std::string& v)
{
  out << "  " << pad_right(k, 13) << v << "\n";
}

std::string size_string(double bytes, bool mibibytes)
{
  const double scaled = mibibytes ? bytes * std::pow(2.0, -20.0) : bytes * 1.0E-6;
  return format_fixed(scaled, 1) + (mibibytes ? " MiB" : " MB");
}

std::string csv_prefix(const Report& r)
{
  return csv_field(r.implementation) + "," + csv_field(r.backend) + "," +
         csv_field(r.device) + "," + precision_name(r.precision) + ",";
}

} // namespace

// ---- text ------------------------------------------------------------------

void print_text_header(const Report& r, std::ostream& out)
{
  out << r.benchmark << " (" << r.implementation << ")\n";
  key(out, "Backend", r.backend);
  key(out, "Device", r.device);
  key(out, "Driver", r.driver);
  key(out, "Precision", precision_name(r.precision));

  const double array_bytes = static_cast<double>(r.array_size) * precision_bytes(r.precision);
  key(out, "Array size", format_thousands(r.array_size) + " elements (" +
                         size_string(array_bytes, r.mibibytes) + " per array, " +
                         size_string(3.0 * array_bytes, r.mibibytes) + " total)");
  key(out, "Block size", std::to_string(r.block_size));

  if (r.compute_units > 0)
    key(out, "Dot groups", std::to_string(r.dot_groups) + " (4 \xC3\x97 " +
                           std::to_string(r.compute_units) + " " + r.compute_unit_label + ")");
  else
    key(out, "Dot groups", std::to_string(r.dot_groups) + " (fallback)");

  key(out, "Iterations", format_thousands(r.num_times) + " (first excluded from statistics)");
  key(out, "FMAD", std::to_string(r.fmad));
  key(out, "Timing", r.timing);
  key(out, "Energy", r.energy.enabled ? "libkernelenergy (board power)" : "disabled (--no-energy)");

  for (const auto& n : r.header_notes)
    out << "  note: " << n << "\n";
  out << std::flush;
}

void print_text_results(const Report& r, std::ostream& out)
{
  const std::string bw_header = r.mibibytes ? "Bandwidth (MiB/s)" : "Bandwidth (GB/s)";
  const std::size_t bw_w = std::max<std::size_t>(16, bw_header.size());
  const std::size_t t_w = 14;
  const double scale = bandwidth_scale(r.mibibytes);

  out << "\n"
      << pad_right("Function", 12) << pad_left(bw_header, bw_w)
      << pad_left("Min (s)", t_w) << pad_left("Max (s)", t_w) << pad_left("Avg (s)", t_w) << "\n";

  for (const auto& rt : r.routines)
  {
    const double mn = rt.min_s();
    out << pad_right(rt.name, 12)
        << pad_left(format_fixed(scale * rt.bytes / mn, 3), bw_w)
        << pad_left(format_fixed(mn, 8), t_w);
    if (rt.one_shot)
      out << dash_cell(t_w) << dash_cell(t_w);
    else
      out << pad_left(format_fixed(rt.max_s(), 8), t_w) << pad_left(format_fixed(rt.avg_s(), 8), t_w);
    out << "\n";
  }

  out << "  note: bandwidth uses the minimum time; Dot time includes copying the "
      << format_thousands(r.dot_groups) << " partial sums to the host\n";
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
  const char* prec = precision_name(r.precision);

  out << "\n"
      << "Correctness (" << prec << ", " << format_thousands(r.num_times) << " iterations)\n"
      << pad_right("Array", 6) << pad_left("Expected", 20)
      << pad_left("Checksum rel err", 20) << pad_left("Max elem rel err", 20) << "\n";

  for (const auto& row : c.rows)
  {
    out << pad_right(row.name, 6)
        << pad_left(format_sci(row.expected, 6), 20)
        << pad_left(format_sci(row.checksum_rel_err, 3), 20);
    if (row.has_max_elem)
      out << pad_left(format_sci(row.max_elem_rel_err, 3), 20);
    else
      out << dash_cell(20);
    out << "\n";
  }

  out << "  note: actual dot = " << format_sci(c.actual_dot, 9) << "\n";
  if (c.gold_underflows)
    out << "  note: gold values fall below the " << prec << " normal range ("
        << format_sci(c.min_normal, 3) << ") at this iteration count (smallest "
        << format_sci(c.min_gold, 3) << "), so the array relative errors are not meaningful\n";
  if (c.dot_product_underflows)
    out << "  note: expected element product " << format_sci(c.product_magnitude, 3)
        << " underflows " << prec << " at this iteration count, so the dot relative error is not meaningful\n";
  if (c.dot_overflows)
    out << "  note: expected dot " << format_sci(std::fabs(c.rows.back().expected), 3)
        << " exceeds the " << prec << " maximum (" << format_thousands(static_cast<long long>(c.max_finite))
        << "), so the device dot overflows and its relative error is not meaningful\n";
  out << std::flush;
}

// ---- CSV -------------------------------------------------------------------

void print_csv(const Report& r, std::ostream& out, bool with_header)
{
  if (with_header)
    out << "implementation,backend,device,precision,array_size,block_size,dot_groups,num_times,"
           "routine,bw_gbs,min_s,max_s,avg_s,energy_j_per_iter,avg_power_w\n";

  const std::string prefix = csv_prefix(r);
  for (const auto& rt : r.routines)
  {
    const double mn = rt.min_s();
    out << prefix << r.array_size << "," << r.block_size << "," << r.dot_groups << ","
        << r.num_times << "," << csv_field(rt.name) << ","
        << format_fixed(1.0E-9 * rt.bytes / mn, 4) << ","
        << format_fixed(mn, 9) << ",";
    if (rt.one_shot)
      out << ",,";
    else
      out << format_fixed(rt.max_s(), 9) << "," << format_fixed(rt.avg_s(), 9) << ",";

    // Energy covers the timed iterations of the loop, not the one-shot Init.
    if (r.energy.enabled && !rt.one_shot)
      out << format_fixed(r.energy.joules_per_iter, 6) << "," << format_fixed(r.energy.avg_power_w, 3);
    else
      out << ",";
    out << "\n";
  }
  out << std::flush;
}

void print_csv_raw(const Report& r, std::ostream& out, bool with_header)
{
  if (with_header)
    out << "implementation,backend,device,precision,array_size,block_size,routine,iteration,time_s,bw_gbs\n";

  const std::string prefix = csv_prefix(r);
  for (const auto& rt : r.routines)
  {
    for (std::size_t j = 0; j < rt.times.size(); j++)
    {
      out << prefix << r.array_size << "," << r.block_size << "," << csv_field(rt.name) << ","
          << (j + 1) << "," << format_fixed(rt.times[j], 9) << ","
          << format_fixed(1.0E-9 * rt.bytes / rt.times[j], 4) << "\n";
    }
  }
  out << std::flush;
}
