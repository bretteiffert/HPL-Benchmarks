// Hartree-Fock report formatters.

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

std::string csv_prefix(const Report& r)
{
  return csv_field(r.implementation) + "," + csv_field(r.backend) + "," +
         csv_field(r.device) + "," + precision_name(r.precision) + ",";
}

// CSV numbers: full precision for energies, plain C exponent form.
std::string csv_sci(double v)
{
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.16e", v);
  return buf;
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
  key(out, "Sqrt mode", sqrt_mode_name(r.sqrt_mode));
  key(out, "Input", r.input + " (" + format_thousands(r.natoms) + " atoms, " +
                    format_thousands(r.ngauss) + " Gaussians)");
  key(out, "Quartets", format_thousands(static_cast<long long>(r.quartets)) + " (from " +
                       format_thousands(static_cast<long long>(r.pairs)) + " atom pairs)");
  key(out, "Block size", std::to_string(r.block_size));
  key(out, "Grid", format_thousands(static_cast<long long>(r.grid)) + " blocks");
  key(out, "Iterations", format_thousands(r.num_times) + " (after 1 untimed build)");
  key(out, "Timing", r.timing);
  key(out, "Energy", r.energy.enabled ? "libkernelenergy (board power)" : "disabled (--no-energy)");

  for (const auto& n : r.header_notes)
    out << "  note: " << n << "\n";
  out << std::flush;
}

void print_text_results(const Report& r, std::ostream& out)
{
  const std::size_t t_w = 14;
  out << "\n"
      << pad_right("Routine", 12) << pad_left("Min (s)", t_w) << pad_left("Max (s)", t_w)
      << pad_left("Avg (s)", t_w) << "\n";
  out << pad_right(r.fock.name, 12)
      << pad_left(format_fixed(r.fock.min_s(), 8), t_w)
      << pad_left(format_fixed(r.fock.max_s(), 8), t_w)
      << pad_left(format_fixed(r.fock.avg_s(), 8), t_w) << "\n";
  out << "  note: each timed build adds into the same Fock matrix; the checksum and the check use the untimed first build\n";
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
  const std::string prec = precision_name(r.precision);
  const std::size_t label_w = 42, value_w = 24;
  auto row = [&](const std::string& label, const std::string& value) {
    out << pad_right(label, label_w) << pad_left(value, value_w) << "\n";
  };

  out << "\n"
      << "Correctness (" << prec << ", sqrt " << sqrt_mode_name(r.sqrt_mode)
      << ", untimed first build vs fp64 CPU reference)\n";
  out << pad_right("Quantity", label_w) << pad_left("Value", value_w) << "\n";

  row("2e- energy (" + prec + " Fock, " + prec + " sum)", format_sci(r.checksum, 16));

  if (r.checked)
  {
    const FockCheck& c = r.check;
    row("2e- energy (" + prec + " Fock, fp64 sum)", format_sci(c.e_gpu, 16));
    row("2e- energy (fp64 CPU reference)", format_sci(c.e_ref, 16));
    row("rel err, Fock build only", format_sci(c.rel_loss, 6));
    row("rel err, incl. " + prec + " sum", format_sci(c.rel_inpipe, 6));
    row("max |F - F_ref|", format_sci(c.max_abs, 6));
    row("max |F - F_ref| / max |F_ref|", format_sci(c.max_rel, 6));
    row("rms |F - F_ref|", format_sci(c.rms_abs, 6));
    row("max |F(i,j) - F(j,i)|", format_sci(c.asym, 6));
    row("2e- energy, symmetrized F (diagnostic)", format_sci(c.e_symmetrized, 16));
  }
  else
  {
    out << "  note: reference check skipped: " << r.check_skipped << "\n";
  }
  out << "  note: the 2e- energy is an unsymmetrized benchmark checksum, not a physical two-electron energy\n";
  out << std::flush;
}

// ---- CSV -------------------------------------------------------------------

void print_csv(const Report& r, std::ostream& out, bool with_header)
{
  if (with_header)
    out << "implementation,backend,device,precision,sqrt_mode,input,natoms,ngauss,block_size,quartets,groups,num_times,"
           "routine,min_s,max_s,avg_s,e2_energy,rel_err_ref,energy_j_per_iter,avg_power_w\n";

  out << csv_prefix(r) << sqrt_mode_name(r.sqrt_mode) << "," << csv_field(r.input) << ","
      << r.natoms << "," << r.ngauss << "," << r.block_size << "," << r.quartets << ","
      << r.grid << "," << r.num_times << ","
      << csv_field(r.fock.name) << ","
      << format_fixed(r.fock.min_s(), 9) << "," << format_fixed(r.fock.max_s(), 9) << ","
      << format_fixed(r.fock.avg_s(), 9) << ","
      << csv_sci(r.checksum) << ",";
  if (r.checked) out << csv_sci(r.check.rel_inpipe);
  out << ",";
  if (r.energy.enabled)
    out << format_fixed(r.energy.joules_per_iter, 6) << "," << format_fixed(r.energy.avg_power_w, 3);
  else
    out << ",";
  out << "\n" << std::flush;
}

void print_csv_raw(const Report& r, std::ostream& out, bool with_header)
{
  if (with_header)
    out << "implementation,backend,device,precision,sqrt_mode,input,natoms,ngauss,block_size,quartets,groups,"
           "routine,iteration,time_s\n";

  const std::string prefix = csv_prefix(r) + sqrt_mode_name(r.sqrt_mode) + "," + csv_field(r.input) + "," +
                             std::to_string(r.natoms) + "," + std::to_string(r.ngauss) + "," +
                              std::to_string(r.block_size) + "," + std::to_string(r.quartets) + "," +
                              std::to_string(r.grid) + "," + csv_field(r.fock.name) + ",";
  for (std::size_t j = 0; j < r.fock.times.size(); j++)
    out << prefix << (j + 1) << "," << format_fixed(r.fock.times[j], 9) << "\n";
  out << std::flush;
}
