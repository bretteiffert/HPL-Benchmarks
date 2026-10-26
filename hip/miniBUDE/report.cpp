// miniBUDE report formatters.

#include "report.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

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

std::string join(const std::vector<std::size_t>& xs, const char* sep)
{
  std::string s;
  for (std::size_t i = 0; i < xs.size(); i++) s += (i ? sep : "") + std::to_string(xs[i]);
  return s;
}

std::string config_label(const ConfigResult& c)
{
  return "ppwi " + std::to_string(c.sample.ppwi) + ", wg " + std::to_string(c.sample.wgsize);
}

std::string csv_prefix(const Report& r)
{
  return csv_field(r.implementation) + "," + csv_field(r.backend) + "," + csv_field(r.device) + "," +
         precision_name(r.precision) + "," + csv_field(r.deck) + "," + std::to_string(r.natlig) + "," +
         std::to_string(r.natpro) + "," + std::to_string(r.nposes) + ",";
}

std::string csv_sci(double v)
{
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6e", v);
  return buf;
}

// Prints rows of a per-configuration table, or one "all" row when every
// configuration has the same cells.
void print_collapsible(std::ostream& out, const Report& r,
                       const std::vector<std::vector<std::string>>& cells,
                       const std::vector<std::size_t>& widths)
{
  bool all_same = cells.size() > 1;
  for (std::size_t i = 1; i < cells.size() && all_same; i++)
    if (cells[i] != cells[0]) all_same = false;

  auto print_row = [&](const std::string& ppwi, const std::string& wg, const std::vector<std::string>& row) {
    out << pad_left(ppwi, 6) << pad_left(wg, 9);
    for (std::size_t j = 0; j < row.size(); j++) out << pad_left(row[j], widths[j]);
    out << "\n";
  };

  if (all_same)
  {
    print_row("all", "all", cells[0]);
    return;
  }
  for (std::size_t i = 0; i < cells.size(); i++)
    print_row(std::to_string(r.results[i].sample.ppwi), std::to_string(r.results[i].sample.wgsize), cells[i]);
}

std::string histogram_string(const Metrics& m)
{
  std::string s;
  for (std::size_t i = 0; i < ULP_NBUCKETS; ++i)
    if (m.ulpHistogram[i] > 0)
      s += (s.empty() ? "" : "  ") + ulpBucketLabel(i) + ": " + format_thousands((long long)m.ulpHistogram[i]);
  return s.empty() ? "(nothing compared)" : s;
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
  key(out, "Deck", r.deck + " (" + format_thousands((long long)r.natlig) + " ligand atoms, " +
                   format_thousands((long long)r.natpro) + " protein atoms, " +
                   format_thousands((long long)r.ntypes) + " forcefield types)");
  key(out, "Poses", format_thousands((long long)r.nposes) + " (of " + format_thousands((long long)r.max_poses) +
                    " in the deck)");
  key(out, "PPWI", join(r.ppwis, ", "));
  key(out, "WG size", join(r.wgsizes, ", "));
  key(out, "Iterations", format_thousands((long long)r.iterations) + " (after " +
                         format_thousands((long long)r.warmup) + " warm-up)");
  key(out, "Reference", r.reference);
  key(out, "Build", r.build);
  key(out, "Timing", r.timing);
  key(out, "Energy", r.energy_enabled ? "libkernelenergy (board power), one window per configuration"
                                      : "disabled (--no-energy)");
  for (const auto& n : r.header_notes) out << "  note: " << n << "\n";
  for (const auto& sk : r.skipped) out << "  note: skipping " << sk << "\n";
  out << std::flush;
}

void print_text_best(const Report& r, std::ostream& out)
{
  if (r.results.empty()) return;
  const auto& b = r.results[r.best];
  out << "  note: best: ppwi " << b.sample.ppwi << ", wgsize " << b.sample.wgsize << " (lowest avg, "
      << format_fixed(b.avg_s, 8) << " s)\n" << std::flush;
}

void print_text_results(const Report& r, std::ostream& out)
{
  out << "\n"
      << pad_left("PPWI", 6) << pad_left("WG size", 9)
      << pad_left("Min (s)", 13) << pad_left("Max (s)", 13) << pad_left("Avg (s)", 13) << pad_left("Stddev (s)", 13)
      << pad_left("GFLOP/s", 11) << pad_left("GFInst/s", 11) << pad_left("GInter/s", 11)
      << pad_left("Overhead (s)", 14) << pad_left("Context (s)", 13) << "\n";
  for (const auto& c : r.results)
  {
    out << pad_left(std::to_string(c.sample.ppwi), 6) << pad_left(std::to_string(c.sample.wgsize), 9)
        << pad_left(format_fixed(c.min_s, 8), 13) << pad_left(format_fixed(c.max_s, 8), 13)
        << pad_left(format_fixed(c.avg_s, 8), 13) << pad_left(format_fixed(c.stddev_s, 8), 13)
        << pad_left(format_fixed(c.gflops, 3), 11) << pad_left(format_fixed(c.gfinsts, 3), 11)
        << pad_left(format_fixed(c.ginteractions, 3), 11)
        << pad_left(format_fixed(c.launch_overhead_s, 8), 14) << pad_left(format_fixed(c.sample.context_s, 8), 13)
        << "\n";
  }
  print_text_best(r, out);
  out << "  note: overhead is host wall time minus device time per launch; context is allocation and upload\n";
  out << std::flush;
}

void print_text_energy(const Report& r, std::ostream& out)
{
  out << "\n";
  if (!r.energy_enabled)
  {
    out << "Energy  disabled (--no-energy)\n" << std::flush;
    return;
  }

  const std::size_t lw = 18;
  out << pad_right("Energy", lw) << pad_left("J/iter", 12) << pad_left("Total (Wh)", 14)
      << pad_left("Avg power (W)", 16) << pad_left("Time (s)", 12)
      << pad_left("Iterations", 13) << pad_left("Samples", 10) << "\n";
  bool few = false;
  for (const auto& c : r.results)
  {
    const EnergyResult& e = c.sample.energy;
    out << pad_right(config_label(c), lw)
        << pad_left(format_fixed(e.joules_per_iter, 6), 12)
        << pad_left(format_fixed(e.total_wh, 6), 14)
        << pad_left(format_fixed(e.avg_power_w, 2), 16)
        << pad_left(format_fixed(e.seconds, 6), 12)
        << pad_left(format_thousands((long long)e.iterations), 13)
        << pad_left(format_thousands(e.samples), 10) << "\n";
    if (e.samples < 2) few = true;
  }
  if (few)
    out << "  note: fewer than two power samples in some configurations; their energy figures may be unreliable "
           "(raise -i)\n";
  out << std::flush;
}

void print_text_correctness(const Report& r, std::ostream& out)
{
  const std::string prec = precision_name(r.precision);
  out << "\n" << "Correctness (" << prec << ")\n";

  // vs the fp64 reference pass
  out << "vs fp64 reference\n";
  bool any_ref = false;
  for (const auto& c : r.results) any_ref = any_ref || c.has_ref;
  if (!any_ref)
  {
    out << "  note: float64 is the reference precision; there is no separate reference pass\n";
  }
  else
  {
    out << pad_left("PPWI", 6) << pad_left("WG size", 9) << pad_left("L2 rel", 12) << pad_left("RMS abs", 12)
        << pad_left("Mean signed", 13) << pad_left("Max abs", 18) << pad_left("Max rel", 18)
        << pad_left("Max ULP", 14) << "\n";
    std::vector<std::vector<std::string>> cells;
    for (const auto& c : r.results)
    {
      const Metrics& m = c.vs_ref;
      cells.push_back({format_sci(m.l2RelNorm, 3), format_sci(m.rmsAbs, 3), format_sci(m.meanSigned, 3),
                       format_sci(m.maxAbs, 3) + " @" + std::to_string(m.maxAbsIdx),
                       format_sci(m.maxRel, 3) + " @" + std::to_string(m.maxRelIdx),
                       format_thousands(m.maxUlp) + " @" + std::to_string(m.maxUlpIdx)});
    }
    print_collapsible(out, r, cells, {12, 12, 13, 18, 18, 14});

    // ULP histograms, grouped when identical
    std::map<std::string, std::vector<std::string>> groups;
    std::vector<std::string> order;
    for (const auto& c : r.results)
    {
      const std::string h = histogram_string(c.vs_ref);
      if (!groups.count(h)) order.push_back(h);
      groups[h].push_back(config_label(c));
    }
    if (order.size() == 1)
      out << "  ULP histogram (all configurations): " << order[0] << "\n";
    else
      for (const auto& h : order)
      {
        std::string who;
        for (const auto& l : groups[h]) who += (who.empty() ? "" : "; ") + l;
        out << "  ULP histogram (" << who << "): " << h << "\n";
      }
    out << "  note: \"@N\" is the pose index; ULP distances are measured after rounding both values to " << prec << "\n";
  }

  // vs the deck reference energies
  out << "vs deck ref_energies.out\n";
  out << pad_left("PPWI", 6) << pad_left("WG size", 9) << pad_left("L2 rel", 12) << pad_left("Max rel", 18)
      << pad_left("Beyond tol", 16) << "\n";
  {
    std::vector<std::vector<std::string>> cells;
    for (const auto& c : r.results)
    {
      const Metrics& m = c.vs_deck;
      cells.push_back({format_sci(m.l2RelNorm, 3), format_sci(m.maxRel, 3) + " @" + std::to_string(m.maxRelIdx),
                       format_thousands((long long)m.beyondTolerance) + "/" + format_thousands((long long)m.compared)});
    }
    print_collapsible(out, r, cells, {12, 18, 16});
  }
  out << "  note: the deck energies are low-precision text, so this comparison is informational; tolerance "
      << format_fixed(REPORT_TOLERANCE_PCT, 3) << "%\n";

  // non-finite results
  for (const auto& c : r.results)
  {
    const std::size_t nf = std::max(c.vs_deck.nonFinite, c.has_ref ? c.vs_ref.nonFinite : 0);
    if (nf > 0)
      out << "  note: " << config_label(c) << ": " << format_thousands((long long)nf)
          << " non-finite energies were left out of the metrics\n";
  }

  // first energies of the first configuration
  if (!r.results.empty() && r.rows > 0)
  {
    const auto& first = r.results[0];
    const std::size_t n = std::min(r.rows, first.sample.energies.size());
    out << "Energies (first " << n << " poses, " << config_label(first) << ")\n";
    for (std::size_t i = 0; i < n; i++)
      out << pad_left(std::to_string(i), 8) << pad_left(format_sci(first.sample.energies[i], 9), 20) << "\n";

    std::vector<std::string> differ;
    for (std::size_t k = 1; k < r.results.size(); k++)
    {
      double maxd = 0.0;
      const auto& e = r.results[k].sample.energies;
      for (std::size_t i = 0; i < e.size() && i < first.sample.energies.size(); i++)
        maxd = std::max(maxd, std::fabs(e[i] - first.sample.energies[i]));
      if (maxd > 0.0 || e != first.sample.energies)
        differ.push_back(config_label(r.results[k]) + " (max |diff| " + format_sci(maxd, 3) + ")");
    }
    if (r.results.size() > 1)
    {
      if (differ.empty())
        out << "  note: all configurations give bit-identical energies\n";
      else
        for (const auto& d : differ) out << "  note: differs from the first configuration: " << d << "\n";
    }
  }
  out << std::flush;
}

// ---- CSV -------------------------------------------------------------------

void print_csv(const Report& r, std::ostream& out, bool with_header)
{
  if (with_header)
    out << "implementation,backend,device,precision,deck,natlig,natpro,nposes,ppwi,wgsize,iterations,"
           "min_s,max_s,avg_s,stddev_s,gflops,gfinsts,ginteractions,launch_overhead_s,context_s,"
           "energy_j_per_iter,avg_power_w,ref_l2_rel,ref_max_rel,ref_max_ulp,deck_l2_rel,deck_beyond_tol\n";

  const std::string prefix = csv_prefix(r);
  for (const auto& c : r.results)
  {
    out << prefix << c.sample.ppwi << "," << c.sample.wgsize << "," << r.iterations << ","
        << format_fixed(c.min_s, 9) << "," << format_fixed(c.max_s, 9) << "," << format_fixed(c.avg_s, 9) << ","
        << format_fixed(c.stddev_s, 9) << ","
        << format_fixed(c.gflops, 4) << "," << format_fixed(c.gfinsts, 4) << "," << format_fixed(c.ginteractions, 4) << ","
        << format_fixed(c.launch_overhead_s, 9) << "," << format_fixed(c.sample.context_s, 9) << ",";
    if (r.energy_enabled)
      out << format_fixed(c.sample.energy.joules_per_iter, 6) << "," << format_fixed(c.sample.energy.avg_power_w, 3);
    else
      out << ",";
    out << ",";
    if (c.has_ref)
      out << csv_sci(c.vs_ref.l2RelNorm) << "," << csv_sci(c.vs_ref.maxRel) << "," << c.vs_ref.maxUlp;
    else
      out << ",,";
    out << "," << csv_sci(c.vs_deck.l2RelNorm) << "," << c.vs_deck.beyondTolerance << "\n";
  }
  out << std::flush;
}

void print_csv_raw(const Report& r, std::ostream& out, bool with_header)
{
  if (with_header)
    out << "implementation,backend,device,precision,deck,natlig,natpro,nposes,ppwi,wgsize,iteration,time_s\n";
  const std::string prefix = csv_prefix(r);
  for (const auto& c : r.results)
    for (std::size_t j = 0; j < c.sample.kernel_s.size(); j++)
      out << prefix << c.sample.ppwi << "," << c.sample.wgsize << "," << (j + 1) << ","
          << format_fixed(c.sample.kernel_s[j], 9) << "\n";
  out << std::flush;
}
