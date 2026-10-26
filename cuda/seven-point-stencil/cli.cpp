// Command-line parsing and --help for the seven-point stencil.

#include "cli.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <sstream>
#include <vector>

namespace {

bool parse_size(const std::string& s, std::size_t& out)
{
  if (s.empty() || s[0] == '-' || s[0] == '+') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0' || v > (unsigned long long)INT_MAX) return false;
  out = static_cast<std::size_t>(v);
  return true;
}

const char* value_for(int& i, int argc, char* argv[])
{
  if (i + 1 >= argc) throw CliError(std::string("missing value for ") + argv[i]);
  return argv[++i];
}

// "X", "X,Y" or "X,Y,Z"; missing trailing values take `fill`.
Dims dims_value(int& i, int argc, char* argv[], const char* what, std::size_t max_parts, std::size_t fill)
{
  const std::string v = value_for(i, argc, argv);
  std::vector<std::size_t> xs;
  std::istringstream ss(v);
  std::string token;
  while (std::getline(ss, token, ','))
  {
    std::size_t n = 0;
    if (!parse_size(token, n) || n == 0)
      throw CliError(std::string("invalid ") + what + " '" + v + "' (positive integers separated by commas)");
    xs.push_back(n);
  }
  if (xs.empty() || xs.size() > max_parts || v.back() == ',')
    throw CliError(std::string("invalid ") + what + " '" + v + "' (expected " +
                   (max_parts == 3 ? "X[,Y[,Z]]" : "X,Y,Z") + ")");
  while (xs.size() < 3) xs.push_back(fill);
  return {xs[0], xs[1], xs[2]};
}

} // namespace

Options parse_cli(int argc, char* argv[])
{
  Options o;
  bool csv = false, csv_raw = false;

  for (int i = 1; i < argc; i++)
  {
    const std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") { o.help = true; }
    else if (arg == "--list")           { o.list = true; }
    else if (arg == "--device")
    {
      std::size_t n = 0;
      const std::string v = value_for(i, argc, argv);
      if (!parse_size(v, n)) throw CliError("invalid device index '" + v + "'");
      o.device = static_cast<int>(n);
    }
    else if (arg == "-s")
    {
      std::size_t n = 0;
      const std::string v = value_for(i, argc, argv);
      if (!parse_size(v, n) || n == 0) throw CliError("invalid grid side '" + v + "' (a positive integer)");
      o.grid = {n, n, n};
    }
    else if (arg == "--grid")       { o.grid = dims_value(i, argc, argv, "grid", 3, 0); }
    else if (arg == "--block-size") { o.block = dims_value(i, argc, argv, "block size", 3, 1); }
    else if (arg == "-n")
    {
      std::size_t n = 0;
      const std::string v = value_for(i, argc, argv);
      if (!parse_size(v, n) || n == 0) throw CliError("invalid number of iterations '" + v + "' (an integer >= 1)");
      o.num_times = static_cast<int>(n);
    }
    else if (arg == "--double")    { o.precision = Precision::Double; }
    else if (arg == "--float")     { o.precision = Precision::Float; }
    else if (arg == "--half")      { o.precision = Precision::Half; }
    else if (arg == "--csv")       { csv = true; }
    else if (arg == "--csv-raw")   { csv_raw = true; }
    else if (arg == "--no-energy") { o.energy = false; }
    else if (!arg.empty() && arg.find_first_not_of("0123456789") == std::string::npos)
    {
      throw CliError("positional sizes are no longer accepted ('" + arg +
                     "'); use --grid X,Y,Z (or -s L) and --block-size X,Y,Z");
    }
    else
    {
      const std::size_t eq = arg.find('=');
      if (arg.rfind("-", 0) == 0 && eq != std::string::npos)
        throw CliError("unrecognized argument '" + arg + "' (use '" + arg.substr(0, eq) + " " + arg.substr(eq + 1) + "')");
      throw CliError("unrecognized argument '" + arg + "'");
    }
  }

  if (o.grid.y == 0 || o.grid.z == 0)
    throw CliError("--grid needs all three sizes, X,Y,Z");
  if (csv && csv_raw)
    throw CliError("--csv and --csv-raw cannot be used together");
  if (o.block.x != 64 || o.block.y != 4 || o.block.z != 1)
    throw CliError("the pinned configuration requires --block-size 64,4,1");
  if (csv) o.output = OutputMode::Csv;
  if (csv_raw) o.output = OutputMode::CsvRaw;
  return o;
}

std::string help_text(const char* program)
{
  std::ostringstream h;
  h << "Usage: " << program << " [OPTIONS]\n"
    << "\n"
    << "Seven-point finite-difference Laplacian on an equidistant 3D grid. The input\n"
    << "is a quadratic field whose exact interior Laplacian is 3.\n"
    << "\n"
    << "Options:\n"
    << "  -s L                   cubic grid of L x L x L points (default 1024)\n"
    << "      --grid X,Y,Z       grid of X x Y x Z points\n"
    << "      --block-size X,Y,Z pinned threads per block; must be 64,4,1\n"
    << "                         (default 64,4,1)\n"
    << "  -n NUM                 timed runs after one untimed run (default 100)\n"
    << "      --double           float64 (default)\n"
    << "      --float            float32\n"
    << "      --half             float16; only up to 105 points per side, where the\n"
    << "                         centre coefficient stays within the float16 range\n"
    << "      --device INDEX     device to run on; the energy profiler uses the same\n"
    << "                         index (default 0)\n"
    << "      --list             list devices (index: name) and exit\n"
    << "      --csv              summary CSV on stdout, report on stderr\n"
    << "      --csv-raw          one CSV row per timed run on stdout, report on stderr\n"
    << "      --no-energy        skip board-power measurement\n"
    << "  -h, --help             show this help and exit\n"
    << "\n"
    << "Bandwidth is the theoretical traffic: every point read except the 8 corners\n"
    << "and 12 edges, every interior point written. The last precision flag wins.\n"
    << "Options take their value as a separate argument (-n 100, not -n=100).\n"
    << "\n"
    << "Environment (all optional):\n"
    << "  GPU_NAME               device name to report instead of the queried one\n"
    << "  GPU_DRIVER             driver string to report instead of the queried one\n"
    << "\n"
    << "Build-time settings:\n"
    << "  make ARCH=...          GPU architecture; detected from the installed GPU\n";
  return h.str();
}
