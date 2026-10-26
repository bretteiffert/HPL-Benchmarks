// Command-line parsing and --help for miniBUDE.

#include "cli.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <sstream>

namespace {

bool parse_size(const std::string& s, std::size_t& out)
{
  if (s.empty() || s[0] == '-' || s[0] == '+') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0') return false;
  out = static_cast<std::size_t>(v);
  return true;
}

const char* value_for(int& i, int argc, char* argv[])
{
  if (i + 1 >= argc) throw CliError(std::string("missing value for ") + argv[i]);
  return argv[++i];
}

std::size_t size_value(int& i, int argc, char* argv[], const char* what, std::size_t min)
{
  const std::string v = value_for(i, argc, argv);
  std::size_t n = 0;
  if (!parse_size(v, n) || n < min || n > static_cast<std::size_t>(INT_MAX))
    throw CliError(std::string("invalid ") + what + " '" + v + "' (must be an integer >= " + std::to_string(min) + ")");
  return n;
}

std::vector<std::size_t> list_value(int& i, int argc, char* argv[], const char* what)
{
  const std::string v = value_for(i, argc, argv);
  std::vector<std::size_t> xs;
  std::istringstream ss(v);
  std::string token;
  while (std::getline(ss, token, ','))
  {
    std::size_t n = 0;
    if (!parse_size(token, n) || n == 0 || n > static_cast<std::size_t>(INT_MAX))
      throw CliError(std::string("invalid ") + what + " '" + token + "' in '" + v +
                     "' (a comma-separated list of positive integers)");
    xs.push_back(n);
  }
  if (xs.empty() || v.back() == ',')
    throw CliError(std::string("invalid ") + what + " list '" + v + "'");
  return xs;
}

std::string ppwi_list()
{
  std::string s;
  for (std::size_t i = 0; i < kPPWIs.size(); i++) s += (i ? "," : "") + std::to_string(kPPWIs[i]);
  return s;
}

} // namespace

Options parse_cli(int argc, char* argv[])
{
  Options o;
  bool csv = false, csv_raw = false;

  for (int i = 1; i < argc; i++)
  {
    const std::string arg = argv[i];

    if (arg == "-h" || arg == "--help" || arg == "help")       { o.help = true; }
    else if (arg == "-l" || arg == "--list" || arg == "list")  { o.list = true; }
    else if (arg == "-d" || arg == "--device")                 { o.device = value_for(i, argc, argv); }
    else if (arg == "--deck")                                  { o.deck = value_for(i, argc, argv); }
    else if (arg == "-i" || arg == "--iter")                   { o.iterations = size_value(i, argc, argv, "iteration count", 1); }
    else if (arg == "--warmup")                                { o.warmup = size_value(i, argc, argv, "warm-up count", 0); }
    else if (arg == "-n" || arg == "--poses")                  { o.poses = size_value(i, argc, argv, "pose count", 0); }
    else if (arg == "-o" || arg == "--out")                    { o.out = value_for(i, argc, argv); }
    else if (arg == "-r" || arg == "--rows")                   { o.rows = size_value(i, argc, argv, "row count", 0); }
    else if (arg == "-w" || arg == "--wgsize")                 { o.wgsizes = list_value(i, argc, argv, "work-group size"); }
    else if (arg == "-p" || arg == "--ppwi")
    {
      if (i + 1 < argc && std::string(argv[i + 1]) == "all")
      {
        i++;
        o.ppwis.assign(kPPWIs.begin(), kPPWIs.end());
      }
      else
      {
        o.ppwis = list_value(i, argc, argv, "PPWI");
        for (auto p : o.ppwis)
          if (std::find(kPPWIs.begin(), kPPWIs.end(), p) == kPPWIs.end())
            throw CliError("PPWI " + std::to_string(p) + " is not compiled in; use one of " + ppwi_list() + " or 'all'");
      }
    }
    else if (arg == "--double")    { o.precision = Precision::Double; }
    else if (arg == "--float")     { o.precision = Precision::Float; }
    else if (arg == "--csv")       { csv = true; }
    else if (arg == "--csv-raw")   { csv_raw = true; }
    else if (arg == "--no-energy") { o.energy = false; }
    else if (arg == "--half")
    {
      throw CliError("--half is not available for miniBUDE; use --float or --double");
    }
    else
    {
      const std::size_t eq = arg.find('=');
      if (arg.rfind("-", 0) == 0 && eq != std::string::npos)
        throw CliError("unrecognized argument '" + arg + "' (use '" + arg.substr(0, eq) + " " + arg.substr(eq + 1) + "')");
      throw CliError("unrecognized argument '" + arg + "'");
    }
  }

  if (csv && csv_raw)
    throw CliError("--csv and --csv-raw cannot be used together");
  if (o.ppwis != std::vector<std::size_t>{1})
    throw CliError("the pinned configuration requires --ppwi 1");
  if (o.wgsizes != std::vector<std::size_t>{256})
    throw CliError("the pinned configuration requires --wgsize 256");
  if (csv) o.output = OutputMode::Csv;
  if (csv_raw) o.output = OutputMode::CsvRaw;
  return o;
}

std::string help_text(const char* program)
{
  std::ostringstream h;
  h << "Usage: " << program << " [OPTIONS]\n"
    << "\n"
    << "miniBUDE: protein-ligand docking energies with one pinned launch configuration.\n"
    << "\n"
    << "Options:\n"
    << "      --deck DIR         input deck directory (default " << DATA_DIR << ")\n"
    << "  -i, --iter I           timed launches per configuration (default 8)\n"
    << "      --warmup N         untimed launches before them (default 2)\n"
    << "  -n, --poses N          poses to compute, 0 for all in the deck (default 0)\n"
    << "  -p, --ppwi N           pinned poses per work-item; N must be 1 (default 1)\n"
    << "  -w, --wgsize N         pinned work-group size; N must be 256 (default 256)\n"
    << "      --float            float32 (default); adds one untimed fp64 reference pass\n"
    << "      --double           optional float64 run\n"
    << "  -d, --device INDEX     device index from --list, or a substring of its name;\n"
    << "                         the energy profiler uses the same device (default 0)\n"
    << "  -l, --list             list devices (index: name) and exit\n"
    << "  -o, --out PATH         save the first configuration's energies to PATH\n"
    << "  -r, --rows N           energies shown in the text report (default 8)\n"
    << "      --csv              summary CSV on stdout, report on stderr\n"
    << "      --csv-raw          one CSV row per timed launch on stdout, report on stderr\n"
    << "      --no-energy        skip board-power measurement\n"
    << "  -h, --help             show this help and exit\n"
    << "\n"
    << "The pose count must be a multiple of 256. The last precision flag wins.\n"
    << "Options take their value\n"
    << "as a separate argument (-w 32, not -w=32).\n"
    << "\n"
    << "Environment (all optional):\n"
    << "  GPU_NAME               device name to report instead of the queried one\n"
    << "  GPU_DRIVER             driver string to report instead of the queried one\n"
    << "\n"
    << "Build-time settings (make VAR=value):\n"
    << "  ARCH                   GPU architecture; detected from the installed GPU\n"
    << "  EXTRA_FLAGS            -DUSE_SHARED (forcefield in shared memory),\n"
    << "                         -DNO_RESTRICT, -DMANAGED or -DPAGEFAULT (allocation)\n";
  return h.str();
}
