// Command-line parsing and --help for Hartree-Fock.

#include "cli.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <sstream>

namespace {

bool parse_ll(const char* s, long long& out)
{
  if (s == nullptr || *s == '\0') return false;
  errno = 0;
  char* end = nullptr;
  const long long v = std::strtoll(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') return false;
  out = v;
  return true;
}

const char* value_for(int& i, int argc, char* argv[])
{
  if (i + 1 >= argc) throw CliError(std::string("missing value for ") + argv[i]);
  return argv[++i];
}

int int_value(int& i, int argc, char* argv[], const char* what, long long min)
{
  const char* v = value_for(i, argc, argv);
  long long n = 0;
  if (!parse_ll(v, n) || n < min || n > INT_MAX)
    throw CliError(std::string("invalid ") + what + " '" + v + "' (must be an integer >= " +
                   std::to_string(min) + ")");
  return static_cast<int>(n);
}

} // namespace

Options parse_cli(int argc, char* argv[])
{
  Options o;
  bool csv = false, csv_raw = false;

  for (int i = 1; i < argc; i++)
  {
    const std::string arg = argv[i];

    if (arg == "-h" || arg == "--help")        { o.help = true; }
    else if (arg == "--list")                  { o.list = true; }
    else if (arg == "--device")                { o.device = int_value(i, argc, argv, "device index", 0); }
    else if (arg == "-n" || arg == "--iters")  { o.num_times = int_value(i, argc, argv, "number of iterations", 1); }
    else if (arg == "--block-size")            { o.block_size = int_value(i, argc, argv, "block size", 1); }
    else if (arg == "--max-check-atoms")       { o.max_check_atoms = int_value(i, argc, argv, "--max-check-atoms value", 0); }
    else if (arg == "--double")                { o.precision = Precision::Double; }
    else if (arg == "--float")                 { o.precision = Precision::Float; }
    else if (arg == "--legacy-sqrt")           { o.sqrt_mode = SqrtMode::Legacy; }
    else if (arg == "--check")                 { o.check = true; o.force_check = true; }
    else if (arg == "--no-check")              { o.check = false; o.force_check = false; }
    else if (arg == "--csv")                   { csv = true; }
    else if (arg == "--csv-raw")               { csv_raw = true; }
    else if (arg == "--no-energy")             { o.energy = false; }
    else if (arg == "--half")
    {
      throw CliError("--half is not available for Hartree-Fock (the 1e-12 screening thresholds are below "
                     "the float16 range); use --float or --double");
    }
    else if (arg.size() > 1 && arg[0] == '-')
    {
      const std::size_t eq = arg.find('=');
      if (arg.rfind("--", 0) == 0 && eq != std::string::npos)
      {
        std::string flag = arg.substr(0, eq);
        if (flag == "--iters") flag = "-n";
        throw CliError("unrecognized argument '" + arg + "' (use '" + flag + " " + arg.substr(eq + 1) + "')");
      }
      throw CliError("unrecognized argument '" + arg + "'");
    }
    else
    {
      if (!o.input.empty())
        throw CliError("unexpected argument '" + arg + "' (only one input file is allowed; already have '" +
                       o.input + "')");
      o.input = arg;
    }
  }

  if (csv && csv_raw)
    throw CliError("--csv and --csv-raw cannot be used together");
  if (csv) o.output = OutputMode::Csv;
  if (csv_raw) o.output = OutputMode::CsvRaw;

  if (!o.help && !o.list && o.input.empty())
    throw CliError("missing input file");

  return o;
}

std::string help_text(const char* program)
{
  std::ostringstream h;
  h << "Usage: " << program << " INPUT [OPTIONS]\n"
    << "\n"
    << "Hartree-Fock Fock-matrix build (two-electron integrals over s-type Gaussians).\n"
    << "INPUT is a geometry file, e.g. ../../data/hartree-fock/he64.\n"
    << "\n"
    << "Options:\n"
    << "  -n, --iters NUM        timed Fock builds after one untimed build (default 10)\n"
    << "      --double           float64 (default)\n"
    << "      --float            float32\n"
    << "      --legacy-sqrt      Boys-function square roots in float32 (sqrtf) whatever\n"
    << "                         the precision, as the old mixed-precision build did\n"
    << "      --block-size N     threads per block, up to the device limit (default 256)\n"
    << "      --device INDEX     device to run on; the energy profiler uses the same\n"
    << "                         index (default 0)\n"
    << "      --list             list devices (index: name) and exit\n"
    << "      --check            always run the fp64 CPU reference check\n"
    << "      --no-check         skip the fp64 CPU reference check\n"
    << "      --max-check-atoms N  run the check automatically only up to N atoms\n"
    << "                         (default 128; the CPU reference grows as atoms^4)\n"
    << "      --csv              summary CSV on stdout, report on stderr\n"
    << "      --csv-raw          one CSV row per timed build on stdout, report on stderr\n"
    << "      --no-energy        skip board-power measurement\n"
    << "  -h, --help             show this help and exit\n"
    << "\n"
    << "The last precision flag wins. Options take their value as a separate\n"
    << "argument (-n 10, not --iters=10). Timed builds add into the same Fock\n"
    << "matrix; the checksum and the check use the untimed first build.\n"
    << "\n"
    << "Environment (all optional):\n"
    << "  GPU_NAME               device name to report instead of the queried one\n"
    << "  GPU_DRIVER             driver string to report instead of the queried one\n"
    << "\n"
    << "Build-time settings:\n"
    << "  make ARCH=...          GPU architecture; detected from the installed GPU\n";
  return h.str();
}
