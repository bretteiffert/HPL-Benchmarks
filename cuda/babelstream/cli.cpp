// Command-line parsing and --help for BabelStream.

#include "cli.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace {

// Whole-string base-10 integer; no sign for unsigned use, no trailing junk.
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

} // namespace

Options parse_cli(int argc, char* argv[])
{
  Options o;
  bool csv = false, csv_raw = false;

  for (int i = 1; i < argc; i++)
  {
    const std::string arg = argv[i];

    if (arg == "-h" || arg == "--help")
    {
      o.help = true;
    }
    else if (arg == "--list")
    {
      o.list = true;
    }
    else if (arg == "--device")
    {
      const char* v = value_for(i, argc, argv);
      long long n = 0;
      if (!parse_ll(v, n) || n < 0 || n > INT_MAX)
        throw CliError(std::string("invalid device index '") + v + "'");
      o.device = static_cast<int>(n);
    }
    else if (arg == "-s" || arg == "--arraysize")
    {
      const char* v = value_for(i, argc, argv);
      long long n = 0;
      if (!parse_ll(v, n) || n <= 0)
        throw CliError(std::string("invalid array size '") + v + "' (must be a positive integer)");
      o.array_size = n;
    }
    else if (arg == "-n" || arg == "--numtimes")
    {
      const char* v = value_for(i, argc, argv);
      long long n = 0;
      if (!parse_ll(v, n) || n < 0 || n > UINT_MAX)
        throw CliError(std::string("invalid number of times '") + v + "'");
      if (n < 2)
        throw CliError("number of times must be 2 or more (the first iteration is warm-up), got " +
                       std::to_string(n));
      o.num_times = static_cast<unsigned int>(n);
    }
    else if (arg == "--block-size")
    {
      const char* v = value_for(i, argc, argv);
      long long n = 0;
      if (!parse_ll(v, n) || n <= 0 || n > INT_MAX)
        throw CliError(std::string("invalid block size '") + v + "' (must be a positive integer)");
      o.block_size = static_cast<int>(n);
    }
    else if (arg == "--double") { o.precision = Precision::Double; }
    else if (arg == "--float")  { o.precision = Precision::Float; }
    else if (arg == "--half")   { o.precision = Precision::Half; }
    else if (arg == "--mibibytes") { o.mibibytes = true; }
    else if (arg == "--csv")       { csv = true; }
    else if (arg == "--csv-raw")   { csv_raw = true; }
    else if (arg == "--no-energy") { o.energy = false; }
    else
    {
      const std::size_t eq = arg.find('=');
      if (arg.rfind("--", 0) == 0 && eq != std::string::npos)
        throw CliError("unrecognized argument '" + arg + "' (use '" + arg.substr(0, eq) + " " +
                       arg.substr(eq + 1) + "')");
      throw CliError("unrecognized argument '" + arg + "'");
    }
  }

  if (csv && csv_raw)
    throw CliError("--csv and --csv-raw cannot be used together");
  if (csv) o.output = OutputMode::Csv;
  if (csv_raw) o.output = OutputMode::CsvRaw;

  return o;
}

std::string help_text(const char* program)
{
  std::ostringstream h;
  h << "Usage: " << program << " [OPTIONS]\n"
    << "\n"
    << "BabelStream memory-bandwidth benchmark: Copy, Mul, Add, Triad and Dot.\n"
    << "\n"
    << "Options:\n"
    << "  -s, --arraysize SIZE   elements per array (default 33554432)\n"
    << "  -n, --numtimes NUM     iterations, NUM >= 2; the first is warm-up and\n"
    << "                         excluded from statistics, CSV and energy (default 1000)\n"
    << "      --double           float64 (default)\n"
    << "      --float            float32\n"
    << "      --half             float16 (values decay below the fp16 range after\n"
    << "                         a few hundred iterations; see the correctness notes)\n"
    << "      --block-size N     threads per block, a power of two up to 1024 and the\n"
    << "                         device limit; SIZE must be a multiple of it (default 1024)\n"
    << "      --device INDEX     device to run on; the energy profiler uses the same\n"
    << "                         index (default 0)\n"
    << "      --list             list devices (index: name) and exit\n"
    << "      --csv              summary CSV on stdout, report on stderr\n"
    << "      --csv-raw          one CSV row per timed iteration on stdout, report on stderr\n"
    << "      --no-energy        skip board-power measurement\n"
    << "      --mibibytes        text report in MiB/s and MiB (CSV is always GB/s)\n"
    << "  -h, --help             show this help and exit\n"
    << "\n"
    << "The last precision flag wins. Options take their value as a separate\n"
    << "argument (--device 1, not --device=1).\n"
    << "\n"
    << "Environment (all optional):\n"
    << "  GPU_NAME               device name to report instead of the queried one\n"
    << "  GPU_DRIVER             driver string to report instead of the queried one\n"
    << "\n"
    << "Build-time settings:\n"
    << "  make ARCH=...          GPU architecture; detected from the installed GPU\n"
    << "  make EXTRA_FLAGS=-DFMAD=0\n"
    << "                         FMAD=1 (default) uses explicit fused multiply-add in\n"
    << "                         Triad and Dot, 0 writes a*b+c; the compiler contracts\n"
    << "                         that to an FMA by default, so both normally build the same\n";
  return h.str();
}
