// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

<<<<<<< HEAD
// BabelStream driver: parses the command line, validates the configuration
// against the device limits before allocating anything, runs the five
// kernels num_times times with device-event timing (the first iteration is
// warm-up and is excluded from statistics, CSV and energy), and hands one
// report struct to the formatters.

#include <climits>
#include <iostream>
#include <memory>
#include <string>

#include "backend.h"
#include "cli.h"
#include "energy.h"
#include "report.h"
#include "stream.h"

namespace {

bool is_power_of_two(long long v) { return v > 0 && (v & (v - 1)) == 0; }

// Throws CliError naming the limit that a setting breaks.
void validate(const Options& o, const DeviceInfo& dev, int dot_groups)
{
  const std::string device = "device " + std::to_string(dev.index) + " (" + dev.name + ")";

  if (o.precision == Precision::Half && !dev.half_supported)
    throw CliError("float16 is not supported on " + device +
                   " (needs native fp16 arithmetic); use --float or --double");

  if (!is_power_of_two(o.block_size))
    throw CliError("block size " + std::to_string(o.block_size) +
                   " must be a power of two (the Dot reduction halves the block each step)");

  if (o.block_size > kMaxBlockSize)
    throw CliError("block size " + std::to_string(o.block_size) + " exceeds the Dot reduction's " +
                   std::to_string(kMaxBlockSize) + " shared-memory slots");

  if (dev.max_threads_per_block > 0 && o.block_size > dev.max_threads_per_block)
    throw CliError("block size " + std::to_string(o.block_size) + " exceeds the " +
                   std::to_string(dev.max_threads_per_block) + " threads per block allowed on " + device);

  if (o.array_size > INT_MAX)
    throw CliError("array size " + std::to_string(o.array_size) +
                   " does not fit the kernels' 32-bit index (maximum " + std::to_string(INT_MAX) + ")");

  if (o.array_size % o.block_size != 0)
    throw CliError("array size " + std::to_string(o.array_size) + " is not a multiple of the block size " +
                   std::to_string(o.block_size) + " (the kernels have no bounds check)");

  const long long dot_stride = static_cast<long long>(o.block_size) * dot_groups;
  if (o.array_size + dot_stride > INT_MAX)
    throw CliError("array size " + std::to_string(o.array_size) + " plus the Dot grid stride " +
                   std::to_string(dot_stride) + " overflows the kernels' 32-bit index (maximum " +
                   std::to_string(INT_MAX) + ")");

  const long long grid = o.array_size / o.block_size;
  if (dev.max_grid_x > 0 && grid > dev.max_grid_x)
    throw CliError("grid of " + std::to_string(grid) + " blocks exceeds the device maximum of " +
                   std::to_string(dev.max_grid_x) + "; use a larger --block-size");
}

int run(const Options& o)
{
  const bool csv_mode = (o.output != OutputMode::Text);
  std::ostream& text_out = csv_mode ? std::cerr : std::cout;

  const int count = device_count();
  if (count == 0)
    throw std::runtime_error(std::string("no ") + backend_name() + " devices found");
  if (o.device >= count)
    throw CliError("device index " + std::to_string(o.device) + " is out of range (" +
                   std::to_string(count) + (count == 1 ? " device" : " devices") + " found; see --list)");

  const DeviceInfo dev = query_device(o.device);

  Report r;
  r.implementation = backend_name();
  r.backend = backend_name();
  r.device = dev.name;
  r.driver = dev.driver;
  r.precision = o.precision;
  r.array_size = o.array_size;
  r.block_size = o.block_size;
  r.compute_units = dev.compute_units;
  r.compute_unit_label = compute_unit_label();
  r.num_times = o.num_times;
  r.fmad = fmad_setting();
  r.timing = "device timestamps";
  r.mibibytes = o.mibibytes;
  r.header_notes = dev.notes;

  if (dev.compute_units > 0)
  {
    r.dot_groups = 4 * dev.compute_units;
  }
  else
  {
    r.dot_groups = 64;
    std::string unit = compute_unit_label();   // "SMs" -> "SM"
    if (!unit.empty() && unit.back() == 's') unit.pop_back();
    r.header_notes.push_back(unit + " count unavailable from the runtime and vendor library; Dot uses 64 groups");
  }

  validate(o, dev, r.dot_groups);

  EnergyMeter meter(o.device, o.energy);
  r.energy.enabled = meter.enabled();

  print_text_header(r, text_out);

  StreamConfig cfg;
  cfg.array_size = static_cast<int>(o.array_size);
  cfg.block_size = o.block_size;
  cfg.dot_groups = r.dot_groups;
  std::unique_ptr<Stream> stream = make_stream(o.precision, cfg);

  const double elem = static_cast<double>(precision_bytes(o.precision));
  const double n = static_cast<double>(o.array_size);

  RoutineResult init  {"Init",  3 * elem * n, true,  {}};
  RoutineResult copy  {"Copy",  2 * elem * n, false, {}};
  RoutineResult mul   {"Mul",   2 * elem * n, false, {}};
  RoutineResult add   {"Add",   3 * elem * n, false, {}};
  RoutineResult triad {"Triad", 3 * elem * n, false, {}};
  RoutineResult dot   {"Dot",   2 * elem * n, false, {}};

  init.times.push_back(stream->init());

  for (unsigned int k = 0; k < o.num_times; k++)
  {
    // Iteration 0 is warm-up: excluded from statistics, CSV and energy.
    const bool timed = (k > 0);
    if (k == 1) meter.start();

    const double t_copy  = stream->copy();
    const double t_mul   = stream->mul();
    const double t_add   = stream->add();
    const double t_triad = stream->triad();
    const double t_dot   = stream->dot();

    if (timed)
    {
      copy.times.push_back(t_copy);
      mul.times.push_back(t_mul);
      add.times.push_back(t_add);
      triad.times.push_back(t_triad);
      dot.times.push_back(t_dot);
    }
  }
  meter.stop();

  r.routines = {init, copy, mul, add, triad, dot};
  r.energy = meter.result(o.num_times - 1);
  r.correctness = stream->check(o.num_times);

  switch (o.output)
  {
    case OutputMode::Text:
      print_text_results(r, std::cout);
      break;
    case OutputMode::Csv:
      print_csv(r, std::cout);
      break;
    case OutputMode::CsvRaw:
      print_csv_raw(r, std::cout);
      break;
  }
  print_text_energy(r, text_out);
  print_text_correctness(r, text_out);

  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  Options o;
  try
  {
    o = parse_cli(argc, argv);

    if (o.help)
    {
      std::cout << help_text(argv[0]);
      return 0;
    }

    if (o.list)
    {
      const auto devices = list_devices();
      if (devices.empty())
      {
        std::cerr << "error: no " << backend_name() << " devices found" << std::endl;
        return 1;
      }
      for (const auto& d : devices)
        std::cout << d.first << ": " << d.second << "\n";
      return 0;
    }

    return run(o);
  }
  catch (const CliError& e)
  {
    std::cerr << "error: " << e.what() << " (try '--help')" << std::endl;
    return 1;
  }
  catch (const std::exception& e)
  {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
=======
// Runs the five BabelStream kernels with HIP event timing and reports bandwidth after discarding the first sample. Correctness is checked by replaying the scalar recurrence, using compensated host sums, elementwise errors, and a dot-product underflow diagnostic. The default array has 2^25 elements, and output supports decimal or binary bandwidth units plus CSV.

#include "hip/hip_runtime.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>

#define VERSION_STRING "5.0"

#include "Stream.h"
#include "HIPStream.h"
#include "hip/hip_runtime.h"

int ARRAY_SIZE = 33554432;
unsigned int num_times = 1000;
unsigned int deviceIndex = 0;
bool use_float = false;
bool mibibytes = false;
bool csv_output = false;

void check_error_main(void)
{
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
  {
    std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
    exit(err);
  }
}

template <typename T>
void run();

void parseArguments(int argc, char *argv[]);

int main(int argc, char *argv[])
{

  parseArguments(argc, argv);

  if (use_float)
    run<float>();
  else
    run<double>();
}

template <typename T>
std::vector<std::vector<double>> run_all(Stream<T> *stream, T& sum)
{

  std::vector<std::vector<double>> timings(5);

  for (unsigned int k = 0; k < num_times; k++)
  {
    stream->copy();
    timings[0].push_back(stream->getTimeTaken());

    stream->mul();
    timings[1].push_back(stream->getTimeTaken());

    stream->add();
    timings[2].push_back(stream->getTimeTaken());

    stream->triad();
    timings[3].push_back(stream->getTimeTaken());

    sum = stream->dot();
    timings[4].push_back(stream->getTimeTaken());

  }

  return timings;
}

template <typename T>
void check_solution(const unsigned int ntimes,
                    std::vector<T>& a, std::vector<T>& b, std::vector<T>& c,
                    const T dot_sum)
{
  T goldA = startA;
  T goldB = startB;
  T goldC = startC;
  const T scalar = startScalar;

  for (unsigned int i = 0; i < ntimes; i++)
  {
    goldC = goldA;
    goldB = scalar * goldC;
    goldC = goldA + goldB;
    goldA = goldB + scalar * goldC;
  }

  auto kahan_sum = [](const std::vector<T>& v)
  {
    double sum = 0.0;
    double comp = 0.0;
    for (size_t i = 0; i < v.size(); i++)
    {
      const double y = static_cast<double>(v[i]) - comp;
      const double t = sum + y;
      comp = (t - sum) - y;
      sum = t;
    }
    return sum;
  };

  auto metrics = [&kahan_sum](const std::vector<T>& v, const T gold,
                              double& checksum_rel_err, double& max_rel_err)
  {
    const double gold_d = static_cast<double>(gold);
    const double actual_sum = kahan_sum(v);
    const double expected_sum = gold_d * static_cast<double>(v.size());

    checksum_rel_err = (expected_sum != 0.0)
      ? std::fabs(actual_sum - expected_sum) / std::fabs(expected_sum)
      : std::fabs(actual_sum);

    double max_err = 0.0;
    for (size_t i = 0; i < v.size(); i++)
    {
      const double abs_err = std::fabs(static_cast<double>(v[i]) - gold_d);
      const double rel_err = (gold_d != 0.0) ? abs_err / std::fabs(gold_d) : abs_err;
      if (rel_err > max_err) max_err = rel_err;
    }
    max_rel_err = max_err;
  };

  double checksumA, checksumB, checksumC;
  double maxErrA, maxErrB, maxErrC;

  metrics(a, goldA, checksumA, maxErrA);
  metrics(b, goldB, checksumB, maxErrB);
  metrics(c, goldC, checksumC, maxErrC);

  const double expected_dot =
    static_cast<double>(goldA) * static_cast<double>(goldB) * static_cast<double>(ARRAY_SIZE);
  const double actual_dot = static_cast<double>(dot_sum);

  const double product_magnitude =
    std::fabs(static_cast<double>(goldA) * static_cast<double>(goldB));
  const bool dot_underflows =
    product_magnitude < static_cast<double>(std::numeric_limits<T>::min());

  const double dotErr = (expected_dot != 0.0)
    ? std::fabs(actual_dot - expected_dot) / std::fabs(expected_dot)
    : std::fabs(actual_dot);

  std::streamsize ss = std::cout.precision();
  std::ostream& out = csv_output ? std::cerr : std::cout;

  out << std::endl;
  out << "Correctness metrics (" << (sizeof(T) == sizeof(float) ? "float32" : "float64")
      << ", " << ntimes << " iterations)" << std::endl;

  out << std::left  << std::setw(10) << "Array"
      << std::left  << std::setw(18) << "Expected"
      << std::left  << std::setw(18) << "Checksum rel err"
      << std::left  << std::setw(18) << "Max elem rel err"
      << std::endl;

  auto report = [&](const char* name, double gold, double chk, double mx)
  {
    out << std::left << std::setw(10) << name
        << std::left << std::setw(18) << std::scientific << std::setprecision(6) << gold
        << std::left << std::setw(18) << std::scientific << std::setprecision(3) << chk
        << std::left << std::setw(18) << std::scientific << std::setprecision(3) << mx
        << std::endl;
  };

  report("a", static_cast<double>(goldA), checksumA, maxErrA);
  report("b", static_cast<double>(goldB), checksumB, maxErrB);
  report("c", static_cast<double>(goldC), checksumC, maxErrC);

  out << std::left << std::setw(10) << "dot"
      << std::left << std::setw(18) << std::scientific << std::setprecision(6) << expected_dot
      << std::left << std::setw(18) << std::scientific << std::setprecision(3) << dotErr
      << std::left << std::setw(18) << "-"
      << std::endl;
  out << "  (actual dot   = " << std::scientific << std::setprecision(9) << actual_dot << ")" << std::endl;
  if (dot_underflows)
  {
    out << "  (note: expected element product " << std::scientific << std::setprecision(3)
        << product_magnitude << " underflows this precision at this iteration count,"
        << " so the dot relative error above is not meaningful)" << std::endl;
  }

  std::cout.unsetf(std::ios_base::floatfield);
  std::cout.precision(ss);
}

template <typename T>
void run()
{
  std::streamsize ss = std::cout.precision();

  if (!csv_output) {
    std::cout << "Running kernels " << num_times << " times" << std::endl;

    if (sizeof(T) == sizeof(float))
      std::cout << "Precision: float" << std::endl;
    else
      std::cout << "Precision: double" << std::endl;


    if (mibibytes)
    {
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*std::pow(2.0, -20.0) << " MiB"
                << " (=" << ARRAY_SIZE*sizeof(T)*std::pow(2.0, -30.0) << " GiB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*std::pow(2.0, -20.0) << " MiB"
                << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*std::pow(2.0, -30.0) << " GiB)" << std::endl;
    }
    else
    {
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
                << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
                << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
    }
    std::cout.precision(ss);
  }

  int device = deviceIndex;
  hipSetDevice(device);
  check_error_main();
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);
  check_error_main();

  Stream<T> *stream;
  stream = new CUDAStream<T>(ARRAY_SIZE, deviceIndex);

  stream->init_arrays(startA, startB, startC);
  auto initElapsedS = stream->getTimeTaken();

  T sum{};

  std::vector<std::vector<double>> timings;

  timings = run_all<T>(stream, sum);

  auto initBWps = ((mibibytes ? std::pow(2.0, -20.0) : 1.0E-9) * (3 * sizeof(T) * ARRAY_SIZE)) / initElapsedS;

  if (!csv_output) {
    std::cout << "Init: "
      << std::setw(7)
      << initElapsedS
      << " s (="
      << initBWps
      << (mibibytes ? " MiBytes/sec" : " GBytes/sec")
      << ")" << std::endl;

    std::cout
      << std::left << std::setw(12) << "Function"
      << std::left << std::setw(12) << ((mibibytes) ? "MiBytes/sec" : "GBytes/sec")
      << std::left << std::setw(12) << "Min (sec)"
      << std::left << std::setw(12) << "Max"
      << std::left << std::setw(12) << "Average"
      << std::endl
      << std::fixed;
  }
  else {
    std::cout << "backend,GPU,precision,vec_size,routine,BW_GBs" << std::endl;
  }

  std::vector<std::string> labels;
  std::vector<size_t> sizes;

  labels = {"Copy", "Mul", "Add", "Triad", "Dot"};
  sizes = {
    2 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE};

    for (size_t i = 0; i < timings.size(); ++i)
    {
      auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

      double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);

      std::string precision;
      if (sizeof(T) == sizeof(float))
        precision = "float32";
      else
        precision = "float64";

      if (!csv_output) {
        std::cout
          << std::left << std::setw(12) << labels[i]
          << std::left << std::setw(12) << std::setprecision(3) <<
            ((mibibytes) ? std::pow(2.0, -20.0) : 1.0E-9) * sizes[i] / (*minmax.first)
          << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
          << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
          << std::left << std::setw(12) << std::setprecision(5) << average
          << std::endl;
      }
      else {
        for (size_t j = 0; j < timings[i].size(); ++j) {
          std::cout << "CUDA," << props.name << "," << precision
                    << "," << ARRAY_SIZE << "," << labels[i] << ","
                    << 1.0E-9 * sizes[i] / timings[i][j] << std::endl;
        }
      }
    }

  std::vector<T> a(ARRAY_SIZE);
  std::vector<T> b(ARRAY_SIZE);
  std::vector<T> c(ARRAY_SIZE);
  stream->read_arrays(a, b, c);

  check_solution<T>(num_times, a, b, c, sum);

  delete stream;
}

int parseUInt(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

int parseInt(const char *str, int *output)
{
  char *next;
  *output = strtol(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!std::string("--list").compare(argv[i]))
    {
      listDevices();
      exit(EXIT_SUCCESS);
    }
    else if (!std::string("--device").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &deviceIndex))
      {
        std::cerr << "Invalid device index." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--arraysize").compare(argv[i]) ||
             !std::string("-s").compare(argv[i]))
    {
      if (++i >= argc || !parseInt(argv[i], &ARRAY_SIZE) || ARRAY_SIZE <= 0)
      {
        std::cerr << "Invalid array size." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--numtimes").compare(argv[i]) ||
             !std::string("-n").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &num_times))
      {
        std::cerr << "Invalid number of times." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (num_times < 2)
      {
        std::cerr << "Number of times must be 2 or more" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--float").compare(argv[i]))
    {
      use_float = true;
    }
    else if (!std::string("--mibibytes").compare(argv[i]))
    {
      mibibytes = true;
    }
    else if (!std::string("--csv").compare(argv[i]))
    {
      csv_output = true;
    }
    else if (!std::string("--help").compare(argv[i]) ||
             !std::string("-h").compare(argv[i]))
    {
      std::cout << std::endl;
      std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "      --list               List available devices" << std::endl;
      std::cout << "      --device     INDEX   Select device at INDEX" << std::endl;
      std::cout << "  -s  --arraysize  SIZE    Use SIZE elements in the array" << std::endl;
      std::cout << "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)" << std::endl;
      std::cout << "      --float              Use floats (rather than doubles)" << std::endl;
      std::cout << "      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << std::endl;
      exit(EXIT_SUCCESS);
    }
    else
    {
      std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
      exit(EXIT_FAILURE);
    }
>>>>>>> main
  }
}
