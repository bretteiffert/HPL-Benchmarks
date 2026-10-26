// miniBUDE problem types and the run interface. Plain C++ (no vendor
// headers) so host-only modules can include it. The deck layout, typed
// constants, device structs and converters are the original bude.h code;
// the run parameters, the Bude backend class and the sample struct are
// replaced by Deck (deck.h), the functions below and the report.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "energy.h"

// ---- original bude.h definitions -------------------------------------------

#define REPORT_TOLERANCE_PCT 0.025

#define HBTYPE_F 70
#define HBTYPE_E 69

template <int Bits> struct PrecisionOf;
template <> struct PrecisionOf<32> { using type = float; };
template <> struct PrecisionOf<64> { using type = double; };
template <int Bits> using precision_t = typename PrecisionOf<Bits>::type;

template <typename T> struct PrecisionBits;
template <> struct PrecisionBits<float> { static constexpr size_t value = 32; };
template <> struct PrecisionBits<double> { static constexpr size_t value = 64; };

template <typename T> struct Consts {
  static constexpr T ZERO = T(0);
  static constexpr T QUARTER = T(0.25);
  static constexpr T HALF = T(0.5);
  static constexpr T ONE = T(1);
  static constexpr T TWO = T(2);
  static constexpr T FOUR = T(4);
  static constexpr T CNSTNT = T(45);
  static constexpr T HARDNESS = T(38);
  static constexpr T NPNPDIST = T(5.5);
  static constexpr T NPPDIST = T(1);
  static constexpr T MAXVAL = std::numeric_limits<T>::max();
};

struct __attribute__((__packed__)) Atom {
  float x, y, z;
  int32_t type;
};

struct __attribute__((__packed__)) FFParams {
  int32_t hbtype;
  float radius;
  float hphb;
  float elsc;
};

template <typename T> struct alignas(16) AtomT {
  T x, y, z;
  int32_t type;
};

template <typename T> struct alignas(16) FFParamsT {
  int32_t hbtype;
  T radius;
  T hphb;
  T elsc;
};

template <typename T> struct Vec3 { T x, y, z; };
template <typename T> struct Vec4 { T x, y, z, w; };

template <typename T> [[nodiscard]] std::vector<AtomT<T>> convertAtoms(const std::vector<Atom> &xs) {
  std::vector<AtomT<T>> out(xs.size());
  for (size_t i = 0; i < xs.size(); ++i)
    out[i] = AtomT<T>{T(xs[i].x), T(xs[i].y), T(xs[i].z), xs[i].type};
  return out;
}

template <typename T> [[nodiscard]] std::vector<FFParamsT<T>> convertForcefield(const std::vector<FFParams> &xs) {
  std::vector<FFParamsT<T>> out(xs.size());
  for (size_t i = 0; i < xs.size(); ++i)
    out[i] = FFParamsT<T>{xs[i].hbtype, T(xs[i].radius), T(xs[i].hphb), T(xs[i].elsc)};
  return out;
}

template <typename T> [[nodiscard]] std::vector<T> convertPoses(const std::vector<float> &xs) {
  return std::vector<T>(xs.begin(), xs.end());
}

// ---- run interface ------------------------------------------------------------

struct Deck;

enum class Precision { Double, Float };

inline const char* precision_name(Precision p) { return p == Precision::Double ? "float64" : "float32"; }
inline std::size_t precision_bits(Precision p) { return p == Precision::Double ? 64 : 32; }

// PPWI values compiled into the binary (the original DEFAULT_PPWI).
constexpr std::array<std::size_t, 8> kPPWIs = {1, 2, 4, 8, 16, 32, 64, 128};

// One timed (ppwi, wgsize) configuration.
struct Sample
{
  std::size_t ppwi = 0, wgsize = 0;
  std::vector<double> energies;       // one per pose, widened to double
  std::vector<double> kernel_s;       // device-event seconds, timed iterations only
  std::vector<double> wall_s;         // host seconds around the same launches
  double context_s = 0.0;             // allocation and upload time
  EnergyResult energy;                // board energy over the timed iterations
};

// Runs `warmup` untimed and `iterations` timed launches of one configuration.
// The meter is started after the warm-up and stopped after the last timed
// launch. Throws std::runtime_error on runtime failures.
Sample run_config(Precision p, const Deck& deck, std::size_t ppwi, std::size_t wgsize, int device,
                  std::size_t warmup, std::size_t iterations, EnergyMeter& meter);

// One untimed fp64 launch of a configuration: the reference for float32 runs.
std::vector<double> reference_pass(const Deck& deck, std::size_t ppwi, std::size_t wgsize, int device);

// Empty when the compiled kernel matches the device, otherwise a note.
std::string kernel_arch_note(int device);

// Bytes of shared memory per block the device allows.
std::size_t shared_mem_per_block(int device);

// Build-time options compiled in.
bool shared_forcefield();          // USE_SHARED
const char* allocation_mode();     // device, managed or pagefault allocation
bool restrict_pointers();          // !NO_RESTRICT
