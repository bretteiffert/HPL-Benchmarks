// Defines miniBUDE's fixed fp32 deck layout, precision-selected device data, typed constants, run parameters, and backend interface. Host conversions widen deck values before upload, while sampled energies are stored as doubles with their actual kernel precision recorded. Device-event and matching wall-clock samples are kept separately because these timings are not directly comparable with upstream host-clock miniBUDE results.

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define DEFAULT_ITERS 8
#define DEFAULT_ENERGY_ENTRIES 8

#define DATA_DIR "../../data/miniBUDE/bm1"
#define FILE_LIGAND "/ligand.in"
#define FILE_PROTEIN "/protein.in"
#define FILE_FORCEFIELD "/forcefield.in"
#define FILE_POSES "/poses.in"
#define FILE_REF_ENERGIES "/ref_energies.out"

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

struct Params {
  std::vector<Atom> protein;
  std::vector<Atom> ligand;
  std::vector<FFParams> forcefield;
  std::array<std::vector<float>, 6> poses;

  std::vector<float> refEnergies;
  size_t maxPoses, iterations, warmupIterations, outRows;
  std::string deckDir, output, deviceSelector;
  bool csv;
  bool list;

  [[nodiscard]] size_t totalIterations() const { return iterations + warmupIterations; }

  [[nodiscard]] size_t natpro() const { return protein.size(); }
  [[nodiscard]] size_t natlig() const { return ligand.size(); }
  [[nodiscard]] size_t ntypes() const { return forcefield.size(); }
  [[nodiscard]] size_t nposes() const { return poses[0].size(); }
};

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

using TimePoint = std::chrono::high_resolution_clock::time_point;

[[nodiscard]] static inline double elapsedMillis(const TimePoint &start, const TimePoint &end) {
  auto elapsedNs = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
  return elapsedNs * 1e-6;
}

[[nodiscard]] static inline TimePoint now() { return std::chrono::high_resolution_clock::now(); }

struct Sample {
  size_t ppwi, wgsize, precisionBits;
  std::vector<double> energies;

  std::vector<double> kernelMillis;

  std::vector<double> wallMillis;

  std::optional<std::pair<TimePoint, TimePoint>> contextTime;
  Sample(size_t ppwi, size_t wgsize, size_t precisionBits, size_t nposes)
      : ppwi(ppwi), wgsize(wgsize), precisionBits(precisionBits), energies(nposes), kernelMillis(), wallMillis(),
        contextTime() {}
};

using Device = std::pair<size_t, std::string>;

template <typename T, size_t PPWI> class Bude {
public:
  virtual ~Bude() = default;
  [[nodiscard]] virtual std::string name() = 0;
  [[nodiscard]] virtual std::vector<Device> enumerateDevices() = 0;
  [[nodiscard]] virtual bool compatible(const Params &p, size_t wgsize, size_t device) const { return true; };
  [[nodiscard]] virtual Sample fasten(const Params &p, size_t wgsize, size_t device) const = 0;
};
