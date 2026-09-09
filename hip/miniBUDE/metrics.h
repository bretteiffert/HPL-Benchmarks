// Provides informational correctness metrics without a pass/fail gate for the HIP miniBUDE driver. It computes compensated L2, RMS, signed, maximum, non-finite, and tolerance counts, plus ULP distances after rounding both operands to the tested precision. Deck references are used for norm comparisons but not meaningful ULP analysis because their text representation has limited precision.

#include "hip/hip_runtime.h"
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "bude.h"

namespace ulp {

[[nodiscard]] inline int64_t ordered(float f) {
  int32_t i;
  std::memcpy(&i, &f, sizeof(i));
  return i < 0 ? int64_t(0x80000000LL - int64_t(uint32_t(i))) : int64_t(i);
}

[[nodiscard]] inline int64_t ordered(double d) {
  int64_t i;
  std::memcpy(&i, &d, sizeof(i));
  return i < 0 ? (INT64_MIN - i) : i;
}

template <typename T> [[nodiscard]] inline int64_t distance(T a, T b) {
  if (std::isnan(a) || std::isnan(b)) return INT64_MAX;
  if (a == b) return 0;
  if (std::isinf(a) || std::isinf(b)) return INT64_MAX;
  const auto d = ordered(a) - ordered(b);
  return d < 0 ? -d : d;
}

}

static constexpr std::array<int64_t, 9> ULP_BUCKETS = {1, 2, 4, 8, 16, 64, 256, 1024, 4096};
static constexpr size_t ULP_NBUCKETS = ULP_BUCKETS.size() + 1;

[[nodiscard]] inline std::string ulpBucketLabel(size_t i) {
  if (i == 0) return "0";
  if (i == ULP_NBUCKETS - 1) return ">=" + std::to_string(ULP_BUCKETS.back());
  const auto lo = ULP_BUCKETS[i - 1];
  const auto hi = ULP_BUCKETS[i] - 1;
  return lo == hi ? std::to_string(lo) : std::to_string(lo) + "-" + std::to_string(hi);
}

struct Metrics {
  double l2RelNorm = 0.0;
  double rmsAbs = 0.0;
  double meanSigned = 0.0;

  double maxAbs = 0.0;
  size_t maxAbsIdx = 0;
  double maxRel = 0.0;
  size_t maxRelIdx = 0;

  int64_t maxUlp = 0;
  size_t maxUlpIdx = 0;
  std::array<size_t, ULP_NBUCKETS> ulpHistogram{};

  size_t compared = 0;
  size_t nonFinite = 0;
  size_t beyondTolerance = 0;

  [[nodiscard]] bool empty() const { return compared == 0; }
};

namespace detail {
struct Kahan {
  double sum = 0.0, c = 0.0;
  void add(double x) {
    const double y = x - c;
    const double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
};
}

[[nodiscard]] inline Metrics computeMetrics(const std::vector<double> &measured,
                                             const std::vector<double> &reference,
                                            size_t precisionBits) {
  Metrics m;
  const size_t n = std::min(measured.size(), reference.size());

  detail::Kahan errSq, refSq, signedErr;

  for (size_t i = 0; i < n; ++i) {
    const double x = measured[i];
    const double r = reference[i];

    if (!std::isfinite(x) || !std::isfinite(r)) {
      m.nonFinite++;
      continue;
    }

    const double err = x - r;
    const double absErr = std::fabs(err);

    errSq.add(err * err);
    refSq.add(r * r);
    signedErr.add(err);

    if (absErr > m.maxAbs) {
      m.maxAbs = absErr;
      m.maxAbsIdx = i;
    }

    const double denom = std::fabs(r);
    if (denom > 0.0) {
      const double rel = absErr / denom;
      if (rel > m.maxRel) {
        m.maxRel = rel;
        m.maxRelIdx = i;
      }
      if (rel * 100.0 > REPORT_TOLERANCE_PCT) m.beyondTolerance++;
    }

    const int64_t u = precisionBits == 32 ? ulp::distance(float(x), float(r)) : ulp::distance(x, r);
    if (u > m.maxUlp) {
      m.maxUlp = u;
      m.maxUlpIdx = i;
    }
    size_t bucket = ULP_NBUCKETS - 1;
    for (size_t b = 0; b < ULP_BUCKETS.size(); ++b) {
      if (u < ULP_BUCKETS[b]) {
        bucket = b;
        break;
      }
    }
    m.ulpHistogram[bucket]++;

    m.compared++;
  }

  if (m.compared > 0) {
    m.rmsAbs = std::sqrt(errSq.sum / double(m.compared));
    m.meanSigned = signedErr.sum / double(m.compared);
    m.l2RelNorm = refSq.sum > 0.0 ? std::sqrt(errSq.sum) / std::sqrt(refSq.sum) : 0.0;
  }

  return m;
}

[[nodiscard]] inline Metrics compareToDeck(const std::vector<double> &measured,
                                           const std::vector<float> &refEnergies) {
  std::vector<double> ref(refEnergies.begin(),
                          refEnergies.begin() + int64_t(std::min(refEnergies.size(), measured.size())));
  return computeMetrics(measured, ref, 64);
}
