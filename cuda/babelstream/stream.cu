// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

// BabelStream on CUDA: device buffers, event-timed blocking launches, the
// partial-sum dot whose measured region includes the device-to-host copy of
// the partial sums (but not the final host sum), warm-up on separate scratch
// storage, and the CPU reference (scalar recurrence, compensated host sums,
// elementwise errors). Nothing here prints; failures throw.

#include "stream.h"
#include "kernels.cuh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(TBSIZE == kMaxBlockSize, "kMaxBlockSize must match TBSIZE in kernels.cuh");

int fmad_setting() { return FMAD; }

namespace {

void api_check(cudaError_t err, const char* what)
{
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(err));
}

void check_launch(const char* kernel)
{
  api_check(cudaGetLastError(), kernel);
}

// Host-side arithmetic in the run precision. For float and double this is
// plain T arithmetic, exactly as in the original check_solution. __half host
// arithmetic is not guaranteed by every toolkit, so each operation is done in
// float and rounded back to __half (round to nearest even).
template <class T>
struct HostOps
{
  static double to_double(T x) { return static_cast<double>(x); }
  static T from_double(double x) { return static_cast<T>(x); }
  static T add(T x, T y) { return x + y; }
  static T mul(T x, T y) { return x * y; }
  static double min_normal() { return static_cast<double>(std::numeric_limits<T>::min()); }
  static double max_finite() { return static_cast<double>(std::numeric_limits<T>::max()); }
};

template <>
struct HostOps<__half>
{
  static double to_double(__half x) { return static_cast<double>(__half2float(x)); }
  static __half from_double(double x) { return __float2half(static_cast<float>(x)); }
  static __half add(__half x, __half y) { return __float2half(__half2float(x) + __half2float(y)); }
  static __half mul(__half x, __half y) { return __float2half(__half2float(x) * __half2float(y)); }
  static double min_normal() { return 6.103515625e-05; }   // 2^-14
  static double max_finite() { return 65504.0; }
};

template <class T>
class CUDAStream : public Stream
{
  public:
    explicit CUDAStream(const StreamConfig& cfg);
    ~CUDAStream() override;

    double init() override;
    double copy() override;
    double mul() override;
    double add() override;
    double triad() override;
    double dot() override;
    double last_dot() const override { return last_dot_; }
    Correctness check(unsigned int ntimes) override;

  private:
    int array_size;
    int block_size;
    int dot_num_blocks;

    std::vector<T> sums;

    T *d_a = nullptr;
    T *d_b = nullptr;
    T *d_c = nullptr;
    T *d_sum = nullptr;

    cudaEvent_t start_ev = nullptr;
    cudaEvent_t stop_ev = nullptr;

    double last_dot_ = 0.0;

    void record_start();
    double record_stop(const char* routine);
    void warmup();
    void release();
};

template <class T>
CUDAStream<T>::CUDAStream(const StreamConfig& cfg)
  : array_size(cfg.array_size), block_size(cfg.block_size), dot_num_blocks(cfg.dot_groups),
    sums(static_cast<std::size_t>(cfg.dot_groups))
{
  const std::size_t array_bytes = sizeof(T) * static_cast<std::size_t>(array_size);
  try
  {
    api_check(cudaMalloc(&d_a, array_bytes), "cudaMalloc (a)");
    api_check(cudaMalloc(&d_b, array_bytes), "cudaMalloc (b)");
    api_check(cudaMalloc(&d_c, array_bytes), "cudaMalloc (c)");
    api_check(cudaMalloc(&d_sum, dot_num_blocks * sizeof(T)), "cudaMalloc (dot partial sums)");
    api_check(cudaEventCreate(&start_ev), "cudaEventCreate");
    api_check(cudaEventCreate(&stop_ev), "cudaEventCreate");
    warmup();
  }
  catch (...)
  {
    release();
    throw;
  }
}

template <class T>
CUDAStream<T>::~CUDAStream()
{
  release();
}

template <class T>
void CUDAStream<T>::release()
{
  // Best effort: never throw from cleanup.
  if (d_a) (void)cudaFree(d_a);
  if (d_b) (void)cudaFree(d_b);
  if (d_c) (void)cudaFree(d_c);
  if (d_sum) (void)cudaFree(d_sum);
  if (start_ev) (void)cudaEventDestroy(start_ev);
  if (stop_ev) (void)cudaEventDestroy(stop_ev);
  d_a = d_b = d_c = d_sum = nullptr;
  start_ev = stop_ev = nullptr;
}

template <class T>
void CUDAStream<T>::record_start()
{
  api_check(cudaEventRecord(start_ev), "cudaEventRecord");
}

template <class T>
double CUDAStream<T>::record_stop(const char* routine)
{
  api_check(cudaEventRecord(stop_ev), "cudaEventRecord");
  api_check(cudaEventSynchronize(stop_ev), "cudaEventSynchronize");

  float elapsed_ms = 0.0f;
  api_check(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev), "cudaEventElapsedTime");
  if (!(elapsed_ms > 0.0f))
    throw std::runtime_error(std::string("device timer returned no measurement for ") + routine);

  return static_cast<double>(elapsed_ms) * 1.0E-3;
}

template <class T>
double CUDAStream<T>::init()
{
  const T initA = HostOps<T>::from_double(startA);
  const T initB = HostOps<T>::from_double(startB);
  const T initC = HostOps<T>::from_double(startC);
  record_start();
  init_kernel<<<array_size/block_size, block_size>>>(d_a, d_b, d_c, initA, initB, initC);
  check_launch("init_kernel");
  return record_stop("Init");
}

template <class T>
double CUDAStream<T>::copy()
{
  record_start();
  copy_kernel<<<array_size/block_size, block_size>>>(d_a, d_c);
  check_launch("copy_kernel");
  return record_stop("Copy");
}

template <class T>
double CUDAStream<T>::mul()
{
  record_start();
  mul_kernel<<<array_size/block_size, block_size>>>(d_b, d_c);
  check_launch("mul_kernel");
  return record_stop("Mul");
}

template <class T>
double CUDAStream<T>::add()
{
  record_start();
  add_kernel<<<array_size/block_size, block_size>>>(d_a, d_b, d_c);
  check_launch("add_kernel");
  return record_stop("Add");
}

template <class T>
double CUDAStream<T>::triad()
{
  record_start();
  triad_kernel<<<array_size/block_size, block_size>>>(d_a, d_b, d_c);
  check_launch("triad_kernel");
  return record_stop("Triad");
}

template <class T>
double CUDAStream<T>::dot()
{
  record_start();
  dot_kernel<<<dot_num_blocks, block_size>>>(d_a, d_b, d_sum, array_size);
  check_launch("dot_kernel");

  api_check(cudaMemcpy(sums.data(), d_sum, dot_num_blocks*sizeof(T), cudaMemcpyDeviceToHost),
        "cudaMemcpy (dot partial sums)");

  const double seconds = record_stop("Dot");

  double sum = 0.0;
  for (int i = 0; i < dot_num_blocks; i++)
  {
    sum += HostOps<T>::to_double(sums[i]);
  }
  last_dot_ = sum;

  return seconds;
}

template <class T>
void CUDAStream<T>::warmup()
{
  const int warmup_size = block_size;

  T *w_a = nullptr, *w_b = nullptr, *w_c = nullptr, *w_sum = nullptr;
  auto free_all = [&]() {
    if (w_a) (void)cudaFree(w_a);
    if (w_b) (void)cudaFree(w_b);
    if (w_c) (void)cudaFree(w_c);
    if (w_sum) (void)cudaFree(w_sum);
  };

  try
  {
    api_check(cudaMalloc(&w_a, sizeof(T) * warmup_size), "cudaMalloc (warm-up)");
    api_check(cudaMalloc(&w_b, sizeof(T) * warmup_size), "cudaMalloc (warm-up)");
    api_check(cudaMalloc(&w_c, sizeof(T) * warmup_size), "cudaMalloc (warm-up)");
    api_check(cudaMalloc(&w_sum, sizeof(T) * dot_num_blocks), "cudaMalloc (warm-up)");

    const T initA = HostOps<T>::from_double(startA);
    const T initB = HostOps<T>::from_double(startB);
    const T initC = HostOps<T>::from_double(startC);

    init_kernel<<<1, block_size>>>(w_a, w_b, w_c, initA, initB, initC);
    check_launch("init_kernel (warm-up)");
    copy_kernel<<<1, block_size>>>(w_a, w_c);
    check_launch("copy_kernel (warm-up)");
    mul_kernel<<<1, block_size>>>(w_b, w_c);
    check_launch("mul_kernel (warm-up)");
    add_kernel<<<1, block_size>>>(w_a, w_b, w_c);
    check_launch("add_kernel (warm-up)");
    triad_kernel<<<1, block_size>>>(w_a, w_b, w_c);
    check_launch("triad_kernel (warm-up)");

    dot_kernel<<<dot_num_blocks, block_size>>>(w_a, w_b, w_sum, warmup_size);
    check_launch("dot_kernel (warm-up)");
    api_check(cudaMemcpy(sums.data(), w_sum, dot_num_blocks*sizeof(T), cudaMemcpyDeviceToHost),
          "cudaMemcpy (warm-up)");

    api_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize (warm-up)");
  }
  catch (...)
  {
    free_all();
    throw;
  }
  free_all();
}

template <class T>
Correctness CUDAStream<T>::check(unsigned int ntimes)
{
  using H = HostOps<T>;

  std::vector<T> a(static_cast<std::size_t>(array_size));
  std::vector<T> b(static_cast<std::size_t>(array_size));
  std::vector<T> c(static_cast<std::size_t>(array_size));
  api_check(cudaMemcpy(a.data(), d_a, a.size()*sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy (a)");
  api_check(cudaMemcpy(b.data(), d_b, b.size()*sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy (b)");
  api_check(cudaMemcpy(c.data(), d_c, c.size()*sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy (c)");

  // Scalar recurrence, as in the original check_solution.
  T goldA = H::from_double(startA);
  T goldB = H::from_double(startB);
  T goldC = H::from_double(startC);
  const T scalar = H::from_double(startScalar);

  for (unsigned int i = 0; i < ntimes; i++)
  {
    goldC = goldA;
    goldB = H::mul(scalar, goldC);
    goldC = H::add(goldA, goldB);
    goldA = H::add(goldB, H::mul(scalar, goldC));
  }

  auto kahan_sum = [](const std::vector<T>& v)
  {
    double sum = 0.0;
    double comp = 0.0;
    for (std::size_t i = 0; i < v.size(); i++)
    {
      const double y = H::to_double(v[i]) - comp;
      const double t = sum + y;
      comp = (t - sum) - y;
      sum = t;
    }
    return sum;
  };

  auto metrics = [&kahan_sum](const char* name, const std::vector<T>& v, const T gold)
  {
    CorrectnessRow row;
    row.name = name;
    const double gold_d = H::to_double(gold);
    row.expected = gold_d;

    const double actual_sum = kahan_sum(v);
    const double expected_sum = gold_d * static_cast<double>(v.size());
    row.checksum_rel_err = (expected_sum != 0.0)
      ? std::fabs(actual_sum - expected_sum) / std::fabs(expected_sum)
      : std::fabs(actual_sum);

    double max_err = 0.0;
    for (std::size_t i = 0; i < v.size(); i++)
    {
      const double abs_err = std::fabs(H::to_double(v[i]) - gold_d);
      const double rel_err = (gold_d != 0.0) ? abs_err / std::fabs(gold_d) : abs_err;
      if (rel_err > max_err) max_err = rel_err;
    }
    row.max_elem_rel_err = max_err;
    return row;
  };

  Correctness out;
  out.rows.push_back(metrics("a", a, goldA));
  out.rows.push_back(metrics("b", b, goldB));
  out.rows.push_back(metrics("c", c, goldC));

  const double gA = H::to_double(goldA);
  const double gB = H::to_double(goldB);
  const double gC = H::to_double(goldC);

  const double expected_dot = gA * gB * static_cast<double>(array_size);
  const double actual_dot = last_dot_;

  CorrectnessRow dot_row;
  dot_row.name = "dot";
  dot_row.expected = expected_dot;
  dot_row.checksum_rel_err = (expected_dot != 0.0)
    ? std::fabs(actual_dot - expected_dot) / std::fabs(expected_dot)
    : std::fabs(actual_dot);
  dot_row.has_max_elem = false;
  out.rows.push_back(dot_row);

  out.actual_dot = actual_dot;
  out.min_normal = H::min_normal();
  out.max_finite = H::max_finite();

  out.product_magnitude = std::fabs(gA * gB);
  out.dot_product_underflows = out.product_magnitude < out.min_normal;

  out.min_gold = std::fmin(std::fabs(gA), std::fmin(std::fabs(gB), std::fabs(gC)));
  out.gold_underflows = out.min_gold < out.min_normal;

  out.dot_overflows = std::fabs(expected_dot) > out.max_finite;

  return out;
}

} // namespace

std::unique_ptr<Stream> make_stream(Precision p, const StreamConfig& cfg)
{
  switch (p)
  {
    case Precision::Double: return std::unique_ptr<Stream>(new CUDAStream<double>(cfg));
    case Precision::Float:  return std::unique_ptr<Stream>(new CUDAStream<float>(cfg));
    case Precision::Half:   return std::unique_ptr<Stream>(new CUDAStream<__half>(cfg));
  }
  throw std::runtime_error("unknown precision");
}
