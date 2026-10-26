// Seven-point stencil problem: device buffers, the quadratic test field, the
// untimed first Laplacian, timed Laplacians, and the correctness check
// against the exact interior value 3. Plain C++ (no vendor headers) so the
// host-only modules can include it. Nothing in here prints.

#pragma once

#include <cstddef>
#include <memory>

enum class Precision { Double, Float, Half };

inline const char* precision_name(Precision p)
{
  switch (p)
  {
    case Precision::Double: return "float64";
    case Precision::Float:  return "float32";
    case Precision::Half:   return "float16";
  }
  return "unknown";
}

inline std::size_t precision_bytes(Precision p)
{
  switch (p)
  {
    case Precision::Double: return 8;
    case Precision::Float:  return 4;
    case Precision::Half:   return 2;
  }
  return 0;
}

struct Dims
{
  std::size_t x = 0, y = 0, z = 0;
  std::size_t count() const { return x * y * z; }
};

// Theoretical traffic of one Laplacian, as in the original: every point is
// read except the 8 corners and 12 edges, and every interior point is written.
inline std::size_t fetch_bytes(const Dims& n, std::size_t elem)
{
  return (n.x * n.y * n.z - 8 - 4 * (n.x - 2) - 4 * (n.y - 2) - 4 * (n.z - 2)) * elem;
}
inline std::size_t write_bytes(const Dims& n, std::size_t elem)
{
  return ((n.x - 2) * (n.y - 2) * (n.z - 2)) * elem;
}

// Launch grid of the Laplacian for a block shape (the original laplacian()).
inline Dims launch_grid(const Dims& n, const Dims& b)
{
  return {(n.x - 1) / b.x + 1, (n.y - 1) / b.y + 1, (n.z - 1) / b.z + 1};
}

struct Correctness
{
  double max_abs_err = 0.0;
  double l2_rel_err = 0.0;
  double mean_abs_err = 0.0;
  std::size_t num_checked = 0;
};

class Stencil
{
  public:
    virtual ~Stencil() {}

    // Untimed: fills u with the test field and runs one Laplacian.
    virtual void first_run() = 0;

    // One timed Laplacian; device-measured seconds. Throws on failure.
    virtual double timed_run() = 0;

    // Interior of f against the exact value 3, read back one z-slice at a
    // time (the grid is never copied to the host whole).
    virtual Correctness check() const = 0;
};

std::unique_ptr<Stencil> make_stencil(Precision p, const Dims& n, const Dims& block);

// Device limits the validation needs beyond backend.h.
struct LaunchLimits
{
  Dims max_block;   // maxThreadsDim
  Dims max_grid;    // maxGridSize
};
LaunchLimits launch_limits(int device);

// Threads per block of the fixed test-field launch (TBSize in the original).
constexpr std::size_t kSetupBlock = 1024;
