// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

// BabelStream problem: device buffers, timed launches and the CPU reference.
// This header is plain C++ (no vendor headers) so the host-only modules can
// include it. Every launch is timed with device events and blocks until the
// kernel has finished. Nothing in here prints.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#define startA (0.1)
#define startB (0.2)
#define startC (0.0)
#define startScalar (0.4)

enum class Precision { Double, Float, Half };

// "float64", "float32", "float16"
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

// Bytes per element.
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

struct StreamConfig
{
  int array_size = 0;
  int block_size = 0;
  int dot_groups = 0;
};

struct CorrectnessRow
{
  std::string name;          // a, b, c, dot
  double expected = 0.0;
  double checksum_rel_err = 0.0;
  bool has_max_elem = true;  // false for dot
  double max_elem_rel_err = 0.0;
};

// Numbers only; the report turns the flags into note lines.
struct Correctness
{
  std::vector<CorrectnessRow> rows;   // a, b, c, dot
  double actual_dot = 0.0;

  double min_normal = 0.0;            // smallest normal value of the run precision
  double max_finite = 0.0;            // largest finite value of the run precision

  // Smallest |gold| among a, b, c is below min_normal: the arrays have decayed
  // to subnormals or zero and their relative errors are not meaningful.
  bool gold_underflows = false;
  double min_gold = 0.0;

  // |gold a * gold b| is below min_normal: the dot relative error is not meaningful.
  bool dot_product_underflows = false;
  double product_magnitude = 0.0;

  // |expected dot| exceeds max_finite: the device dot overflows.
  bool dot_overflows = false;
};

class Stream
{
  public:
    virtual ~Stream() {}

    // Each call launches one kernel and returns its device-measured duration
    // in seconds. Throws std::runtime_error if the launch or the timer fails.
    virtual double init() = 0;
    virtual double copy() = 0;
    virtual double mul() = 0;
    virtual double add() = 0;
    virtual double triad() = 0;
    virtual double dot() = 0;   // includes copying the partial sums to the host

    // Result of the most recent dot(), summed on the host in the run precision.
    virtual double last_dot() const = 0;

    // Read the arrays back and compare them with the scalar recurrence
    // replayed `ntimes` times on the host.
    virtual Correctness check(unsigned int ntimes) = 0;
};

// Allocates the buffers and warms every kernel on separate scratch storage.
// The config must already be validated (see main.cpp).
std::unique_ptr<Stream> make_stream(Precision p, const StreamConfig& cfg);

// Shared-memory slots in the Dot reduction (TBSIZE in kernels.cuh): the
// largest block size the reduction can use.
constexpr int kMaxBlockSize = 256;

// Value of the FMAD build setting compiled into the kernels.
int fmad_setting();
