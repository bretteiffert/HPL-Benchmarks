// Hartree-Fock problem: device buffers, the untimed first Fock build, timed
// builds, and the fp64 CPU reference. Plain C++ (no vendor headers) so the
// host-only modules can include it. Nothing in here prints.
//
// The first build is untimed and fills the Fock matrix that the checksum and
// the correctness check use. Every timed build adds into the same Fock
// buffer, as in the original benchmark, so the matrix is not re-zeroed.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "deck.h"

enum class Precision { Double, Float };

inline const char* precision_name(Precision p)
{
  return p == Precision::Double ? "float64" : "float32";
}

enum class SqrtMode { Native, Legacy };

// "native" or "legacy-sqrtf" (the original SQRT_MODE strings)
inline const char* sqrt_mode_name(SqrtMode m)
{
  return m == SqrtMode::Native ? "native" : "legacy-sqrtf";
}

// Unique (i >= j) atom pairs.
inline uint64_t pair_count(uint32_t natoms)
{
  return (((uint64_t)natoms * natoms) + natoms) / 2;
}

// Unique (ij >= kl) pair quartets: one kernel thread each.
inline uint64_t quartet_count(uint32_t natoms)
{
  const uint64_t nn = pair_count(natoms);
  return ((nn * nn) + nn) / 2;
}

// Blocks for threads 0..nnnn (thread 0 exits at once), as in the original.
inline uint64_t grid_blocks(uint32_t natoms, uint32_t block_size)
{
  return ((quartet_count(natoms) + 1) + block_size - 1) / block_size;
}

// Fock matrix of the untimed first build compared with the fp64 CPU build.
// Numbers only; the report formats them.
struct FockCheck
{
  double e_gpu = 0.0;        // 2e- energy, device Fock summed in fp64
  double e_ref = 0.0;        // 2e- energy, fp64 CPU reference
  double rel_loss = 0.0;     // |e_gpu - e_ref| / |e_ref|: Fock build only
  double rel_inpipe = 0.0;   // |checksum - e_ref| / |e_ref|: including the run-precision sum
  double max_abs = 0.0;      // max |F - F_ref|
  double max_rel = 0.0;      // max |F - F_ref| / max |F_ref|
  double rms_abs = 0.0;      // rms |F - F_ref|
  double asym = 0.0;         // max |F(i,j) - F(j,i)| of the device Fock
  double e_symmetrized = 0.0;  // 2e- energy from F + F^T (diagnostic)
};

class HartreeFock
{
  public:
    virtual ~HartreeFock() {}

    // Untimed first build; blocks until done and keeps a host copy of Fock.
    virtual void first_build() = 0;

    // 2e- energy checksum of the first build, summed in the run precision
    // (the original "2e- energy=" line).
    virtual double checksum() const = 0;

    // One timed build (accumulates into Fock). Device-measured seconds.
    // Throws if the launch or the timer fails.
    virtual double timed_build() = 0;

    // fp64 CPU reference compared with the first build's Fock matrix.
    virtual FockCheck check() const = 0;
};

// Prepares host data (normalised coefficients, geometry in bohr, density,
// Schwarz screening in the run precision), allocates and copies device buffers.
std::unique_ptr<HartreeFock> make_hartree_fock(Precision p, SqrtMode m, const Deck& deck,
                                               uint32_t block_size);

// Whether the device can do atomicAdd on double (needed for --double).
bool fp64_atomic_add_supported(int device);
