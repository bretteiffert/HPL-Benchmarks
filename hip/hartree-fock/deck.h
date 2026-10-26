// Hartree-Fock input file ("deck"): number of Gaussians, number of atoms,
// one exponent/coefficient pair per Gaussian, then x y z per atom (Angstrom).
// Values are kept in double exactly as read; the run converts them.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct Deck
{
  std::string path;
  std::string name;            // file name without directories, e.g. "he64"
  uint32_t ngauss = 0;
  uint32_t natoms = 0;
  std::vector<double> xpnt;    // ngauss exponents
  std::vector<double> coef;    // ngauss contraction coefficients
  std::vector<double> geom;    // 3 * natoms, laid out x[natoms], y[natoms], z[natoms]
};

// A problem with the input file. main prints "error: <what> (try '--help')".
struct DeckError : std::runtime_error
{
  explicit DeckError(const std::string& what) : std::runtime_error(what) {}
};

// Throws DeckError on a missing, truncated or malformed file.
Deck read_deck(const std::string& path);
