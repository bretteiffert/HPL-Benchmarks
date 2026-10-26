// miniBUDE input deck: a directory with ligand.in, protein.in,
// forcefield.in, poses.in (binary, fp32) and ref_energies.out (text).

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "bude.h"

#define DATA_DIR "../../data/miniBUDE/bm1"

struct Deck
{
  std::string dir;
  std::string name;                       // last path component, e.g. "bm1"
  std::vector<Atom> protein;
  std::vector<Atom> ligand;
  std::vector<FFParams> forcefield;
  std::array<std::vector<float>, 6> poses;  // the first nposes() of each of the 6 columns
  std::vector<float> refEnergies;           // the first nposes() reference energies
  std::size_t maxPoses = 0;                 // poses in the file

  std::size_t natpro() const { return protein.size(); }
  std::size_t natlig() const { return ligand.size(); }
  std::size_t ntypes() const { return forcefield.size(); }
  std::size_t nposes() const { return poses[0].size(); }
};

// A problem with the deck. main prints "error: <what> (try '--help')".
struct DeckError : std::runtime_error
{
  explicit DeckError(const std::string& what) : std::runtime_error(what) {}
};

// Reads the deck and keeps the first `nposes` poses (0 = all). Throws
// DeckError for missing, empty, truncated or inconsistent files and for atom
// types outside the forcefield.
Deck read_deck(const std::string& dir, std::size_t nposes);
