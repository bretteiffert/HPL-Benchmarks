// Hartree-Fock input file parsing (same layout as the original read_file).

#include "deck.h"

#include <cmath>
#include <fstream>

namespace {

std::string base_name(const std::string& path)
{
  std::string p = path;
  while (p.size() > 1 && p.back() == '/') p.pop_back();
  const std::size_t slash = p.find_last_of('/');
  return (slash == std::string::npos) ? p : p.substr(slash + 1);
}

// Reads one non-negative integer count.
uint32_t read_count(std::ifstream& in, const std::string& path, const char* what)
{
  long long v = 0;
  if (!(in >> v))
    throw DeckError("input file '" + path + "': could not read the " + what);
  if (v <= 0 || v > 0xFFFFFFFFLL)
    throw DeckError("input file '" + path + "': the " + what + " must be a positive integer, got " +
                    std::to_string(v));
  return static_cast<uint32_t>(v);
}

double read_value(std::ifstream& in, const std::string& path, const std::string& what)
{
  double v = 0.0;
  if (!(in >> v))
    throw DeckError("input file '" + path + "': missing or malformed " + what);
  if (!std::isfinite(v))
    throw DeckError("input file '" + path + "': " + what + " is not a finite number");
  return v;
}

} // namespace

Deck read_deck(const std::string& path)
{
  std::ifstream in(path);
  if (!in)
    throw DeckError("cannot read input file '" + path + "'");

  Deck d;
  d.path = path;
  d.name = base_name(path);
  d.ngauss = read_count(in, path, "number of Gaussians");
  d.natoms = read_count(in, path, "number of atoms");

  d.xpnt.resize(d.ngauss);
  d.coef.resize(d.ngauss);
  for (uint32_t i = 0; i < d.ngauss; i++)
  {
    const std::string n = std::to_string(i + 1);
    d.xpnt[i] = read_value(in, path, "exponent of Gaussian " + n);
    d.coef[i] = read_value(in, path, "coefficient of Gaussian " + n);
    if (!(d.xpnt[i] > 0.0))
      throw DeckError("input file '" + path + "': exponent of Gaussian " + n + " must be positive");
  }

  d.geom.resize(3 * static_cast<std::size_t>(d.natoms));
  static const char* axis[3] = {"x", "y", "z"};
  for (uint32_t i = 0; i < d.natoms; i++)
    for (int j = 0; j < 3; j++)
      d.geom[j * static_cast<std::size_t>(d.natoms) + i] =
        read_value(in, path, std::string(axis[j]) + " coordinate of atom " + std::to_string(i + 1));

  return d;
}
