// miniBUDE deck reading (same file layout and pose handling as the original
// parseParams), with errors instead of silent truncation or crashes.

#include "deck.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>

namespace {

std::string base_name(const std::string& path)
{
  std::string p = path;
  while (p.size() > 1 && p.back() == '/') p.pop_back();
  const std::size_t slash = p.find_last_of('/');
  return (slash == std::string::npos) ? p : p.substr(slash + 1);
}

template <typename T>
std::vector<T> read_n_struct(const std::string& path, const char* what)
{
  std::ifstream s(path, std::ios::binary);
  if (!s)
    throw DeckError("cannot read the " + std::string(what) + " file '" + path + "'");
  std::vector<char> bytes((std::istreambuf_iterator<char>(s)), std::istreambuf_iterator<char>());
  if (bytes.empty())
    throw DeckError("the " + std::string(what) + " file '" + path + "' is empty");
  if (bytes.size() % sizeof(T) != 0)
    throw DeckError("the " + std::string(what) + " file '" + path + "' has " + std::to_string(bytes.size()) +
                    " bytes, not a multiple of the " + std::to_string(sizeof(T)) + "-byte record");
  std::vector<T> xs(bytes.size() / sizeof(T));
  std::copy(bytes.begin(), bytes.end(), reinterpret_cast<char*>(xs.data()));
  return xs;
}

void check_types(const std::vector<Atom>& atoms, std::size_t ntypes, const std::string& path, const char* what)
{
  for (std::size_t i = 0; i < atoms.size(); i++)
    if (atoms[i].type < 0 || static_cast<std::size_t>(atoms[i].type) >= ntypes)
      throw DeckError(std::string(what) + " atom " + std::to_string(i) + " in '" + path + "' has type " +
                      std::to_string(atoms[i].type) + ", outside the forcefield's " + std::to_string(ntypes) +
                      " types");
}

} // namespace

Deck read_deck(const std::string& dir, std::size_t nposes)
{
  Deck d;
  d.dir = dir;
  d.name = base_name(dir);

  const std::string ligand_path = dir + "/ligand.in";
  const std::string protein_path = dir + "/protein.in";
  const std::string forcefield_path = dir + "/forcefield.in";
  const std::string poses_path = dir + "/poses.in";
  const std::string ref_path = dir + "/ref_energies.out";

  d.ligand = read_n_struct<Atom>(ligand_path, "ligand");
  d.protein = read_n_struct<Atom>(protein_path, "protein");
  d.forcefield = read_n_struct<FFParams>(forcefield_path, "forcefield");
  check_types(d.ligand, d.ntypes(), ligand_path, "ligand");
  check_types(d.protein, d.ntypes(), protein_path, "protein");

  const auto poses = read_n_struct<float>(poses_path, "poses");
  if (poses.size() % 6 != 0)
    throw DeckError("the poses file '" + poses_path + "' has " + std::to_string(poses.size()) +
                    " values, not a multiple of 6");
  d.maxPoses = poses.size() / 6;
  if (nposes == 0) nposes = d.maxPoses;
  if (nposes > d.maxPoses)
    throw DeckError("requested " + std::to_string(nposes) + " poses but the deck '" + dir + "' has only " +
                    std::to_string(d.maxPoses));

  for (std::size_t i = 0; i < 6; ++i)
  {
    d.poses[i].resize(nposes);
    std::copy(std::next(poses.cbegin(), static_cast<long>(i * d.maxPoses)),
              std::next(poses.cbegin(), static_cast<long>(i * d.maxPoses + nposes)), d.poses[i].begin());
  }

  std::ifstream input(ref_path);
  if (!input)
    throw DeckError("cannot read the reference energies file '" + ref_path + "'");
  std::string line;
  std::size_t lineno = 0;
  while (d.refEnergies.size() < nposes && std::getline(input, line))
  {
    lineno++;
    errno = 0;
    char* end = nullptr;
    const float v = std::strtof(line.c_str(), &end);
    if (end == line.c_str() || errno == ERANGE)
      throw DeckError("line " + std::to_string(lineno) + " of '" + ref_path + "' is not a number: '" + line + "'");
    d.refEnergies.push_back(v);
  }
  if (d.refEnergies.size() < nposes)
    throw DeckError("'" + ref_path + "' has " + std::to_string(d.refEnergies.size()) +
                    " reference energies, fewer than the " + std::to_string(nposes) + " poses");

  return d;
}
