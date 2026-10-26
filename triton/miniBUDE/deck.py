"""miniBUDE input deck: ligand.in, protein.in, forcefield.in, poses.in (binary,
little-endian) and ref_energies.out (text). Errors are DeckError, never a crash."""

import os

import numpy as np

DISK_ATOM = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("type", "<i4")]
)
DISK_FFPARAMS = np.dtype(
    [("hbtype", "<i4"), ("radius", "<f4"), ("hphb", "<f4"), ("elsc", "<f4")]
)


class DeckError(Exception):
    pass


class Deck:
    def __init__(self):
        self.dir = ""
        self.name = ""
        self.protein = None
        self.ligand = None
        self.forcefield = None
        self.poses = [None] * 6
        self.ref_energies = np.zeros(0, dtype=np.float32)
        self.max_poses = 0

    @property
    def natpro(self):
        return len(self.protein)

    @property
    def natlig(self):
        return len(self.ligand)

    @property
    def ntypes(self):
        return len(self.forcefield)

    @property
    def nposes(self):
        return len(self.poses[0])


def _read_records(deck_dir, name, dtype):
    path = os.path.join(deck_dir, name)
    if not os.path.isfile(path):
        raise DeckError("deck '%s' has no %s" % (deck_dir, name))
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) % dtype.itemsize:
        raise DeckError("%s is %d bytes, not a whole number of %d-byte records"
                        % (path, len(raw), dtype.itemsize))
    if not raw:
        raise DeckError("%s is empty" % path)
    return np.frombuffer(raw, dtype=dtype).copy()


def load(deck_dir, nposes_arg):
    """nposes_arg 0 means every pose in the deck."""
    d = Deck()
    d.dir = deck_dir
    d.name = os.path.basename(os.path.normpath(deck_dir))
    if not os.path.isdir(deck_dir):
        raise DeckError("deck directory '%s' not found" % deck_dir)

    d.ligand = _read_records(deck_dir, "ligand.in", DISK_ATOM)
    d.protein = _read_records(deck_dir, "protein.in", DISK_ATOM)
    d.forcefield = _read_records(deck_dir, "forcefield.in", DISK_FFPARAMS)

    # Type indices are used directly as forcefield offsets in the kernel.
    for label, atoms in (("ligand", d.ligand), ("protein", d.protein)):
        bad = np.nonzero((atoms["type"] < 0) | (atoms["type"] >= d.ntypes))[0]
        if bad.size:
            i = int(bad[0])
            raise DeckError("%s atom %d has type %d, but forcefield.in defines types 0-%d"
                            % (label, i, int(atoms["type"][i]), d.ntypes - 1))

    poses = _read_records(deck_dir, "poses.in", np.dtype("<f4"))
    if len(poses) % 6:
        raise DeckError("poses.in holds %d floats, not a multiple of 6" % len(poses))
    d.max_poses = len(poses) // 6
    n = d.max_poses if nposes_arg == 0 else nposes_arg
    if n > d.max_poses:
        raise DeckError("requested %d poses, but the deck has %d" % (n, d.max_poses))
    for k in range(6):
        d.poses[k] = poses[k * d.max_poses: k * d.max_poses + n].copy()

    path = os.path.join(deck_dir, "ref_energies.out")
    if not os.path.isfile(path):
        raise DeckError("deck '%s' has no ref_energies.out" % deck_dir)
    refs = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                refs.append(float(s))
            except ValueError:
                raise DeckError("%s line %d: expected a number, got '%s'" % (path, lineno, s))
    d.ref_energies = np.array(refs, dtype=np.float32)
    if d.nposes > len(d.ref_energies):
        raise DeckError("ref_energies.out has %d energies, fewer than the %d poses"
                        % (len(d.ref_energies), d.nposes))
    return d
