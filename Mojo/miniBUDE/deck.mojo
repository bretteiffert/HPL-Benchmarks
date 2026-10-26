# miniBUDE input deck and golden-energy file.
#
# Binary layout as in the original (little-endian):
#   ligand.in / protein.in   16 bytes per atom: x, y, z (float32), type (int32)
#   forcefield.in            16 bytes per type: hbtype (int32), radius, hphb,
#                            elsc (float32)
#   poses.in                 6 float32 columns of max_poses values each
#   ref_energies.out         one energy per line (low precision)
# Changes from the original: errors name the file and the problem instead of
# crashing (missing files, truncated records, atom types outside the force
# field, non-numeric lines).

from std.collections import List
from std.os.path import basename, exists, join


struct Deck(Copyable):
    var dir: String
    var name: String
    var prot_x: List[Float32]
    var prot_y: List[Float32]
    var prot_z: List[Float32]
    var prot_t: List[Int32]
    var lig_x: List[Float32]
    var lig_y: List[Float32]
    var lig_z: List[Float32]
    var lig_t: List[Int32]
    var ff_hbtype: List[Int32]
    var ff_radius: List[Float32]
    var ff_hphb: List[Float32]
    var ff_elsc: List[Float32]
    var poses: List[List[Float32]]
    var max_poses: Int
    var ref_energies: List[Float32]

    def __init__(out self):
        self.dir = ""
        self.name = ""
        self.prot_x = List[Float32]()
        self.prot_y = List[Float32]()
        self.prot_z = List[Float32]()
        self.prot_t = List[Int32]()
        self.lig_x = List[Float32]()
        self.lig_y = List[Float32]()
        self.lig_z = List[Float32]()
        self.lig_t = List[Int32]()
        self.ff_hbtype = List[Int32]()
        self.ff_radius = List[Float32]()
        self.ff_hphb = List[Float32]()
        self.ff_elsc = List[Float32]()
        self.poses = List[List[Float32]]()
        self.max_poses = 0
        self.ref_energies = List[Float32]()

    def natpro(self) -> Int:
        return len(self.prot_x)

    def natlig(self) -> Int:
        return len(self.lig_x)

    def nposes(self) -> Int:
        return len(self.poses[0])

    def ntypes(self) -> Int:
        return len(self.ff_hbtype)


def le_f32(b: List[UInt8], off: Int) -> Float32:
    return b.unsafe_ptr().unsafe_offset(off).unsafe_bitcast[Float32]()[]


def le_i32(b: List[UInt8], off: Int) -> Int32:
    return b.unsafe_ptr().unsafe_offset(off).unsafe_bitcast[Int32]()[]


def _read_bytes(path: String) raises -> List[UInt8]:
    if not exists(path):
        raise Error("deck file not found: " + path)
    try:
        with open(path, "r") as f:
            return f.read_bytes()
    except e:
        raise Error("cannot read " + path + ": " + String(e))


def _read_text(path: String) raises -> String:
    if not exists(path):
        raise Error("file not found: " + path)
    try:
        with open(path, "r") as f:
            return f.read()
    except e:
        raise Error("cannot read " + path + ": " + String(e))


def _records(b: List[UInt8], path: String, what: String) raises -> Int:
    if len(b) == 0:
        raise Error(path + " is empty (no " + what + ")")
    if len(b) % 16 != 0:
        raise Error(
            path
            + " is "
            + String(len(b))
            + " bytes, not a whole number of 16-byte "
            + what
            + " records"
        )
    return len(b) // 16


def read_numbers(path: String) raises -> List[Float64]:
    """One number per line; blank lines ignored."""
    var text = _read_text(path)
    var out = List[Float64]()
    var line_no = 0
    for line in text.split("\n"):
        line_no += 1
        var t = String(String(line).strip())
        if t.byte_length() == 0:
            continue
        try:
            out.append(atof(t))
        except:
            raise Error(
                path + " line " + String(line_no) + ": '" + t + "' is not a number"
            )
    return out^


def load_deck(deck_dir: String, nposes_arg: Int) raises -> Deck:
    var d = Deck()
    d.dir = deck_dir
    d.name = basename(deck_dir)
    if d.name.byte_length() == 0:  # trailing slash
        d.name = basename(String(deck_dir.rstrip("/")))

    var lig_path = join(deck_dir, "ligand.in")
    var pro_path = join(deck_dir, "protein.in")
    var ff_path = join(deck_dir, "forcefield.in")
    var pos_path = join(deck_dir, "poses.in")
    var atoms_l = _read_bytes(lig_path)
    var atoms_p = _read_bytes(pro_path)
    var ffb = _read_bytes(ff_path)
    var posb = _read_bytes(pos_path)

    for i in range(_records(atoms_l, lig_path, "atom")):
        d.lig_x.append(le_f32(atoms_l, i * 16 + 0))
        d.lig_y.append(le_f32(atoms_l, i * 16 + 4))
        d.lig_z.append(le_f32(atoms_l, i * 16 + 8))
        d.lig_t.append(le_i32(atoms_l, i * 16 + 12))

    for i in range(_records(atoms_p, pro_path, "atom")):
        d.prot_x.append(le_f32(atoms_p, i * 16 + 0))
        d.prot_y.append(le_f32(atoms_p, i * 16 + 4))
        d.prot_z.append(le_f32(atoms_p, i * 16 + 8))
        d.prot_t.append(le_i32(atoms_p, i * 16 + 12))

    for i in range(_records(ffb, ff_path, "force-field")):
        d.ff_hbtype.append(le_i32(ffb, i * 16 + 0))
        d.ff_radius.append(le_f32(ffb, i * 16 + 4))
        d.ff_hphb.append(le_f32(ffb, i * 16 + 8))
        d.ff_elsc.append(le_f32(ffb, i * 16 + 12))

    # The kernel indexes the force field by atom type without a bounds check.
    var ntypes = d.ntypes()
    for i in range(d.natlig()):
        var t = Int(d.lig_t[i])
        if t < 0 or t >= ntypes:
            raise Error(
                lig_path + ": atom " + String(i) + " has type " + String(t)
                + ", outside the " + String(ntypes) + " force-field types"
            )
    for i in range(d.natpro()):
        var t = Int(d.prot_t[i])
        if t < 0 or t >= ntypes:
            raise Error(
                pro_path + ": atom " + String(i) + " has type " + String(t)
                + ", outside the " + String(ntypes) + " force-field types"
            )

    if len(posb) % 4 != 0:
        raise Error(pos_path + " is not a whole number of float32 values")
    var total = len(posb) // 4
    if total == 0 or total % 6 != 0:
        raise Error(
            pos_path + " holds " + String(total)
            + " values, not a positive multiple of 6"
        )
    var max_poses = total // 6
    d.max_poses = max_poses
    var n = max_poses if nposes_arg == 0 else nposes_arg
    if n > max_poses:
        raise Error(
            "requested poses ("
            + String(n)
            + ") exceed the deck's "
            + String(max_poses)
            + " poses"
        )
    for k in range(6):
        var col = List[Float32]()
        for i in range(n):
            col.append(le_f32(posb, (k * max_poses + i) * 4))
        d.poses.append(col^)

    var ref_path = join(deck_dir, "ref_energies.out")
    var refs = read_numbers(ref_path)
    for i in range(len(refs)):
        d.ref_energies.append(Float32(refs[i]))
    if n > len(d.ref_energies):
        raise Error(
            ref_path
            + " has "
            + String(len(d.ref_energies))
            + " energies, fewer than the "
            + String(n)
            + " poses"
        )
    return d^
