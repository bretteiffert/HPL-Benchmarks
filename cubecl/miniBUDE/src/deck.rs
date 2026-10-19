//! Input deck: the binary protein, ligand, forcefield and pose files plus the
//! text reference energies. Formats are unchanged from the original reader.

use std::fs;

#[derive(Clone, Copy)]
pub struct Atom {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub ty: i32,
}

#[derive(Clone, Copy)]
pub struct FFParams {
    pub hbtype: i32,
    pub radius: f32,
    pub hphb: f32,
    pub elsc: f32,
}

pub struct Deck {
    pub dir: String,
    pub protein: Vec<Atom>,
    pub ligand: Vec<Atom>,
    pub forcefield: Vec<FFParams>,
    /// Six pose components (three rotations, three translations), each `nposes` long.
    pub poses: [Vec<f32>; 6],
    /// Poses available in the deck, before any `--poses` limit.
    pub max_poses: usize,
    pub ref_energies: Vec<f32>,
}

impl Deck {
    pub fn natpro(&self) -> usize {
        self.protein.len()
    }
    pub fn natlig(&self) -> usize {
        self.ligand.len()
    }
    pub fn nposes(&self) -> usize {
        self.poses[0].len()
    }
}

fn le_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn le_i32(b: &[u8]) -> i32 {
    i32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn read_bytes(path: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|e| format!("could not read {}: {}", path, e))
}

fn read_atoms(path: &str) -> Result<Vec<Atom>, String> {
    Ok(read_bytes(path)?
        .chunks_exact(16)
        .map(|c| Atom {
            x: le_f32(&c[0..4]),
            y: le_f32(&c[4..8]),
            z: le_f32(&c[8..12]),
            ty: le_i32(&c[12..16]),
        })
        .collect())
}

fn read_ffparams(path: &str) -> Result<Vec<FFParams>, String> {
    Ok(read_bytes(path)?
        .chunks_exact(16)
        .map(|c| FFParams {
            hbtype: le_i32(&c[0..4]),
            radius: le_f32(&c[4..8]),
            hphb: le_f32(&c[8..12]),
            elsc: le_f32(&c[12..16]),
        })
        .collect())
}

fn read_f32s(path: &str) -> Result<Vec<f32>, String> {
    Ok(read_bytes(path)?.chunks_exact(4).map(le_f32).collect())
}

/// Reads the deck in `dir`, keeping the first `nposes` poses (0 = all).
pub fn read_deck(dir: &str, nposes: usize) -> Result<Deck, String> {
    let ligand = read_atoms(&format!("{}/ligand.in", dir))?;
    let protein = read_atoms(&format!("{}/protein.in", dir))?;
    let forcefield = read_ffparams(&format!("{}/forcefield.in", dir))?;

    let poses = read_f32s(&format!("{}/poses.in", dir))?;
    if poses.len() % 6 != 0 {
        return Err(format!(
            "{}/poses.in holds {} values, which is not divisible by 6",
            dir,
            poses.len()
        ));
    }
    let max_poses = poses.len() / 6;
    let n = if nposes == 0 { max_poses } else { nposes };
    if n > max_poses {
        return Err(format!(
            "requested {} poses, but the deck has only {}",
            n, max_poses
        ));
    }
    let pose = |k: usize| poses[k * max_poses..k * max_poses + n].to_vec();

    let ref_path = format!("{}/ref_energies.out", dir);
    let refs = fs::read_to_string(&ref_path)
        .map_err(|e| format!("could not read {}: {}", ref_path, e))?;
    let ref_energies = refs
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            l.trim()
                .parse::<f32>()
                .map_err(|e| format!("{} line {}: {}", ref_path, i + 1, e))
        })
        .collect::<Result<Vec<f32>, String>>()?;
    if n > ref_energies.len() {
        return Err(format!(
            "{} has {} reference energies, fewer than the {} poses requested",
            ref_path,
            ref_energies.len(),
            n
        ));
    }

    // Atom and forcefield type indices must stay inside the forcefield table,
    // since the kernel indexes it without bounds checks.
    let ntypes = forcefield.len();
    for (name, atoms) in [("protein", &protein), ("ligand", &ligand)] {
        if let Some(a) = atoms.iter().find(|a| a.ty < 0 || a.ty as usize >= ntypes) {
            return Err(format!(
                "{} atom type {} is outside the forcefield's {} entries",
                name, a.ty, ntypes
            ));
        }
    }

    Ok(Deck {
        dir: dir.to_string(),
        protein,
        ligand,
        forcefield,
        poses: [pose(0), pose(1), pose(2), pose(3), pose(4), pose(5)],
        max_poses,
        ref_energies,
    })
}
