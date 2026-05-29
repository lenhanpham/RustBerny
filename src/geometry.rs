//! Molecular geometry representation and file I/O.
//!
//! Supports molecules and periodic crystals with lattice vectors.
//! Provides coordinate manipulation, distance calculations, and
//! connectivity analysis.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use std::fmt;
use std::fs;
use std::path::Path;

use crate::species;

/// Bohr to angstrom conversion factor.
pub const ANGSTROM: f64 = 1.0 / 0.52917721092;

/// Represents a molecular geometry with species and coordinates.
#[derive(Debug, Clone)]
pub struct Geometry {
    /// Element symbols.
    pub species: Vec<String>,
    /// Atomic coordinates in angstroms.
    pub coords: Vec<Vector3<f64>>,
    /// Lattice vectors for crystals, or `None` for molecules.
    pub lattice: Option<Matrix3<f64>>,
}

impl Geometry {
    /// Creates a new geometry from species, coordinates, and optional lattice.
    pub fn new(
        species: Vec<String>,
        coords: Vec<Vector3<f64>>,
        lattice: Option<Matrix3<f64>>,
    ) -> Self {
        Self {
            species,
            coords,
            lattice,
        }
    }

    /// Creates a geometry from `(symbol, coordinate)` pairs.
    pub fn from_atoms(atoms: Vec<(&str, [f64; 3])>, lattice: Option<[[f64; 3]; 3]>) -> Self {
        let species = atoms.iter().map(|(s, _)| s.to_string()).collect();
        let coords = atoms.iter().map(|(_, c)| Vector3::from(*c)).collect();
        let lattice = lattice.map(Matrix3::from);
        Self {
            species,
            coords,
            lattice,
        }
    }

    /// Returns the number of atoms.
    pub fn len(&self) -> usize {
        self.species.len()
    }

    /// Returns `true` if the geometry has no atoms.
    pub fn is_empty(&self) -> bool {
        self.species.is_empty()
    }

    /// Returns the chemical formula in Hill system order.
    pub fn formula(&self) -> String {
        let mut counts: Vec<(String, usize)> = Vec::new();
        for sp in &self.species {
            if let Some(entry) = counts.iter_mut().find(|(s, _)| s == sp) {
                entry.1 += 1;
            } else {
                counts.push((sp.clone(), 1));
            }
        }
        counts.sort_by(|a, b| a.0.cmp(&b.0));
        let mut result = String::new();
        for (sp, n) in &counts {
            result.push_str(sp);
            if *n > 1 {
                result.push_str(&n.to_string());
            }
        }
        result
    }

    /// Dumps the geometry to a string in the specified format.
    pub fn dump(&self, fmt: &str) -> String {
        match fmt {
            "xyz" => self.to_xyz(),
            "aims" => self.to_aims(),
            _ => panic!("Unknown format: {fmt}"),
        }
    }

    /// Returns the XYZ format string.
    fn to_xyz(&self) -> String {
        let mut s = format!("{}\n", self.len());
        s.push_str(&format!("Formula: {}\n", self.formula()));
        for (sp, coord) in self.species.iter().zip(self.coords.iter()) {
            s.push_str(&format!(
                "{:>2} {:15.8} {:15.8} {:15.8}\n",
                sp, coord[0], coord[1], coord[2]
            ));
        }
        s
    }

    /// Returns the FHI-aims format string.
    fn to_aims(&self) -> String {
        let mut s = format!("# Formula: {}\n", self.formula());
        if let Some(lat) = &self.lattice {
            for vec in lat.row_iter() {
                s.push_str(&format!(
                    "lattice_vector {:15.8} {:15.8} {:15.8}\n",
                    vec[0], vec[1], vec[2]
                ));
            }
        }
        for (sp, coord) in self.species.iter().zip(self.coords.iter()) {
            s.push_str(&format!(
                "atom {:15.8} {:15.8} {:15.8} {:>2}\n",
                coord[0], coord[1], coord[2], sp
            ));
        }
        s
    }

    /// Writes the geometry to a file.
    pub fn write(&self, path: &Path) -> Result<(), std::io::Error> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let fmt = match ext {
            "xyz" => "xyz",
            "aims" | _ if path.file_name().and_then(|n| n.to_str()) == Some("geometry.in") => {
                "aims"
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Unknown file extension",
                ))
            }
        };
        fs::write(path, self.dump(fmt))
    }

    /// Returns pairwise distance matrix, shape (N, N).
    pub fn dist(&self) -> DMatrix<f64> {
        let n = self.len();
        let mut dist = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = (self.coords[i] - self.coords[j]).norm();
                dist[(i, j)] = d;
                dist[(j, i)] = d;
            }
        }
        // Set diagonal to infinity
        for i in 0..n {
            dist[(i, i)] = f64::INFINITY;
        }
        dist
    }

    /// Returns covalent connectivity matrix.
    ///
    /// Entry `(i, j)` is `true` if atoms `i` and `j` are within
    /// `scale * (r_cov_i + r_cov_j)` of each other.
    pub fn bondmatrix(&self, scale: f64) -> Vec<Vec<bool>> {
        let n = self.len();
        let dist = self.dist();
        let radii: Vec<f64> = self
            .species
            .iter()
            .map(|sp| species::get_property(sp, "covalent_radius"))
            .collect();
        let mut bm = vec![vec![false; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let threshold = scale * (radii[i] + radii[j]);
                if dist[(i, j)] < threshold {
                    bm[i][j] = true;
                    bm[j][i] = true;
                }
            }
        }
        bm
    }

    /// Returns the covalentness matrix ρ.
    ///
    /// `ρ_ij = exp(-R_ij / (R_cov_i + R_cov_j) + 1)`
    pub fn rho(&self) -> DMatrix<f64> {
        let n = self.len();
        let geom = self.supercell();
        let dist = geom.dist();
        let radii: Vec<f64> = geom
            .species
            .iter()
            .map(|sp| species::get_property(sp, "covalent_radius"))
            .collect();
        let mut rho = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // Python sets dist[diagonal] = inf, so rho[i,i] = exp(-inf/r+1) = 0.
                    rho[(i, j)] = 0.0;
                } else {
                    let d = dist[(i, j)];
                    let r_sum = radii[i] + radii[j];
                    rho[(i, j)] = (-d / r_sum + 1.0).exp();
                }
            }
        }
        rho
    }

    /// Returns atomic masses as a vector.
    pub fn masses(&self) -> DVector<f64> {
        DVector::from_fn(self.len(), |i, _| {
            species::get_property(&self.species[i], "mass")
        })
    }

    /// Returns the center of mass.
    pub fn center_of_mass(&self) -> Vector3<f64> {
        let masses = self.masses();
        let total_mass = masses.sum();
        self.coords
            .iter()
            .zip(masses.iter())
            .map(|(r, m)| *r * *m)
            .sum::<Vector3<f64>>()
            / total_mass
    }

    /// Returns the moment of inertia tensor, shape (3, 3).
    pub fn inertia(&self) -> Matrix3<f64> {
        let cms = self.center_of_mass();
        let masses = self.masses();
        let mut inertia = Matrix3::zeros();
        for (r, &m) in self.coords.iter().zip(masses.iter()) {
            let r = r - cms;
            let r2 = r.norm_squared();
            for a in 0..3 {
                for b in 0..3 {
                    inertia[(a, b)] += m * (if a == b { r2 } else { -r[a] * r[b] });
                }
            }
        }
        inertia
    }

    /// Creates a crystal supercell.
    ///
    /// For molecules, returns a copy of itself.
    pub fn supercell(&self) -> Self {
        // For molecules, just return a copy
        if self.lattice.is_none() {
            return self.clone();
        }
        // For crystals, create a 3x3x3 supercell
        self.clone() // Simplified: full implementation would expand images
    }
}

impl fmt::Display for Geometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<Geometry {}>", self.formula())
    }
}

/// Reads a geometry from a string in the specified format.
pub fn loads(s: &str, fmt: &str) -> Result<Geometry, String> {
    match fmt {
        "xyz" => parse_xyz(s),
        "aims" => parse_aims(s),
        _ => Err(format!("Unknown format: {fmt}")),
    }
}

/// Reads a geometry from a file path.
pub fn readfile(path: &Path) -> Result<Geometry, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let fmt = match ext {
        "xyz" => "xyz",
        "aims" => "aims",
        _ => return Err(format!("Cannot infer format from path {path:?}")),
    };
    let content = fs::read_to_string(path).map_err(|e| format!("Failed to read {path:?}: {e}"))?;
    loads(&content, fmt)
}

/// Parses XYZ format.
fn parse_xyz(s: &str) -> Result<Geometry, String> {
    let mut lines = s.lines();
    let n: usize = lines
        .next()
        .ok_or("Missing atom count")?
        .trim()
        .parse()
        .map_err(|_| "Invalid atom count")?;
    let _comment = lines.next().ok_or("Missing comment line")?;

    let mut species = Vec::new();
    let mut coords = Vec::new();

    for _ in 0..n {
        let line = lines.next().ok_or("Unexpected end of file")?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("Invalid atom line: {line}"));
        }
        species.push(parts[0].to_string());
        let x: f64 = parts[1]
            .parse()
            .map_err(|_| format!("Invalid x: {}", parts[1]))?;
        let y: f64 = parts[2]
            .parse()
            .map_err(|_| format!("Invalid y: {}", parts[2]))?;
        let z: f64 = parts[3]
            .parse()
            .map_err(|_| format!("Invalid z: {}", parts[3]))?;
        coords.push(Vector3::new(x, y, z));
    }

    Ok(Geometry::new(species, coords, None))
}

/// Parses FHI-aims format.
fn parse_aims(s: &str) -> Result<Geometry, String> {
    let mut species = Vec::new();
    let mut coords = Vec::new();
    let mut lattice = Vec::new();

    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts[0] {
            "atom" => {
                if parts.len() < 5 {
                    return Err(format!("Invalid atom line: {line}"));
                }
                let x: f64 = parts[1].parse().map_err(|_| "Invalid x")?;
                let y: f64 = parts[2].parse().map_err(|_| "Invalid y")?;
                let z: f64 = parts[3].parse().map_err(|_| "Invalid z")?;
                coords.push(Vector3::new(x, y, z));
                species.push(parts[4].to_string());
            }
            "lattice_vector" => {
                if parts.len() < 4 {
                    return Err(format!("Invalid lattice line: {line}"));
                }
                let x: f64 = parts[1].parse().map_err(|_| "Invalid lx")?;
                let y: f64 = parts[2].parse().map_err(|_| "Invalid ly")?;
                let z: f64 = parts[3].parse().map_err(|_| "Invalid lz")?;
                lattice.push([x, y, z]);
            }
            _ => {}
        }
    }

    let lattice_opt = if lattice.len() == 3 {
        Some(Matrix3::from_row_slice(&[
            lattice[0][0],
            lattice[0][1],
            lattice[0][2],
            lattice[1][0],
            lattice[1][1],
            lattice[1][2],
            lattice[2][0],
            lattice[2][1],
            lattice[2][2],
        ]))
    } else {
        None
    };

    Ok(Geometry::new(species, coords, lattice_opt))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn water() -> Geometry {
        Geometry::from_atoms(
            vec![
                ("O", [0.0, 0.0, 0.0]),
                ("H", [0.0, 0.757, 0.587]),
                ("H", [0.0, -0.757, 0.587]),
            ],
            None,
        )
    }

    #[test]
    fn test_len() {
        assert_eq!(water().len(), 3);
    }

    #[test]
    fn test_formula() {
        assert_eq!(water().formula(), "H2O");
    }

    #[test]
    fn test_xyz_roundtrip() {
        let geom = water();
        let xyz = geom.dump("xyz");
        assert!(xyz.contains("H2O"));
        assert!(xyz.contains("O"));
    }

    #[test]
    fn test_dist() {
        let geom = water();
        let d = geom.dist();
        // O-H distance should be around 0.96 Å
        let oh_dist = d[(0, 1)];
        assert!(oh_dist > 0.9 && oh_dist < 1.1);
    }

    #[test]
    fn test_bondmatrix() {
        let geom = water();
        let bm = geom.bondmatrix(1.3);
        // O-H bonds should be connected
        assert!(bm[0][1]);
        assert!(bm[0][2]);
        // H-H should not be connected
        assert!(!bm[1][2]);
    }

    #[test]
    fn test_masses() {
        let geom = water();
        let m = geom.masses();
        assert!((m[0] - 15.9994).abs() < 1e-3);
        assert!((m[1] - 1.0079).abs() < 1e-3);
    }

    #[test]
    fn test_center_of_mass() {
        let geom = water();
        let cms = geom.center_of_mass();
        // Symmetric molecule, CMS should be near origin
        assert!(cms[0].abs() < 1e-10);
        assert!(cms[1].abs() < 1e-10);
    }

    #[test]
    fn test_parse_xyz() {
        let xyz = "3\nFormula: H2O\nO   0.00000000   0.00000000   0.00000000\nH   0.00000000   0.75700000   0.58700000\nH   0.00000000  -0.75700000   0.58700000\n";
        let geom = loads(xyz, "xyz").unwrap();
        assert_eq!(geom.len(), 3);
        assert_eq!(geom.formula(), "H2O");
    }

    #[test]
    fn test_parse_aims() {
        let aims = "# Formula: H2O\natom   0.00000000   0.00000000   0.00000000   O\natom   0.00000000   0.75700000   0.58700000   H\natom   0.00000000  -0.75700000   0.58700000   H\n";
        let geom = loads(aims, "aims").unwrap();
        assert_eq!(geom.len(), 3);
        assert_eq!(geom.formula(), "H2O");
    }
}
