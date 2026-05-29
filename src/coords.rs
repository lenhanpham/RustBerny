//! Internal redundant coordinates for molecular optimization.
//!
//! Provides bond, angle, and dihedral coordinate types with analytic gradients,
//! B-matrix computation, and adaptive linear-bend handling with dummy atoms.

use itertools::Itertools;
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use std::collections::HashSet;

use crate::geometry::{Geometry, ANGSTROM};
use crate::math;
use crate::species;

/// Linear bend entry threshold in radians (175°).
const LIN_THRE: f64 = 175.0 * std::f64::consts::PI / 180.0;

/// Linear bend exit threshold in radians (170°).
const LIN_EXIT: f64 = 170.0 * std::f64::consts::PI / 180.0;

/// Dummy atom offset from host atom in angstrom.
const DUMMY_OFFSET: f64 = 1.0;

/// Evaluation result: either just the value or value plus gradients.
#[derive(Debug)]
pub enum EvalReturn {
    /// Coordinate value only.
    Value(f64),
    /// Coordinate value and gradients w.r.t. each indexed atom.
    ValueAndGrads(f64, Vec<Vector3<f64>>),
}

/// Trait for internal coordinate types (bond, angle, dihedral).
pub trait InternalCoord: std::fmt::Debug {
    /// Returns the atom indices defining this coordinate.
    fn idx(&self) -> Vec<usize>;

    /// Returns the weakness level (0=strong, 1=weak, 2=superweak).
    fn weak(&self) -> Option<u32>;

    /// Sets the weakness level.
    fn set_weak(&mut self, w: Option<u32>);

    /// Returns the diagonal Hessian guess coefficient.
    fn hessian_guess(&self, rho: &DMatrix<f64>) -> f64;

    /// Returns the coordinate weight.
    fn weight(&self, rho: &DMatrix<f64>, coords: &[Vector3<f64>]) -> f64;

    /// Evaluates the coordinate value and optionally its gradients.
    fn eval(&self, coords: &[Vector3<f64>], grad: bool) -> EvalReturn;

    /// Returns the center for periodic reduction.
    fn center(&self, ijk: &[[i32; 3]]) -> [i32; 3];

    /// Downcast support.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Bond coordinate between atoms i and j.
#[derive(Debug, Clone)]
pub struct Bond {
    /// Atom indices (sorted: i < j).
    pub idx: [usize; 2],
    /// Weakness level.
    pub weak: Option<u32>,
}

impl Bond {
    /// Creates a new bond coordinate.
    pub fn new(i: usize, j: usize, weak: Option<u32>) -> Self {
        let (i, j) = if i > j { (j, i) } else { (i, j) };
        Self { idx: [i, j], weak }
    }
}

impl InternalCoord for Bond {
    fn idx(&self) -> Vec<usize> {
        self.idx.to_vec()
    }

    fn weak(&self) -> Option<u32> {
        self.weak
    }

    fn set_weak(&mut self, w: Option<u32>) {
        self.weak = w;
    }

    fn hessian_guess(&self, rho: &DMatrix<f64>) -> f64 {
        0.45 * rho[(self.idx[0], self.idx[1])]
    }

    fn weight(&self, rho: &DMatrix<f64>, _coords: &[Vector3<f64>]) -> f64 {
        rho[(self.idx[0], self.idx[1])]
    }

    fn eval(&self, coords: &[Vector3<f64>], grad: bool) -> EvalReturn {
        let v = (coords[self.idx[0]] - coords[self.idx[1]]) * ANGSTROM;
        let r = v.norm();
        if !grad {
            return EvalReturn::Value(r);
        }
        let dv = v / r;
        EvalReturn::ValueAndGrads(r, vec![dv, -dv])
    }

    fn center(&self, ijk: &[[i32; 3]]) -> [i32; 3] {
        let a = ijk[self.idx[0]];
        let b = ijk[self.idx[1]];
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Angle coordinate between atoms i, j, k (j is the vertex).
#[derive(Debug, Clone)]
pub struct Angle {
    /// Atom indices (sorted: i < k).
    pub idx: [usize; 3],
    /// Weakness level.
    pub weak: Option<u32>,
}

impl Angle {
    /// Creates a new angle coordinate.
    pub fn new(i: usize, j: usize, k: usize, weak: Option<u32>) -> Self {
        let (i, k) = if i > k { (k, i) } else { (i, k) };
        Self {
            idx: [i, j, k],
            weak,
        }
    }
}

impl InternalCoord for Angle {
    fn idx(&self) -> Vec<usize> {
        self.idx.to_vec()
    }

    fn weak(&self) -> Option<u32> {
        self.weak
    }

    fn set_weak(&mut self, w: Option<u32>) {
        self.weak = w;
    }

    fn hessian_guess(&self, rho: &DMatrix<f64>) -> f64 {
        0.15 * rho[(self.idx[0], self.idx[1])] * rho[(self.idx[1], self.idx[2])]
    }

    fn weight(&self, rho: &DMatrix<f64>, coords: &[Vector3<f64>]) -> f64 {
        let f = 0.12;
        let ang = match self.eval(coords, false) {
            EvalReturn::Value(v) => v,
            _ => unreachable!(),
        };
        (rho[(self.idx[0], self.idx[1])] * rho[(self.idx[1], self.idx[2])]).sqrt()
            * (f + (1.0 - f) * ang.sin())
    }

    fn eval(&self, coords: &[Vector3<f64>], grad: bool) -> EvalReturn {
        let v1 = (coords[self.idx[0]] - coords[self.idx[1]]) * ANGSTROM;
        let v2 = (coords[self.idx[2]] - coords[self.idx[1]]) * ANGSTROM;
        let n1 = v1.norm();
        let n2 = v2.norm();
        let mut dot_product = v1.dot(&v2) / (n1 * n2);
        dot_product = dot_product.clamp(-1.0, 1.0);
        let phi = dot_product.acos();

        if !grad {
            return EvalReturn::Value(phi);
        }

        let grads = if (phi - std::f64::consts::PI).abs() < 1e-6 {
            // Near-linear branch
            let pi_minus_phi = std::f64::consts::PI - phi;
            vec![
                pi_minus_phi / (2.0 * n1 * n1) * v1,
                (1.0 / n1 - 1.0 / n2) * pi_minus_phi / (2.0 * n1) * v1,
                pi_minus_phi / (2.0 * n2 * n2) * v2,
            ]
        } else {
            // General branch
            let sin_phi = phi.sin();
            let tan_phi = phi.tan();
            vec![
                tan_phi.recip() * v1 / (n1 * n1) - v2 / (n1 * n2 * sin_phi),
                (v1 + v2) / (n1 * n2 * sin_phi)
                    - tan_phi.recip() * (v1 / (n1 * n1) + v2 / (n2 * n2)),
                tan_phi.recip() * v2 / (n2 * n2) - v1 / (n1 * n2 * sin_phi),
            ]
        };

        EvalReturn::ValueAndGrads(phi, grads)
    }

    fn center(&self, ijk: &[[i32; 3]]) -> [i32; 3] {
        let b = ijk[self.idx[1]];
        [2 * b[0], 2 * b[1], 2 * b[2]]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Dihedral coordinate between atoms i, j, k, l.
#[derive(Debug, Clone)]
pub struct Dihedral {
    /// Atom indices (sorted by j, k).
    pub idx: [usize; 4],
    /// Weakness level.
    pub weak: Option<u32>,
}

impl Dihedral {
    /// Creates a new dihedral coordinate.
    pub fn new(i: usize, j: usize, k: usize, l: usize, weak: Option<u32>) -> Self {
        let (i, j, k, l) = if j > k { (l, k, j, i) } else { (i, j, k, l) };
        Self {
            idx: [i, j, k, l],
            weak,
        }
    }
}

impl InternalCoord for Dihedral {
    fn idx(&self) -> Vec<usize> {
        self.idx.to_vec()
    }

    fn weak(&self) -> Option<u32> {
        self.weak
    }

    fn set_weak(&mut self, w: Option<u32>) {
        self.weak = w;
    }

    fn hessian_guess(&self, rho: &DMatrix<f64>) -> f64 {
        0.005
            * rho[(self.idx[0], self.idx[1])]
            * rho[(self.idx[1], self.idx[2])]
            * rho[(self.idx[2], self.idx[3])]
    }

    fn weight(&self, rho: &DMatrix<f64>, coords: &[Vector3<f64>]) -> f64 {
        let f = 0.12;
        let th1 = match Angle::new(self.idx[0], self.idx[1], self.idx[2], None).eval(coords, false)
        {
            EvalReturn::Value(v) => v,
            _ => unreachable!(),
        };
        let th2 = match Angle::new(self.idx[1], self.idx[2], self.idx[3], None).eval(coords, false)
        {
            EvalReturn::Value(v) => v,
            _ => unreachable!(),
        };
        (rho[(self.idx[0], self.idx[1])]
            * rho[(self.idx[1], self.idx[2])]
            * rho[(self.idx[2], self.idx[3])])
            .powf(1.0 / 3.0)
            * (f + (1.0 - f) * th1.sin())
            * (f + (1.0 - f) * th2.sin())
    }

    fn eval(&self, coords: &[Vector3<f64>], grad: bool) -> EvalReturn {
        let v1 = (coords[self.idx[0]] - coords[self.idx[1]]) * ANGSTROM;
        let v2 = (coords[self.idx[3]] - coords[self.idx[2]]) * ANGSTROM;
        let w = (coords[self.idx[2]] - coords[self.idx[1]]) * ANGSTROM;
        let nw = w.norm();
        let ew = w / nw;

        let a1 = v1 - v1.dot(&ew) * ew;
        let a2 = v2 - v2.dot(&ew) * ew;

        // Sign from determinant
        let det = nalgebra::Matrix3::from_columns(&[v2, v1, w]).determinant();
        let sgn = if det == 0.0 { 1.0 } else { det.signum() };

        let na1 = a1.norm();
        let na2 = a2.norm();
        let mut dot_product = a1.dot(&a2) / (na1 * na2);
        dot_product = dot_product.clamp(-1.0, 1.0);
        let phi = dot_product.acos() * sgn;

        if !grad {
            return EvalReturn::Value(phi);
        }

        let abs_phi = phi.abs();

        let grads = if abs_phi > std::f64::consts::PI - 1e-6 {
            // Near π branch
            let g = math::cross(&w, &a1);
            let ng = g.norm();
            let g_hat = g / ng;
            let a = v1.dot(&ew) / nw;
            let b = v2.dot(&ew) / nw;
            vec![
                g_hat / na1,
                -((1.0 - a) / na1 - b / na2) * g_hat,
                -((1.0 + b) / na2 + a / na1) * g_hat,
                g_hat / na2,
            ]
        } else if abs_phi < 1e-6 {
            // Near 0 branch
            let g = math::cross(&w, &a1);
            let ng = g.norm();
            let g_hat = g / ng;
            let a = v1.dot(&ew) / nw;
            let b = v2.dot(&ew) / nw;
            vec![
                g_hat / na1,
                -((1.0 - a) / na1 + b / na2) * g_hat,
                ((1.0 + b) / na2 - a / na1) * g_hat,
                -g_hat / na2,
            ]
        } else {
            // General branch
            let sin_phi = phi.sin();
            let tan_phi = phi.tan();
            let a = v1.dot(&ew) / nw;
            let b = v2.dot(&ew) / nw;
            vec![
                tan_phi.recip() * a1 / (na1 * na1) - a2 / (na1 * na2 * sin_phi),
                ((1.0 - a) * a2 - b * a1) / (na1 * na2 * sin_phi)
                    - tan_phi.recip() * ((1.0 - a) * a1 / (na1 * na1) - b * a2 / (na2 * na2)),
                ((1.0 + b) * a1 + a * a2) / (na1 * na2 * sin_phi)
                    - tan_phi.recip() * ((1.0 + b) * a2 / (na2 * na2) + a * a1 / (na1 * na1)),
                tan_phi.recip() * a2 / (na2 * na2) - a1 / (na1 * na2 * sin_phi),
            ]
        };

        EvalReturn::ValueAndGrads(phi, grads)
    }

    fn center(&self, ijk: &[[i32; 3]]) -> [i32; 3] {
        let b = ijk[self.idx[1]];
        let c = ijk[self.idx[2]];
        [b[0] + c[0], b[1] + c[1], b[2] + c[2]]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Specification for placing a dummy atom near a linear bend.
#[derive(Debug, Clone)]
struct DummySpec {
    i: usize,
    j: usize,
    k: usize,
    ref_dir: Vector3<f64>,
}

impl DummySpec {
    /// Places the dummy atom at the current geometry.
    fn place(&self, coords: &[Vector3<f64>]) -> Vector3<f64> {
        let axis = coords[self.k] - coords[self.i];
        let n_axis = axis.norm();
        if n_axis < 1e-8 {
            return coords[self.j];
        }
        let axis_hat = axis / n_axis;
        let perp = perp_from_ref(&self.ref_dir, &axis_hat).unwrap_or_else(|| pick_perp(&axis_hat));
        coords[self.j] + DUMMY_OFFSET * perp
    }

    /// Places the dummy and returns Jacobians w.r.t. host atoms (i, j, k).
    fn place_and_jacobians(
        &self,
        coords: &[Vector3<f64>],
    ) -> (Vector3<f64>, Matrix3<f64>, Matrix3<f64>, Matrix3<f64>) {
        let ri = coords[self.i];
        let rj = coords[self.j];
        let rk = coords[self.k];
        let i3 = Matrix3::identity();
        let z3 = Matrix3::zeros();

        let w = rk - ri;
        let nw = w.norm();
        if nw < 1e-8 {
            return (rj, z3, i3, z3);
        }
        let ahat = w / nw;
        let a_dot_ref = ahat.dot(&self.ref_dir);
        let q = self.ref_dir - a_dot_ref * ahat;
        let nq = q.norm();
        if nq < 1e-8 {
            return (rj + DUMMY_OFFSET * pick_perp(&ahat), z3, i3, z3);
        }
        let phat = q / nq;
        let d = rj + DUMMY_OFFSET * phat;

        let pa = i3 - ahat * ahat.transpose();
        let pp = i3 - phat * phat.transpose();

        let dq_drk = -(ahat * q.transpose() + a_dot_ref * pa) / nw;
        let dp_drk = (pp / nq) * dq_drk;
        let jk = DUMMY_OFFSET * dp_drk;
        let ji = -jk;

        (d, ji, i3, jk)
    }
}

/// Projects ref onto plane perpendicular to axis and normalizes.
fn perp_from_ref(ref_dir: &Vector3<f64>, axis: &Vector3<f64>) -> Option<Vector3<f64>> {
    let perp = ref_dir - ref_dir.dot(axis) * axis;
    let n = perp.norm();
    if n < 1e-8 {
        return None;
    }
    Some(perp / n)
}

/// Returns a unit vector perpendicular to axis.
fn pick_perp(axis: &Vector3<f64>) -> Vector3<f64> {
    let mut e = Vector3::zeros();
    let min_idx = axis
        .abs()
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    e[min_idx] = 1.0;
    perp_from_ref(&e, axis).unwrap_or_else(|| {
        let mut e2 = Vector3::zeros();
        e2[(min_idx + 1) % 3] = 1.0;
        perp_from_ref(&e2, axis).unwrap()
    })
}

/// Checks if an atom has exactly two covalent neighbors (sp-like).
fn is_sp_like_centre(bondmatrix: &[Vec<bool>], j: usize) -> bool {
    bondmatrix[j].iter().filter(|&&b| b).count() == 2
}

/// Yields all connectivity-allowed triples (j, i, k).
fn iter_candidate_triples(bondmatrix: &[Vec<bool>]) -> Vec<(usize, usize, usize)> {
    let mut triples = Vec::new();
    for j in 0..bondmatrix.len() {
        let neighbors: Vec<usize> = bondmatrix[j]
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();
        for pair in neighbors.iter().combinations(2) {
            let (i, k) = (*pair[0], *pair[1]);
            triples.push((j, i, k));
        }
    }
    triples
}

/// Finds near-linear triples in the bond matrix.
fn find_linear_triples(
    bondmatrix: &[Vec<bool>],
    coords: &[Vector3<f64>],
) -> Vec<(usize, usize, usize)> {
    iter_candidate_triples(bondmatrix)
        .into_iter()
        .filter(|&(j, i, k)| {
            let ang = Angle::new(i, j, k, None).eval(coords, false);
            match ang {
                EvalReturn::Value(v) => v > LIN_THRE,
                _ => false,
            }
        })
        .collect()
}

/// Finds connected fragments via BFS.
fn get_clusters(bondmatrix: &[Vec<bool>]) -> (Vec<Vec<usize>>, Vec<Vec<bool>>) {
    let n = bondmatrix.len();
    let mut assigned = vec![false; n];
    let mut clusters = Vec::new();

    for start in 0..n {
        if assigned[start] {
            continue;
        }
        let mut cluster = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        assigned[start] = true;

        while let Some(node) = queue.pop_front() {
            cluster.push(node);
            for (neighbor, &connected) in bondmatrix[node].iter().enumerate() {
                if connected && !assigned[neighbor] {
                    assigned[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        clusters.push(cluster);
    }

    // Build cluster adjacency matrix
    let mut c = vec![vec![false; n]; n];
    for cluster in &clusters {
        for &i in cluster {
            for &j in cluster {
                c[i][j] = true;
            }
        }
    }

    (clusters, c)
}

/// Generates dihedral coordinates for a bond.
fn get_dihedrals(
    center: &[usize],
    coords: &[Vector3<f64>],
    bondmatrix: &[Vec<bool>],
    c: &[Vec<bool>],
    superweak: bool,
) -> Vec<Dihedral> {
    let lin_thre_dih = 5.0 * std::f64::consts::PI / 180.0;

    let neigh_l: Vec<usize> = bondmatrix[center[0]]
        .iter()
        .enumerate()
        .filter(|&(n, &b)| b && !center.contains(&n))
        .map(|(n, _)| n)
        .collect();
    let neigh_r: Vec<usize> = bondmatrix[center[center.len() - 1]]
        .iter()
        .enumerate()
        .filter(|&(n, &b)| b && !center.contains(&n))
        .map(|(n, _)| n)
        .collect();

    let angles_l: Vec<f64> = neigh_l
        .iter()
        .filter_map(
            |&i| match Angle::new(i, center[0], center[1], None).eval(coords, false) {
                EvalReturn::Value(v) => Some(v),
                _ => None,
            },
        )
        .collect();
    let angles_r: Vec<f64> = neigh_r
        .iter()
        .filter_map(|&j| {
            match Angle::new(center[center.len() - 2], center[center.len() - 1], j, None)
                .eval(coords, false)
            {
                EvalReturn::Value(v) => Some(v),
                _ => None,
            }
        })
        .collect();

    let nonlinear_l: Vec<usize> = neigh_l
        .iter()
        .zip(angles_l.iter())
        .filter(|&(_, &ang)| ang < std::f64::consts::PI - lin_thre_dih && ang >= lin_thre_dih)
        .map(|(&n, _)| n)
        .collect();
    let nonlinear_r: Vec<usize> = neigh_r
        .iter()
        .zip(angles_r.iter())
        .filter(|&(_, &ang)| ang < std::f64::consts::PI - lin_thre_dih && ang >= lin_thre_dih)
        .map(|(&n, _)| n)
        .collect();

    let mut dihedrals = Vec::new();
    if center[0] < center[center.len() - 1] {
        let nweak = center.windows(2).filter(|w| !c[w[0]][w[1]]).count();
        for &nl in &nonlinear_l {
            for &nr in &nonlinear_r {
                if nl == nr {
                    continue;
                }
                let weak = nweak
                    + if c[nl][center[0]] { 0 } else { 1 }
                    + if c[center[0]][nr] { 0 } else { 1 };
                if !superweak && weak > 1 {
                    continue;
                }
                dihedrals.push(Dihedral::new(
                    nl,
                    center[0],
                    *center.last().unwrap(),
                    nr,
                    Some(weak as u32),
                ));
            }
        }
    }

    // Handle linear extensions
    let linear_l: Vec<usize> = neigh_l
        .iter()
        .zip(angles_l.iter())
        .filter(|&(_, &ang)| ang >= std::f64::consts::PI - lin_thre_dih || ang < lin_thre_dih)
        .map(|(&n, _)| n)
        .collect();
    let linear_r: Vec<usize> = neigh_r
        .iter()
        .zip(angles_r.iter())
        .filter(|&(_, &ang)| ang >= std::f64::consts::PI - lin_thre_dih || ang < lin_thre_dih)
        .map(|(&n, _)| n)
        .collect();

    if center.len() <= 3 {
        if !linear_l.is_empty() && linear_r.is_empty() {
            let mut new_center = linear_l.clone();
            new_center.extend_from_slice(center);
            dihedrals.extend(get_dihedrals(&new_center, coords, bondmatrix, c, superweak));
        } else if linear_r.is_empty() && !linear_l.is_empty() {
            // This case is handled by the condition above
        } else if !linear_r.is_empty() && linear_l.is_empty() {
            let mut new_center = center.to_vec();
            new_center.extend_from_slice(&linear_r);
            dihedrals.extend(get_dihedrals(&new_center, coords, bondmatrix, c, superweak));
        }
    }

    dihedrals
}

/// Manages the internal coordinate system for a geometry.
pub struct InternalCoords {
    /// All coordinates (bonds, angles, dihedrals).
    coords: Vec<Box<dyn InternalCoord>>,
    /// Dummy atom specifications for linear bends.
    dummy_specs: Vec<DummySpec>,
    /// Current dummy atom positions.
    dummy_atoms: Vec<Vector3<f64>>,
    /// Number of real atoms.
    n_real: usize,
    /// Fragment membership.
    #[allow(dead_code)]
    fragments: Vec<Vec<usize>>,
    /// Bond matrix snapshot for rebuild detection.
    bondmatrix: Option<Vec<Vec<bool>>>,
    /// Current set of linear triples.
    linear_set: HashSet<(usize, usize, usize)>,
}

impl InternalCoords {
    /// Builds the coordinate system for a geometry.
    pub fn new(geom: &Geometry, dihedral: bool, superweakdih: bool) -> Self {
        let super_geom = geom.supercell();
        let n_real = super_geom.len();
        let dist = super_geom.dist();

        // Build bond matrix from covalent radii
        let radii: Vec<f64> = super_geom
            .species
            .iter()
            .map(|sp| species::get_property(sp, "covalent_radius"))
            .collect();
        let mut bondmatrix: Vec<Vec<bool>> = vec![vec![false; n_real]; n_real];
        for i in 0..n_real {
            for j in (i + 1)..n_real {
                if dist[(i, j)] < 1.3 * (radii[i] + radii[j]) {
                    bondmatrix[i][j] = true;
                    bondmatrix[j][i] = true;
                }
            }
        }

        // Find connected fragments
        let (fragments, mut c) = get_clusters(&bondmatrix);

        // Expand connectivity using vdW radii
        let vdw_radii: Vec<f64> = super_geom
            .species
            .iter()
            .map(|sp| species::get_property(sp, "vdw_radius"))
            .collect();
        let mut shift = 0.0;
        loop {
            let all_connected = c.iter().all(|row| row.iter().all(|&b| b));
            if all_connected {
                break;
            }
            for i in 0..n_real {
                for j in (i + 1)..n_real {
                    if !c[i][j] && dist[(i, j)] < vdw_radii[i] + vdw_radii[j] + shift {
                        bondmatrix[i][j] = true;
                        bondmatrix[j][i] = true;
                    }
                }
            }
            let (_, new_c) = get_clusters(&bondmatrix);
            for i in 0..n_real {
                for j in 0..n_real {
                    c[i][j] = new_c[i][j];
                }
            }
            shift += 1.0;
        }

        // Add bonds
        let mut coords: Vec<Box<dyn InternalCoord>> = Vec::new();
        for i in 0..n_real {
            for j in (i + 1)..n_real {
                if bondmatrix[i][j] {
                    coords.push(Box::new(Bond::new(i, j, None)));
                }
            }
        }

        // Find linear triples
        let linear_set: HashSet<(usize, usize, usize)> = if geom.lattice.is_none() {
            find_linear_triples(&bondmatrix, &super_geom.coords)
                .into_iter()
                .collect()
        } else {
            HashSet::new()
        };

        // Build angles (skip linear triples)
        for j in 0..n_real {
            let neighbors: Vec<usize> = bondmatrix[j]
                .iter()
                .enumerate()
                .filter(|(_, &b)| b)
                .map(|(i, _)| i)
                .collect();
            for pair in neighbors.iter().combinations(2) {
                let (i, k) = (*pair[0], *pair[1]);
                if linear_set.contains(&(j, i, k)) {
                    continue;
                }
                let ang = Angle::new(i, j, k, None);
                let ang_val = match ang.eval(&super_geom.coords, false) {
                    EvalReturn::Value(v) => v,
                    _ => continue,
                };
                if ang_val > std::f64::consts::FRAC_PI_4 {
                    coords.push(Box::new(ang));
                }
            }
        }

        // Add dummy atoms for linear bends (sp-like centers only)
        let mut dummy_specs = Vec::new();
        let mut dummy_atoms = Vec::new();

        for &(j, i, k) in linear_set.iter() {
            if is_sp_like_centre(&bondmatrix, j) {
                // Create two perpendicular reference directions
                let axis = super_geom.coords[k] - super_geom.coords[i];
                let n_axis = axis.norm();
                let axis_hat = axis / n_axis;
                let ref1 = pick_perp(&axis_hat);
                let ref2 = axis_hat.cross(&ref1);
                let ref2 = ref2 / ref2.norm();

                let d1 = n_real + dummy_specs.len();
                dummy_specs.push(DummySpec {
                    i,
                    j,
                    k,
                    ref_dir: ref1,
                });
                let d2 = n_real + dummy_specs.len();
                dummy_specs.push(DummySpec {
                    i,
                    j,
                    k,
                    ref_dir: ref2,
                });

                // Refresh dummy positions
                dummy_atoms = dummy_specs
                    .iter()
                    .map(|spec| spec.place(&super_geom.coords))
                    .collect();

                // Add four replacement angles
                for &ai in &[i, k] {
                    for &ad in &[d1, d2] {
                        coords.push(Box::new(Angle::new(ai, j, ad, Some(0))));
                    }
                }
            }
        }

        // Add dihedrals (only if the dihedral flag is set, matching Python behaviour)
        if dihedral {
            let mut all_dihedrals = Vec::new();
            for coord in &coords {
                if let Some(bond) = coord.as_any().downcast_ref::<Bond>() {
                    all_dihedrals.extend(get_dihedrals(
                        &bond.idx,
                        &super_geom.coords,
                        &bondmatrix,
                        &c,
                        superweakdih,
                    ));
                }
            }
            for dih in all_dihedrals {
                coords.push(Box::new(dih));
            }
        }

        Self {
            coords,
            dummy_specs,
            dummy_atoms,
            n_real,
            fragments,
            bondmatrix: if geom.lattice.is_none() {
                Some(bondmatrix)
            } else {
                None
            },
            linear_set,
        }
    }

    /// Returns the number of coordinates.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Returns `true` if there are no coordinates.
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Returns bond coordinates.
    pub fn bonds(&self) -> Vec<&dyn InternalCoord> {
        self.coords
            .iter()
            .filter(|c| c.as_any().downcast_ref::<Bond>().is_some())
            .map(|c| c.as_ref())
            .collect()
    }

    /// Returns angle coordinates.
    pub fn angles(&self) -> Vec<&dyn InternalCoord> {
        self.coords
            .iter()
            .filter(|c| c.as_any().downcast_ref::<Angle>().is_some())
            .map(|c| c.as_ref())
            .collect()
    }

    /// Returns dihedral coordinates.
    pub fn dihedrals(&self) -> Vec<&dyn InternalCoord> {
        self.coords
            .iter()
            .filter(|c| c.as_any().downcast_ref::<Dihedral>().is_some())
            .map(|c| c.as_ref())
            .collect()
    }

    /// Returns all coordinates.
    pub fn all_coords(&self) -> &[Box<dyn InternalCoord>] {
        &self.coords
    }

    /// Refreshes dummy atom positions from real coordinates.
    fn refresh_dummies(&mut self, real_coords: &[Vector3<f64>]) {
        self.dummy_atoms = self
            .dummy_specs
            .iter()
            .map(|spec| spec.place(real_coords))
            .collect();
    }

    /// Returns combined (real + dummy) coordinate array.
    fn all_atom_coords(&mut self, geom_super: &Geometry) -> Vec<Vector3<f64>> {
        if self.dummy_specs.is_empty() {
            return geom_super.coords.clone();
        }
        self.refresh_dummies(&geom_super.coords);
        let mut all = geom_super.coords.clone();
        all.extend_from_slice(&self.dummy_atoms);
        all
    }

    /// Returns extended rho matrix for real + dummy centers.
    fn rho_extended(&self, geom_super: &Geometry) -> DMatrix<f64> {
        let rho_real = geom_super.rho();
        let nd = self.dummy_specs.len();
        if nd == 0 {
            return rho_real;
        }
        let n = self.n_real + nd;
        let mut rho = DMatrix::from_element(n, n, 1.0);
        let nr = self.n_real;
        for i in 0..nr {
            for j in 0..nr {
                rho[(i, j)] = rho_real[(i, j)];
            }
        }
        rho
    }

    /// Evaluates all coordinates for a geometry.
    pub fn eval_geom(&mut self, geom: &Geometry, template: Option<&DVector<f64>>) -> DVector<f64> {
        let super_geom = geom.supercell();
        let all_coords = self.all_atom_coords(&super_geom);
        let mut q: Vec<f64> = self
            .coords
            .iter()
            .map(|coord| match coord.eval(&all_coords, false) {
                EvalReturn::Value(v) => v,
                _ => unreachable!(),
            })
            .collect();

        // --- Dihedral unwrapping (Python `eval_geom` with template) ---
        if let Some(template) = template {
            use std::collections::HashSet;
            use std::f64::consts::PI;

            // Helper: return the normalised angle-idx for a dihedral component.
            // Dihedral [a,b,c,d] has two valence angles: (a,b,c) and (b,c,d).
            // The Angle::new constructor normalises so that the first atom < last atom.
            let norm_angle_idx = |i: usize, j: usize, k: usize| -> [usize; 3] {
                if i < k { [i, j, k] } else { [k, j, i] }
            };

            // Pass 1: handle dihedral wrapping by 2π or π.
            let mut swapped_dih_idx: Vec<usize> = Vec::new();   // coord-list positions
            let mut candidate_angles: HashSet<[usize; 3]> = HashSet::new();

            for i in 0..self.coords.len() {
                if let Some(dih) = self.coords[i].as_any().downcast_ref::<Dihedral>() {
                    let diff = q[i] - template[i];
                    if (diff.abs() - 2.0 * PI).abs() < PI / 2.0 {
                        q[i] -= 2.0 * PI * diff.signum();
                    } else if (diff.abs() - PI).abs() < PI / 2.0 {
                        q[i] -= PI * diff.signum();
                        swapped_dih_idx.push(i);
                        let [a, b, c, d] = dih.idx;
                        candidate_angles.insert(norm_angle_idx(a, b, c));
                        candidate_angles.insert(norm_angle_idx(b, c, d));
                    }
                }
            }

            // Pass 2: for each candidate angle, check if all its dihedrals
            // were either swapped or have all angles as candidates, then flip it.
            if !candidate_angles.is_empty() {
                for i in 0..self.coords.len() {
                    if let Some(ang) = self.coords[i].as_any().downcast_ref::<Angle>() {
                        if !candidate_angles.contains(&ang.idx) {
                            continue;
                        }
                        // All dihedrals that contain this angle.
                        let should_swap = (0..self.coords.len()).all(|j| {
                            let dih_opt = self.coords[j].as_any().downcast_ref::<Dihedral>();
                            match dih_opt {
                                None => true, // not a dihedral — skip
                                Some(dih) => {
                                    let [a, b, c, d] = dih.idx;
                                    let a1 = norm_angle_idx(a, b, c);
                                    let a2 = norm_angle_idx(b, c, d);
                                    // Does this dihedral contain our angle?
                                    if a1 != ang.idx && a2 != ang.idx {
                                        return true; // does not contain it — skip
                                    }
                                    // Contains it: must be swapped or all its angles are candidates.
                                    swapped_dih_idx.contains(&j)
                                        || (candidate_angles.contains(&a1)
                                            && candidate_angles.contains(&a2))
                                }
                            }
                        });
                        if should_swap {
                            q[i] = 2.0 * PI - q[i];
                        }
                    }
                }
            }
        }

        DVector::from_vec(q)
    }

    /// Returns the B-matrix, shape (n_coords, 3*n_real).
    pub fn b_matrix(&mut self, geom: &Geometry) -> DMatrix<f64> {
        let super_geom = geom.supercell();
        let n_real = super_geom.len();
        let all_coords = self.all_atom_coords(&super_geom);

        // Pre-compute Jacobians for dummy atoms
        let jacs: Vec<(
            usize,
            usize,
            usize,
            Matrix3<f64>,
            Matrix3<f64>,
            Matrix3<f64>,
        )> = self
            .dummy_specs
            .iter()
            .map(|spec| {
                let (_, j_i, j_j, j_k) = spec.place_and_jacobians(&super_geom.coords);
                (spec.i, spec.j, spec.k, j_i, j_j, j_k)
            })
            .collect();

        let n_coords = self.coords.len();
        let mut b = vec![vec![Vector3::zeros(); n_real]; n_coords];

        for (i, coord) in self.coords.iter().enumerate() {
            let result = coord.eval(&all_coords, true);
            if let EvalReturn::ValueAndGrads(_, grads) = result {
                for (k_idx, grad) in coord.idx().iter().zip(grads.iter()) {
                    if *k_idx < n_real {
                        b[i][*k_idx % n_real] += grad;
                    } else {
                        let (hi, hj, hk, ji, jj, jk) = &jacs[k_idx - n_real];
                        b[i][*hi] += ji.transpose() * grad;
                        b[i][*hj] += jj.transpose() * grad;
                        b[i][*hk] += jk.transpose() * grad;
                    }
                }
            }
        }

        // Flatten to (n_coords, 3*n_real)
        let mut result = DMatrix::zeros(n_coords, 3 * n_real);
        for i in 0..n_coords {
            for j in 0..n_real {
                for k in 0..3 {
                    result[(i, 3 * j + k)] = b[i][j][k];
                }
            }
        }
        result
    }

    /// Returns the diagonal Hessian guess.
    pub fn hessian_guess(&mut self, geom: &Geometry) -> DMatrix<f64> {
        let super_geom = geom.supercell();
        let rho = self.rho_extended(&super_geom);
        let diag: Vec<f64> = self
            .coords
            .iter()
            .map(|coord| coord.hessian_guess(&rho))
            .collect();
        DMatrix::from_diagonal(&DVector::from_vec(diag))
    }

    /// Returns coordinate weights.
    pub fn weights(&mut self, geom: &Geometry) -> DVector<f64> {
        let super_geom = geom.supercell();
        let rho = self.rho_extended(&super_geom);
        let all_coords = self.all_atom_coords(&super_geom);
        DVector::from_fn(self.coords.len(), |i, _| {
            self.coords[i].weight(&rho, &all_coords)
        })
    }

    /// Checks if the coordinate system needs rebuilding.
    pub fn needs_rebuild(&self, geom: &Geometry) -> bool {
        let bondmatrix = match &self.bondmatrix {
            Some(bm) => bm,
            None => return false,
        };

        let super_geom = geom.supercell();
        if super_geom.len() != self.n_real {
            return false;
        }

        let mut candidate = HashSet::new();
        for (j, i, k) in iter_candidate_triples(bondmatrix) {
            let ang = Angle::new(i, j, k, None).eval(&super_geom.coords, false);
            let ang_val = match ang {
                EvalReturn::Value(v) => v,
                _ => continue,
            };
            if self.linear_set.contains(&(j, i, k)) {
                if ang_val > LIN_EXIT {
                    candidate.insert((j, i, k));
                }
            } else if ang_val > LIN_THRE {
                candidate.insert((j, i, k));
            }
        }

        candidate != self.linear_set
    }

    /// Updates geometry from internal coordinate step.
    pub fn update_geom(
        &mut self,
        geom: &Geometry,
        q: &DVector<f64>,
        dq: &DVector<f64>,
        b_inv: &DMatrix<f64>,
    ) -> (DVector<f64>, Geometry) {
        let mut geom = geom.clone();
        let thre = 1e-6;
        let mut _keep_first = (geom.clone(), q.clone(), None::<f64>, None::<f64>);
        let mut q = q.clone();
        let mut dq = dq.clone();

        for i in 0..20 {
            let dq_flat = b_inv * &dq;
            let coords_new: Vec<Vector3<f64>> = geom
                .coords
                .iter()
                .enumerate()
                .map(|(j, c)| {
                    let d = Vector3::new(dq_flat[3 * j], dq_flat[3 * j + 1], dq_flat[3 * j + 2]);
                    c + d / ANGSTROM
                })
                .collect();

            let dcart_rms = math::rms(
                &coords_new
                    .iter()
                    .zip(geom.coords.iter())
                    .map(|(a, b)| (a - b).norm())
                    .collect::<Vec<_>>(),
            );

            geom.coords = coords_new;
            let q_new = self.eval_geom(&geom, Some(&q));
            let dq_rms = math::rms_vec(&(q_new.clone() - &q));
            // Python: q, dq = q_new, dq - (q_new - q)
            // i.e. new remaining displacement = old dq minus the actual q change.
            let q_old = q.clone();
            q = q_new;
            dq = dq - (&q - &q_old);

            if let Some(d) = dcart_rms {
                if d < thre {
                    break;
                }
            }

            if i == 0 {
                _keep_first = (geom.clone(), q.clone(), dcart_rms, dq_rms);
            }
        }

        (q, geom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Geometry;

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
    fn test_bond_eval() {
        let geom = water();
        let mut coords = InternalCoords::new(&geom, true, false);
        let q = coords.eval_geom(&geom, None);
        // O-H bond should be ~0.96 Å
        let bond_val = q[0];
        assert!(
            bond_val > 0.5 && bond_val < 2.0,
            "Bond value {bond_val} out of range"
        );
    }

    #[test]
    fn test_b_matrix_shape() {
        let geom = water();
        let mut coords = InternalCoords::new(&geom, true, false);
        let b = coords.b_matrix(&geom);
        assert_eq!(b.nrows(), coords.len());
        assert_eq!(b.ncols(), 3 * geom.len());
    }

    #[test]
    fn test_hessian_guess() {
        let geom = water();
        let mut coords = InternalCoords::new(&geom, true, false);
        let h = coords.hessian_guess(&geom);
        assert_eq!(h.nrows(), coords.len());
        assert_eq!(h.ncols(), coords.len());
        // Diagonal should be positive
        for i in 0..h.nrows() {
            assert!(h[(i, i)] > 0.0);
        }
    }
}
