//! Core geometry optimizer using rational function optimization.
//!
//! Supports both energy minimization and transition state search.

use crate::coords::{Angle, Bond, Dihedral, InternalCoord, InternalCoords};
use crate::geometry::Geometry;
use crate::hessian;
use crate::math;
use crate::step;
use crate::trust;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Optimizer mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Energy minimization.
    Min,
    /// Transition state search.
    Ts,
}

/// Tunable parameters for the optimizer.
#[derive(Debug, Clone)]
pub struct BernyParams {
    /// Maximum gradient threshold (atomic units).
    pub gradient_max: f64,
    /// RMS gradient threshold (atomic units).
    pub gradient_rms: f64,
    /// Maximum step threshold (atomic units).
    pub step_max: f64,
    /// RMS step threshold (atomic units).
    pub step_rms: f64,
    /// Initial trust radius (atomic units).
    pub trust: f64,
    /// Estimated energy precision (atomic units).
    pub energy_noise: f64,
    /// Whether to form dihedral angles.
    pub dihedral: bool,
    /// Whether to form superweak dihedral angles.
    pub superweak_dih: bool,
    /// Optimization mode.
    pub mode: Mode,
}

impl Default for BernyParams {
    fn default() -> Self {
        Self {
            gradient_max: 0.45e-3,
            gradient_rms: 0.15e-3,
            step_max: 1.8e-3,
            step_rms: 1.2e-3,
            trust: 0.3,
            energy_noise: 2e-8,
            dihedral: true,
            superweak_dih: false,
            mode: Mode::Min,
        }
    }
}

/// A point in optimization space (coordinates, energy, gradient).
#[derive(Debug, Clone)]
pub struct OptPoint {
    /// Internal coordinates.
    pub q: DVector<f64>,
    /// Energy (if computed).
    pub e: Option<f64>,
    /// Gradient in internal coordinates (if computed).
    pub g: Option<DVector<f64>>,
}

/// Mutable optimizer state.
struct BernyState {
    geom: Geometry,
    prev_geom: Geometry,
    params: BernyParams,
    trust: f64,
    coords: InternalCoords,
    h: DMatrix<f64>,
    weights: DVector<f64>,
    future: OptPoint,
    first: bool,
    interpolated: Option<OptPoint>,
    predicted: Option<OptPoint>,
    previous: Option<OptPoint>,
    best: Option<OptPoint>,
    /// Reaction-coordinate eigenvector followed across cycles in TS mode.
    ts_mode: Option<DVector<f64>>,
}

/// Geometry optimizer using rational function optimization.
///
/// Supports minimization (RFO) and transition state search (P-RFO).
pub struct Berny {
    state: BernyState,
    n: usize,
    maxsteps: usize,
    converged: bool,
}

impl Berny {
    /// Creates a new optimizer for the given geometry.
    pub fn new(geom: Geometry, maxsteps: usize, params: BernyParams) -> Self {
        let trust = params.trust;
        let mut coords = InternalCoords::new(&geom, params.dihedral, params.superweak_dih);
        let h = coords.hessian_guess(&geom);
        let weights = coords.weights(&geom);
        let q = coords.eval_geom(&geom, None);
        let future = OptPoint {
            q,
            e: None,
            g: None,
        };

        let prev_geom = geom.clone();

        Self {
            state: BernyState {
                geom,
                prev_geom,
                params,
                trust,
                coords,
                h,
                weights,
                future,
                first: true,
                interpolated: None,
                predicted: None,
                previous: None,
                best: None,
                ts_mode: None,
            },
            n: 0,
            maxsteps,
            converged: false,
        }
    }

    /// Returns the current trust radius.
    pub fn trust(&self) -> f64 {
        self.state.trust
    }

    /// Returns whether the optimizer has converged.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Returns the current step number.
    pub fn step(&self) -> usize {
        self.n
    }

    /// Processes solver output.
    ///
    /// 1. B-matrix and pseudoinverse
    /// 2. Internal-coordinate gradient: `g_int = B_inv^T @ g_cart`
    /// 3. Hessian update (BFGS / TS flowchart) using un-projected H
    /// 4. Trust-radius update
    /// 5. Linear search (minimisation only) → interpolated point
    /// 6. H_proj = proj @ H @ proj + 1000*(I - proj)  (local, not stored in s.h)
    /// 7. Quadratic step (RFO / P-RFO)
    /// 8. Geometry update
    /// 9. Convergence check
    pub fn send(&mut self, result: (f64, Vec<Vec<f64>>, Option<Vec<Vec<f64>>>)) {
        let (energy, gradients, cartesian_hessian) = result;
        let s = &mut self.state;

        s.prev_geom = s.geom.clone();

        // Rebuild the internal-coordinate system when linear-bend topology has
        // changed since the current coordinate set was constructed.
        if !s.first && s.coords.needs_rebuild(&s.geom) {
            let old_h = s.h.clone();
            let old_coords = std::mem::replace(
                &mut s.coords,
                InternalCoords::new(&s.geom, s.params.dihedral, s.params.superweak_dih),
            );
            let guess_h = s.coords.hessian_guess(&s.geom);
            s.h = carry_over_hessian(old_coords.all_coords(), &old_h, s.coords.all_coords(), &guess_h);
            s.weights = s.coords.weights(&s.geom);
            s.future = OptPoint {
                q: s.coords.eval_geom(&s.geom, None),
                e: None,
                g: None,
            };
            s.first = true;
            s.interpolated = None;
            s.predicted = None;
            s.previous = None;
            s.best = None;
        }

        let g_flat: Vec<f64> = gradients
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let g_vec = DVector::from_vec(g_flat);

        // B-matrix: shape (n_internal, 3*n_atoms)
        let b = s.coords.b_matrix(&s.geom);
        let b_t = b.transpose();
        let bbt = &b * &b_t;
        // B_inv = B^T @ pinv(B @ B^T), shape (3*n_atoms, n_internal)
        let b_inv = &b_t * math::pinv(&bbt, &|_| {});

        // g_int = B_inv^T @ g_cart = pinv(B@B^T) @ B @ g_cart
        let g_int = b_inv.transpose() * &g_vec;

        // Current point in internal-coordinate space
        let current = OptPoint {
            q: s.future.q.clone(),
            e: Some(energy),
            g: Some(g_int.clone()),
        };

        // --- Handle first-step initial Cartesian Hessian ---
        let mut skip_hessian_update = false;
        if s.first {
            if let Some(ref cart_h) = cartesian_hessian {
                let n_atoms = s.geom.len();
                let expected = 3 * n_atoms;
                let flat: Vec<f64> = cart_h.iter().flat_map(|row| row.iter().copied()).collect();
                if flat.len() == expected * expected {
                    let h_cart = DMatrix::from_row_slice(expected, expected, &flat);
                    let h_int = b_inv.transpose() * &h_cart * &b_inv;
                    let proj = &b * &b_inv;
                    let n_coords = s.coords.len();
                    let h_proj = &proj * &h_int * &proj
                        + 1000.0 * (DMatrix::identity(n_coords, n_coords) - &proj);
                    s.h = (&h_proj + &h_proj.transpose()) / 2.0;
                    skip_hessian_update = true;
                }
            }

            // For TS searches without an external Hessian, the diagonal guess
            // has all-positive eigenvalues.  Flip one to negative so P-RFO has
            // a well-defined ascent direction from the first step.
            if s.params.mode == Mode::Ts && !skip_hessian_update {
                let eig = s.h.clone().symmetric_eigen();
                let n = eig.eigenvalues.len();
                if n > 0 {
                    // Find the smallest (most softly restrained) eigenvalue.
                    let mut idx: Vec<usize> = (0..n).collect();
                    idx.sort_by(|&a, &b| {
                        eig.eigenvalues[a]
                            .partial_cmp(&eig.eigenvalues[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let min_i = idx[0];
                    let v = eig.eigenvectors.column(min_i);
                    let neg_val = -0.2_f64.max(eig.eigenvalues[min_i].abs());
                    s.h += (neg_val - eig.eigenvalues[min_i]) * v * v.transpose();
                }
            }
        }

        // --- Hessian update (uses un-projected s.h) ---
        if !s.first && !skip_hessian_update {
            if s.params.mode == Mode::Ts {
                // TS: secant pair from immediately preceding step (previous → current)
                let prev = s.previous.as_ref().unwrap();
                let dq_step = current.q.clone() - &prev.q;
                let dg_step = g_int.clone() - prev.g.clone().unwrap();
                s.h = hessian::update_hessian_ts(&s.h, &dq_step, &dg_step);
            } else {
                // Min: secant pair from previous → current (consecutive steps)
                let prev = s.previous.as_ref().unwrap();
                let dq_step = current.q.clone() - &prev.q;
                let dg_step = g_int.clone() - prev.g.clone().unwrap();
                s.h = hessian::update_bfgs(&s.h, &dq_step, &dg_step);
            }
        }

        // Periodic eigenvalue correction for TS search: enforce exactly one
        // negative eigenvalue every 3 cycles so the quasi-Newton updates
        // don't drift away from the saddle-point signature.
        if !s.first && s.params.mode == Mode::Ts && self.n % 3 == 0 {
            ensure_one_negative_eigenvalue(&mut s.h, 0.1);
        }

        // --- Trust-radius update ---
        let mut step_rejected = false;
        if !s.first {
            let prev = s.previous.as_ref().unwrap();
            let pred = s.predicted.as_ref().unwrap();
            let interp = s.interpolated.as_ref().unwrap();
            let d_e = energy - prev.e.unwrap();
            let d_e_pred = pred.e.unwrap() - interp.e.unwrap();
            let dq_norm = (&pred.q - &interp.q).norm();
            s.trust =
                trust::update_trust(s.trust, d_e, d_e_pred, dq_norm, s.params.energy_noise);

            if s.params.mode == Mode::Min && d_e > 0.0 {
                step_rejected = true;
            }
        }

        // --- Linear search (minimisation only) ---
        if !s.first && s.params.mode == Mode::Min {
            let best = s.best.as_ref().unwrap();
            let dq_lin = &best.q - &current.q;
            let g0 = current.g.as_ref().unwrap().dot(&dq_lin);
            let g1 = best.g.as_ref().unwrap().dot(&dq_lin);
            let (t, e_interp) = linear_search(energy, best.e.unwrap(), g0, g1);
            s.interpolated = Some(OptPoint {
                q: &current.q + t * &dq_lin,
                e: Some(e_interp),
                g: Some(current.g.as_ref().unwrap() + t * (best.g.as_ref().unwrap() - current.g.as_ref().unwrap())),
            });
        } else {
            s.interpolated = Some(current.clone());
        }

        if s.trust < 1e-6 {
            panic!("Trust radius got too small — check forces");
        }

        // --- Projected Hessian (local, NOT stored in s.h) ---
        let proj = &b * &b_inv;
        let n_coords = s.coords.len();
        let h_proj =
            &proj * &s.h * &proj + 1000.0 * (DMatrix::identity(n_coords, n_coords) - &proj);
        let h_proj = (&h_proj + &h_proj.transpose()) / 2.0;
        // Condition the spectrum before the step. Minimization needs a
        // positive-definite model, but a transition-state search *requires* the
        // indefinite Hessian: P-RFO ascends the single negative-curvature mode
        // (the reaction coordinate). Forcing positive-definiteness here would
        // erase that mode and collapse the search into a plain minimization.
        let h_proj = if s.params.mode == Mode::Ts {
            h_proj
        } else {
            hessian::force_positive_definite(&h_proj, 1e-4)
        };

        // --- Quadratic step ---
        let interp = s.interpolated.as_ref().unwrap();
        let g_proj = &proj * interp.g.as_ref().unwrap();
        let (dq, d_e, on_sphere) = if s.params.mode == Mode::Ts {
            let mut tracked = s.ts_mode.take();
            let out = step::quadratic_step_ts(&g_proj, &h_proj, &s.weights, s.trust, &mut tracked);
            s.ts_mode = tracked;
            out
        } else {
            step::quadratic_step(&g_proj, &h_proj, &s.weights, s.trust)
        };

        s.predicted = Some(OptPoint {
            q: &interp.q + &dq,
            e: Some(interp.e.unwrap() + d_e),
            g: None,
        });

        let total_dq = &s.predicted.as_ref().unwrap().q - &current.q;

        // --- Geometry update ---
        if step_rejected {
            s.geom = s.prev_geom.clone();
            s.future = OptPoint {
                q: s.coords.eval_geom(&s.geom, None),
                e: None,
                g: None,
            };
        } else {
            let (_q_new, geom_new) =
                s.coords.update_geom(&s.geom, &current.q, &total_dq, &b_inv);
            s.geom = geom_new;
            s.future = OptPoint {
                q: s.coords.eval_geom(&s.geom, None),
                e: None,
                g: None,
            };
            s.previous = Some(current.clone());
        }

        // TS always advances best; for minimisation track lowest energy.
        if !step_rejected
            && (s.params.mode == Mode::Ts
                || s.first
                || s.best.as_ref().map_or(true, |b| b.e.map_or(true, |be| energy < be)))
        {
            s.best = Some(current.clone());
        }
        s.first = false;

        // --- Convergence check ---
        let n_neg = if s.params.mode == Mode::Ts {
            let ev = h_proj.symmetric_eigen().eigenvalues;
            Some(ev.iter().filter(|&&v| v < 0.0).count())
        } else {
            None
        };
        self.converged = is_converged(
            current.g.as_ref().unwrap(),
            &total_dq,
            on_sphere,
            &s.params,
            n_neg,
        );
    }

    /// Returns the current geometry.
    pub fn current_geom(&self) -> &Geometry {
        &self.state.geom
    }
}

impl Iterator for Berny {
    type Item = Geometry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n >= self.maxsteps || self.converged {
            return None;
        }
        self.n += 1;
        Some(self.state.geom.clone())
    }
}

fn coord_key(coord: &dyn InternalCoord) -> (u8, Vec<usize>) {
    let kind = if coord.as_any().is::<Bond>() {
        0
    } else if coord.as_any().is::<Angle>() {
        1
    } else if coord.as_any().is::<Dihedral>() {
        2
    } else {
        255
    };
    (kind, coord.idx())
}

fn carry_over_hessian(
    old_coords: &[Box<dyn InternalCoord>],
    old_h: &DMatrix<f64>,
    new_coords: &[Box<dyn InternalCoord>],
    guess_h: &DMatrix<f64>,
) -> DMatrix<f64> {
    let old_pos: HashMap<(u8, Vec<usize>), usize> = old_coords
        .iter()
        .enumerate()
        .map(|(i, coord)| (coord_key(coord.as_ref()), i))
        .collect();

    let pairs: Vec<(usize, usize)> = new_coords
        .iter()
        .enumerate()
        .filter_map(|(new_i, coord)| old_pos.get(&coord_key(coord.as_ref())).map(|&old_i| (new_i, old_i)))
        .collect();

    let mut h = guess_h.clone();
    for &(new_i, old_i) in &pairs {
        for &(new_j, old_j) in &pairs {
            h[(new_i, new_j)] = old_h[(old_i, old_j)];
        }
    }
    h
}

/// Enforce exactly one negative eigenvalue on the Hessian by spectral
/// adjustment.  Used periodically during TS searches to prevent the
/// quasi-Newton update from drifting away from the saddle-point signature.
///
/// The smallest eigenvalue is pushed to `-epsilon` if not already negative.
/// Any extra negative eigenvalues are flipped to `+epsilon`.
fn ensure_one_negative_eigenvalue(h: &mut DMatrix<f64>, epsilon: f64) {
    let eig = h.clone().symmetric_eigen();
    let n = eig.eigenvalues.len();
    if n == 0 {
        return;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        eig.eigenvalues[a]
            .partial_cmp(&eig.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_neg = eig.eigenvalues.iter().filter(|&&v| v < 0.0).count();

    if n_neg != 1 || eig.eigenvalues[idx[0]] > -(epsilon / 2.0) {
        // Ensure lowest eigenvalue is at most -epsilon.
        let i0 = idx[0];
        let target = -epsilon;
        if eig.eigenvalues[i0] > target {
            let v = eig.eigenvectors.column(i0);
            *h += (target - eig.eigenvalues[i0]) * v * v.transpose();
        }
    }

    for i in &idx[1..] {
        if eig.eigenvalues[*i] < epsilon / 2.0 {
            let v = eig.eigenvectors.column(*i);
            *h += (epsilon - eig.eigenvalues[*i]) * v * v.transpose();
        }
    }
}

/// Performs a 1-D linear search between `current` (t=0) and `best` (t=1) by
/// fitting quartic then cubic through the two energy values and their 1-D
/// directional derivatives.`.
fn linear_search(e0: f64, e1: f64, g0: f64, g1: f64) -> (f64, f64) {
    // Try quartic fit first
    if let (Some(t), Some(e)) = math::fit_quartic(e0, e1, g0, g1) {
        if t >= -1.0 && t <= 2.0 {
            return (t, e);
        }
    }
    // Fall back to cubic
    if let (Some(t), Some(e)) = math::fit_cubic(e0, e1, g0, g1) {
        if t >= 0.0 && t <= 1.0 {
            return (t, e);
        }
    }
    // Final fallback: pick whichever endpoint is lower
    if e0 <= e1 {
        (0.0, e0)
    } else {
        (1.0, e1)
    }
}

/// Checks convergence against the full set of criteria.
///
/// thresholds (Berny defaults):
/// - `gradientrms`  = 0.15e-3
/// - `gradientmax`  = 0.45e-3
/// - `steprms`      = 1.2e-3
/// - `stepmax`      = 1.8e-3
/// - For TS mode: exactly one negative Hessian eigenvalue required.
///
/// When the step is on the trust sphere (`on_sphere == true`) only the
/// gradient criteria are required to declare convergence (the step length
/// tells us we are still constrained, not that we have converged).
fn is_converged(
    gradient: &DVector<f64>,
    step: &DVector<f64>,
    on_sphere: bool,
    params: &BernyParams,
    n_negative_eigenvalues: Option<usize>,
) -> bool {
    let g_rms = math::rms_vec(gradient).unwrap_or(f64::INFINITY);
    let g_max = gradient.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let s_rms = math::rms_vec(step).unwrap_or(f64::INFINITY);
    let s_max = step.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    let grad_ok = g_rms < params.gradient_rms && g_max < params.gradient_max;
    let step_ok = on_sphere || (s_rms < params.step_rms && s_max < params.step_max);

    // For TS the projected Hessian must have exactly one negative eigenvalue.
    let neg_ok = match n_negative_eigenvalues {
        Some(n) => n == 1,
        None => true,
    };

    grad_ok && step_ok && neg_ok
}
