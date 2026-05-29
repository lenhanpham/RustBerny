//! Quadratic step computation for geometry optimization.
//!
//! Provides RFO for minimization and P-RFO for transition state search.

use crate::math;
use nalgebra::{DMatrix, DVector};

/// Computes the RFO step for minimization.
///
/// # Arguments
/// * `g` - Gradient in internal coordinates
/// * `h` - Hessian in internal coordinates
/// * `_w` - Coordinate weights (reserved for future use)
/// * `trust` - Trust radius
///
/// # Returns
/// `(dq, dE, on_sphere)` — step, predicted energy change, whether on trust sphere.
pub fn quadratic_step(
    g: &DVector<f64>,
    h: &DMatrix<f64>,
    _w: &DVector<f64>,
    trust: f64,
) -> (DVector<f64>, f64, bool) {
    let n = h.nrows();

    // Symmetrize H
    let h_sym = (h + h.transpose()) / 2.0;
    let ev = h_sym.clone().symmetric_eigen().eigenvalues;

    // Build RFO matrix: [[H, g], [g^T, 0]]
    let mut rfo = DMatrix::zeros(n + 1, n + 1);
    for i in 0..n {
        for j in 0..n {
            rfo[(i, j)] = h_sym[(i, j)];
        }
        rfo[(i, n)] = g[i];
        rfo[(n, i)] = g[i];
    }

    // Eigendecompose RFO matrix
    let rfo_sym = (&rfo + &rfo.transpose()) / 2.0;
    let eig = rfo_sym.symmetric_eigen();
    let _d = eig.eigenvalues;
    let v = eig.eigenvectors;

    // Step from lowest eigenvector
    let dq = v.column(0).rows(0, n).clone_owned() / v[(n, 0)];

    let mut on_sphere = false;
    let mut dq_out = dq.clone();

    if dq.norm() <= trust {
        // Pure RFO step
    } else {
        // Constrained step: find λ < ev[0]
        let ev0 = ev[0];
        let steplength = |l: f64| -> f64 {
            let shifted = l * DMatrix::identity(n, n) - &h_sym;
            let solved = shifted.try_inverse().unwrap() * g;
            solved.norm() - trust
        };

        match math::findroot(steplength, ev0) {
            Ok(l) => {
                let shifted = l * DMatrix::identity(n, n) - &h_sym;
                dq_out = shifted.try_inverse().unwrap() * g;
                on_sphere = true;
            }
            Err(_) => {
                // Fallback: uniform rescaling
                dq_out = &dq * (trust / dq.norm());
                on_sphere = true;
            }
        }
    }

    let d_e = g.dot(&dq_out) + 0.5 * dq_out.dot(&(&h_sym * &dq_out));
    (dq_out, d_e, on_sphere)
}

/// Computes the P-RFO step for transition state search.
///
/// Maximizes along the transition vector (lowest eigenvector of H) and
/// minimizes in the orthogonal complement.
///
/// # Arguments
/// * `g` - Projected gradient in internal coordinates
/// * `h` - Projected Hessian in internal coordinates
/// * `_w` - Coordinate weights
/// * `trust` - Trust radius
///
/// # Returns
/// `(dq, dE, on_sphere)` — step, predicted energy change, whether on trust sphere.
pub fn quadratic_step_ts(
    g: &DVector<f64>,
    h: &DMatrix<f64>,
    _w: &DVector<f64>,
    trust: f64,
) -> (DVector<f64>, f64, bool) {
    let n = g.len();

    // Eigendecompose H → ev ascending, V columns
    let h_sym = (h + h.transpose()) / 2.0;
    let eig = h_sym.clone().symmetric_eigen();
    let ev = eig.eigenvalues;
    let v = eig.eigenvectors;

    // Transition vector (lowest eigenvector)
    let v0 = v.column(0).clone_owned();
    let v_rest = v.columns(1, n - 1).clone_owned();

    // --- Uphill RFO along transition vector ---
    let g0 = v0.dot(g);
    let mut rfo_ts = DMatrix::zeros(2, 2);
    rfo_ts[(0, 0)] = ev[0];
    rfo_ts[(0, 1)] = g0;
    rfo_ts[(1, 0)] = g0;
    let rfo_ts_sym = (&rfo_ts + &rfo_ts.transpose()) / 2.0;
    let eig_ts = rfo_ts_sym.symmetric_eigen();
    let t0 = eig_ts.eigenvectors[(0, 1)] / eig_ts.eigenvectors[(1, 1)];

    // --- Downhill RFO in orthogonal complement ---
    let g_rest = v_rest.transpose() * g;
    let mut rfo_rest = DMatrix::zeros(n, n);
    for i in 0..(n - 1) {
        rfo_rest[(i, i)] = ev[i + 1];
    }
    for i in 0..(n - 1) {
        rfo_rest[(i, n - 1)] = g_rest[i];
        rfo_rest[(n - 1, i)] = g_rest[i];
    }
    let rfo_rest_sym = (&rfo_rest + &rfo_rest.transpose()) / 2.0;
    let eig_rest = rfo_rest_sym.symmetric_eigen();
    let t_rest = eig_rest.eigenvectors.column(0).rows(0, n - 1).clone_owned()
        / eig_rest.eigenvectors[(n - 1, 0)];

    // Assemble full step
    let mut dq = &v0 * t0 + &v_rest * &t_rest;
    let step_norm = dq.norm();

    let mut on_sphere = false;

    // Trust radius constraint
    if step_norm > trust {
        let ev_rest = ev.rows(1, n - 1).clone_owned();
        let ev_rest_shifted = ev_rest.clone();
        let steplength = |l: f64| -> f64 {
            let t0_l = g0 / (ev[0] - l);
            let mut dq_l = &v0 * t0_l;
            if n > 1 {
                let shifted = DVector::from_fn(n - 1, |i, _| ev_rest_shifted[i] - l);
                let t_rest_l = g_rest.clone().component_div(&shifted);
                dq_l += &v_rest * &t_rest_l;
            }
            dq_l.norm() - trust
        };

        match math::findroot(steplength, ev[0]) {
            Ok(l_opt) => {
                let t0_s = g0 / (ev[0] - l_opt);
                dq = &v0 * t0_s;
                if n > 1 {
                    // MINUS SIGN critical: downhill opposes gradient
                    let shifted = DVector::from_fn(n - 1, |i, _| ev_rest[i] - l_opt);
                    let t_rest_s = &g_rest.component_div(&shifted) * -1.0;
                    dq += &v_rest * &t_rest_s;
                }
                on_sphere = true;
            }
            Err(_) => {
                // Fallback: uniform rescaling
                dq = &dq * (trust / step_norm);
                on_sphere = true;
            }
        }
    }

    let d_e = g.dot(&dq) + 0.5 * dq.dot(&(&h_sym * &dq));
    (dq, d_e, on_sphere)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rfo_unconstrained() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![0.5, 0.8]));
        let g = DVector::from_vec(vec![-0.1, -0.2]);
        let w = DVector::from_vec(vec![1.0, 1.0]);
        let (dq, _d_e, on_sphere) = quadratic_step(&g, &h, &w, 10.0);
        assert!(!on_sphere);
        assert!(dq.norm() > 0.0);
    }

    #[test]
    fn test_prfo_2d() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![-0.5, 0.8]));
        let g = DVector::from_vec(vec![0.1, 0.2]);
        let w = DVector::from_vec(vec![1.0, 1.0]);

        let (dq, _d_e, on_sphere) = quadratic_step_ts(&g, &h, &w, 10.0);
        assert!(!on_sphere);
        // Step should have finite norm
        assert!(dq.norm() > 0.0);
    }
}
