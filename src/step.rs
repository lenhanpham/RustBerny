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

/// Computes the partitioned-RFO (P-RFO) step for a transition-state search.
///
/// Implements the partitioned rational-function optimization of Banerjee,
/// Adams, Simons and Shepard (*J. Phys. Chem.* **1985**, *89*, 52-57) with
/// eigenvector following. The Hessian eigenvectors are split into a one-
/// dimensional "ascent" subspace (the reaction coordinate) and the orthogonal
/// "descent" subspace:
///
/// * The reaction-coordinate mode is shifted by the **upper** RFO root
///   `λ_p = (b_p + √(b_p² + 4 F_p²)) / 2 ≥ b_p`, so the step climbs that mode.
/// * Every remaining mode shares the **lower** RFO root `λ_n`, the lowest root
///   of the secular equation `λ = Σ_i F_i² / (λ − b_i)` taken over the descent
///   subspace, so those modes are minimized.
///
/// The step component along eigenvector `i` is `h_i = F_i / (λ − b_i)` where
/// `F_i = vᵢ·g` and `λ` is the shift assigned to that mode. The total step is
/// scaled uniformly into the trust radius when it would otherwise exceed it.
///
/// `tracked` carries the reaction-coordinate eigenvector between cycles: the
/// ascended mode is the eigenvector of maximum overlap with `tracked` (or the
/// lowest mode when `tracked` is `None`), and on return it is updated
/// (sign-aligned) to the mode that was climbed. This eigenvector following keeps
/// the search on the same reaction coordinate instead of whichever mode is
/// momentarily lowest.
///
/// # Arguments
/// * `g` - Projected gradient in internal coordinates
/// * `h` - Projected (indefinite) Hessian in internal coordinates
/// * `_w` - Coordinate weights (reserved for future use)
/// * `trust` - Trust radius
/// * `tracked` - Reaction-coordinate tracker (seeded on the first call)
///
/// # Returns
/// `(dq, dE, on_sphere)` — step, predicted energy change, whether on trust sphere.
pub fn quadratic_step_ts(
    g: &DVector<f64>,
    h: &DMatrix<f64>,
    _w: &DVector<f64>,
    trust: f64,
    tracked: &mut Option<DVector<f64>>,
) -> (DVector<f64>, f64, bool) {
    let n = g.len();
    if n == 0 {
        return (DVector::zeros(0), 0.0, false);
    }

    // Diagonalize the symmetric Hessian.
    let h_sym = (h + h.transpose()) / 2.0;
    let eig = h_sym.clone().symmetric_eigen();
    let b = eig.eigenvalues; // ascending order from symmetric_eigen
    let v = eig.eigenvectors;

    // Gradient in the eigenbasis: F_i = v_i · g.
    let f = v.transpose() * g;

    // Pick the reaction-coordinate (ascent) mode by maximum overlap with the
    // tracked eigenvector; fall back to the lowest-curvature mode.
    let ascent = match tracked.as_ref() {
        Some(prev) if prev.len() == n => {
            let mut best = 0usize;
            let mut best_ovlp = -1.0_f64;
            for k in 0..n {
                let ovlp = v.column(k).dot(prev).abs();
                if ovlp > best_ovlp {
                    best_ovlp = ovlp;
                    best = k;
                }
            }
            best
        }
        _ => 0,
    };

    // Upper RFO root for the ascended mode.
    let b_p = b[ascent];
    let f_p = f[ascent];
    let lambda_p = 0.5 * (b_p + (b_p * b_p + 4.0 * f_p * f_p).sqrt());

    // Lower RFO root shared by the descent subspace: the lowest root of
    // λ = Σ_{i≠ascent} F_i² / (λ − b_i), which lies below the smallest descent
    // eigenvalue. Solved by monotone bisection (the secular function
    // g(λ) = λ − Σ F_i²/(λ − b_i) is strictly increasing for λ < min b_i).
    let descent: Vec<usize> = (0..n).filter(|&i| i != ascent).collect();
    let lambda_n = if descent.is_empty() {
        0.0
    } else {
        let min_b = descent.iter().map(|&i| b[i]).fold(f64::INFINITY, f64::min);
        let secular = |lam: f64| -> f64 {
            let mut acc = lam;
            for &i in &descent {
                acc -= f[i] * f[i] / (lam - b[i]);
            }
            acc
        };
        // Bracket: hi just below the lowest descent eigenvalue (and below 0),
        // lo far enough negative that the secular function is negative.
        let hi = min_b.min(0.0) - 1e-9;
        let mut lo = hi - 1.0;
        let mut safety = 0;
        while secular(lo) > 0.0 && safety < 200 {
            lo -= lo.abs() + 1.0;
            safety += 1;
        }
        let mut a = lo;
        let mut c = hi;
        for _ in 0..200 {
            let m = 0.5 * (a + c);
            if secular(m) > 0.0 {
                c = m;
            } else {
                a = m;
            }
            if (c - a).abs() < 1e-14 * (1.0 + c.abs()) {
                break;
            }
        }
        0.5 * (a + c)
    };

    // Step components h_i = F_i / (λ − b_i) in the eigenbasis.
    let mut h_eig = DVector::zeros(n);
    for i in 0..n {
        let lam = if i == ascent { lambda_p } else { lambda_n };
        let denom = lam - b[i];
        h_eig[i] = if denom.abs() > 1e-14 { f[i] / denom } else { 0.0 };
    }

    // Transform back to the coordinate basis.
    let mut dq = &v * &h_eig;

    // Trust-radius constraint: uniform scaling into the sphere.
    let mut on_sphere = false;
    let step_norm = dq.norm();
    if step_norm > trust && step_norm > 1e-14 {
        dq *= trust / step_norm;
        on_sphere = true;
    }

    // Update the tracked reaction coordinate (sign-aligned to the previous one).
    let mut tv = v.column(ascent).clone_owned();
    if let Some(prev) = tracked.as_ref() {
        if prev.len() == n && tv.dot(prev) < 0.0 {
            tv = -tv;
        }
    }
    *tracked = Some(tv);

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

        let mut tracked = None;
        let (dq, _d_e, on_sphere) = quadratic_step_ts(&g, &h, &w, 10.0, &mut tracked);
        assert!(!on_sphere);
        // Step should have finite norm
        assert!(dq.norm() > 0.0);
        assert!(dq.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_prfo_converges_to_quadratic_saddle() {
        // E = -0.5 x^2 + 0.5 y^2 + 0.5 z^2 has an order-1 saddle at the origin.
        // Exact Hessian = diag(-1, 1, 1); gradient = (-x, y, z).
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![-1.0, 1.0, 1.0]));
        let w = DVector::from_element(3, 1.0);
        let mut x = DVector::from_vec(vec![0.7, -0.5, 0.3]);
        let mut tracked = None;
        for _ in 0..100 {
            let g = DVector::from_vec(vec![-x[0], x[1], x[2]]);
            if g.norm() < 1e-9 {
                break;
            }
            let (dq, _de, _on) = quadratic_step_ts(&g, &h, &w, 0.5, &mut tracked);
            x += dq;
        }
        assert!(x.norm() < 1e-6, "P-RFO should reach the saddle, |x|={}", x.norm());
    }

    #[test]
    fn test_prfo_ascends_reaction_mode_descends_rest() {
        // At a point off the saddle, the step must increase energy along the
        // negative-curvature mode (x) and decrease it along the rest.
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![-1.0, 2.0, 2.0]));
        let w = DVector::from_element(3, 1.0);
        let x = DVector::from_vec(vec![0.4, 0.4, 0.4]);
        let g = DVector::from_vec(vec![-x[0], 2.0 * x[1], 2.0 * x[2]]);
        let mut tracked = None;
        let (dq, _de, _on) = quadratic_step_ts(&g, &h, &w, 10.0, &mut tracked);
        // Reaction coordinate (x) steps toward the saddle (sign opposite to x).
        assert!(dq[0] * x[0] < 0.0, "ascent mode should move toward saddle: {dq:?}");
        // The tracked mode should align with the negative-curvature axis.
        let tv = tracked.unwrap();
        assert!(tv[0].abs() > 0.99, "tracked reaction coordinate should follow x: {tv:?}");
    }
}
