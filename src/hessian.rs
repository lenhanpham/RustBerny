//! Hessian update methods for geometry optimization.
//!
//! Provides BFGS with PSB fallback for minimization and flowchart SR1/BFGS/PSB
//! for transition state search. Also provides [`force_positive_definite`] for
//! spectral projection.

use nalgebra::{DMatrix, DVector};

/// Updates the Hessian using BFGS with PSB fallback (minimization mode).
///
/// Uses BFGS when the curvature condition `dq · dg > 0` is satisfied.
/// Falls back to PSB when the condition is violated, ensuring the Hessian
/// always receives some information from the secant pair.
///
/// # Arguments
/// * `h` - Current Hessian, shape (n, n)
/// * `dq` - Step in internal coordinates
/// * `dg` - Gradient difference
///
/// # Returns
/// Updated Hessian.
pub fn update_bfgs(h: &DMatrix<f64>, dq: &DVector<f64>, dg: &DVector<f64>) -> DMatrix<f64> {
    let yts = dq.dot(dg);
    if yts > 0.0 {
        // Standard BFGS
        let d_h1 = dg * dg.transpose() / yts;
        let h_dq = h * dq;
        let d_h2 = &h_dq * &h_dq.transpose() / dq.dot(&h_dq);
        h + d_h1 - d_h2
    } else {
        // PSB fallback when curvature condition is not satisfied
        let z = dg - h * dq;
        let sts = dq.dot(dq);
        if sts < 1e-20 {
            return h.clone();
        }
        let sz = dq.dot(&z);
        let d_h = (dq * z.transpose() + z * dq.transpose()) / sts
            - sz * dq * dq.transpose() / (sts * sts);
        h + d_h
    }
}

/// Updates the Hessian using flowchart SR1/BFGS/PSB (transition state mode).
///
/// Selects the appropriate method based on normalised cosine criteria:
/// - SR1 when `z·s / (|z||s|) < -0.1`
/// - BFGS when `y·s / (|y||s|) > +0.1`
/// - PSB otherwise
///
/// # Arguments
/// * `h` - Current Hessian
/// * `dq` - Step in internal coordinates
/// * `dg` - Gradient difference
pub fn update_hessian_ts(h: &DMatrix<f64>, dq: &DVector<f64>, dg: &DVector<f64>) -> DMatrix<f64> {
    let s = dq;
    let y = dg;
    let z = y - h * s;

    let sts = s.dot(s);
    if sts < 1e-20 {
        return h.clone();
    }

    let norm_z = z.norm();
    let norm_s = s.norm();
    let norm_y = y.norm();

    let zts_ratio = z.dot(s) / (norm_z * norm_s + 1e-30);
    let yts_ratio = y.dot(s) / (norm_y * norm_s + 1e-30);

    let d_h = if zts_ratio < -0.1 {
        // SR1
        let zts = z.dot(s);
        if zts.abs() < 1e-20 {
            return h.clone();
        }
        z.clone() * z.transpose() / zts
    } else if yts_ratio > 0.1 {
        // BFGS
        let yts = y.dot(s);
        let h_s = h * s;
        y.clone() * y.transpose() / yts - h_s.clone() * h_s.transpose() / s.dot(&h_s)
    } else {
        // PSB
        let sz = s.dot(&z);
        (s.clone() * z.transpose() + z.clone() * s.transpose()) / sts
            - sz * s.clone() * s.transpose() / (sts * sts)
    };

    h + d_h
}

/// Shift the eigenspectrum of `h` so that all eigenvalues are at least
/// `min_eigenvalue`.
///
/// Uses symmetric eigendecomposition; the result is symmetric by construction.
pub fn force_positive_definite(h: &DMatrix<f64>, min_eigenvalue: f64) -> DMatrix<f64> {
    let sym = (h + h.transpose()) * 0.5;
    let eig = sym.symmetric_eigen();
    let mut vals = eig.eigenvalues.clone();
    for v in vals.iter_mut() {
        if *v < min_eigenvalue {
            *v = min_eigenvalue;
        }
    }
    let d = DMatrix::from_diagonal(&vals);
    &eig.eigenvectors * d * eig.eigenvectors.transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfgs_psb_fallback_on_negative_curvature() {
        let h = DMatrix::identity(2, 2);
        let dq = DVector::from_vec(vec![1.0, 0.0]);
        let dg = DVector::from_vec(vec![-1.0, 0.0]); // dq·dg = -1 < 0
        let h_new = update_bfgs(&h, &dq, &dg);
        // PSB fallback should produce a change (not just return h)
        let diff = (&h_new - &h).norm();
        assert!(diff > 1e-10, "PSB fallback produced no change: diff={diff}");
    }

    #[test]
    fn test_force_positive_definite() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![-0.5, 0.3, 2.0]));
        let h_pd = force_positive_definite(&h, 1e-2);
        let eig = h_pd.symmetric_eigen();
        for v in eig.eigenvalues.iter() {
            assert!(*v >= 1e-2 - 1e-12, "eigenvalue {v} below floor");
        }
    }

    #[test]
    fn test_bfgs_update() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
        let dq = DVector::from_vec(vec![0.5, 0.0]);
        let dg = DVector::from_vec(vec![0.3, 0.1]); // dq·dg = 0.15 > 0
        let h_new = update_bfgs(&h, &dq, &dg);
        // BFGS should update the Hessian
        let diff = (&h_new - &h).norm();
        assert!(diff > 1e-10, "BFGS produced no change: diff={diff}");
    }

    #[test]
    fn test_ts_update_skip_on_zero_displacement() {
        let h = DMatrix::identity(2, 2);
        let dq = DVector::from_vec(vec![0.0, 0.0]);
        let dg = DVector::from_vec(vec![1.0, 0.0]);
        let h_new = update_hessian_ts(&h, &dq, &dg);
        assert_eq!(h_new, h); // Skipped
    }
}
