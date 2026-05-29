//! Hessian update methods for geometry optimization.
//!
//! Provides BFGS for minimization and flowchart SR1/BFGS/PSB for transition state search.

use nalgebra::{DMatrix, DVector};

/// Updates the Hessian using BFGS (minimization mode).
///
/// Skips the update if the curvature condition `dq · dg ≤ 0`.
///
/// # Arguments
/// * `h` - Current Hessian, shape (n, n)
/// * `dq` - Step in internal coordinates
/// * `dg` - Gradient difference
///
/// # Returns
/// Updated Hessian, or the original if update was skipped.
pub fn update_bfgs(h: &DMatrix<f64>, dq: &DVector<f64>, dg: &DVector<f64>) -> DMatrix<f64> {
    let yts = dq.dot(dg);
    if yts <= 0.0 {
        return h.clone();
    }
    let d_h1 = dg * dg.transpose() / yts;
    let h_dq = h * dq;
    let d_h2 = &h_dq * &h_dq.transpose() / dq.dot(&h_dq);
    h + d_h1 - d_h2
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfgs_skip_on_negative_curvature() {
        let h = DMatrix::identity(2, 2);
        let dq = DVector::from_vec(vec![1.0, 0.0]);
        let dg = DVector::from_vec(vec![-1.0, 0.0]); // dq·dg = -1 < 0
        let h_new = update_bfgs(&h, &dq, &dg);
        assert_eq!(h_new, h); // Skipped
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
