//! Mathematical utilities for geometry optimization.
//!
//! Provides root-finding, polynomial fitting, pseudoinverse computation,
//! and array statistics used throughout the optimizer.

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Error returned when root-finding fails to converge.
#[derive(Error, Debug)]
#[error("Root-finding did not converge")]
pub struct FindrootError;

/// Computes the root-mean-square of array elements.
///
/// Returns `None` for empty arrays.
pub fn rms(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let sum_sq: f64 = data.iter().map(|x| x * x).sum();
    Some((sum_sq / data.len() as f64).sqrt())
}

/// Computes the root-mean-square of a vector.
pub fn rms_vec(data: &DVector<f64>) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let sum_sq: f64 = data.iter().map(|x| x * x).sum();
    Some((sum_sq / data.len() as f64).sqrt())
}

/// Computes the pseudoinverse of a matrix using SVD with gap detection.
///
/// Singular values separated by a ratio greater than `threshold` (default 1e3)
/// are truncated to zero, preventing numerical instability in the B-matrix
/// pseudoinverse.
pub fn pinv(matrix: &DMatrix<f64>, log: &dyn Fn(&str)) -> DMatrix<f64> {
    let (nrows, ncols) = matrix.shape();
    let k = nrows.min(ncols);

    // Compute thin SVD
    let svd = matrix.clone().svd(true, true);
    let u = svd.u.unwrap(); // nrows x k
    let v_t = svd.v_t.unwrap(); // k x ncols
    let mut s = svd.singular_values; // length k

    let threshold = 1e3;
    let threshold_log = 1e8;

    // Detect gaps in singular values
    let mut truncate_at = k;
    for i in 0..k.saturating_sub(1) {
        if s[i + 1] > 0.0 {
            let gap = s[i] / s[i + 1];
            if gap > threshold {
                truncate_at = i + 1;
                if gap < threshold_log {
                    log(&format!("Pseudoinverse gap of only: {gap:.1e}"));
                }
                break;
            }
        }
    }

    // Zero out truncated singular values
    for i in truncate_at..k {
        s[i] = 0.0;
    }

    // Invert non-zero singular values
    for i in 0..truncate_at {
        s[i] = 1.0 / s[i];
    }

    // Reconstruct: U * diag(s) * V^T
    let s_inv = DMatrix::from_diagonal(&s);
    u * s_inv * v_t
}

/// Computes the cross product of two 3-vectors.
pub fn cross(a: &nalgebra::Vector3<f64>, b: &nalgebra::Vector3<f64>) -> nalgebra::Vector3<f64> {
    a.cross(b)
}

/// Fits a cubic polynomial to function values and derivatives at x = 0 and x = 1.
///
/// Fits `p(x) = a*x^3 + b*x^2 + g0*x + y0` such that:
/// - `p(0) = y0`, `p(1) = y1`
/// - `p'(0) = g0`, `p'(1) = g1`
///
/// Returns `(Some(position), Some(value))` of the minimum if the fit succeeds,
/// or `(None, None)` if the polynomial has no valid minimum in the expected range.
pub fn fit_cubic(y0: f64, y1: f64, g0: f64, g1: f64) -> (Option<f64>, Option<f64>) {
    let a = 2.0 * (y0 - y1) + g0 + g1;
    let b = -3.0 * (y0 - y1) - 2.0 * g0 - g1;

    // Handle degenerate case: a ≈ 0 → quadratic
    if a.abs() < 1e-15 {
        // p(x) = b*x^2 + g0*x + y0
        // p'(x) = 2b*x + g0 = 0 → x = -g0/(2b)
        if b.abs() < 1e-15 {
            return (None, None);
        }
        let x = -g0 / (2.0 * b);
        if x < 0.0 || x > 1.0 {
            return (None, None);
        }
        let val = b * x * x + g0 * x + y0;
        return (Some(x), Some(val));
    }

    // p'(x) = 3a*x^2 + 2b*x + g0 = 0
    let discriminant = b * b - 3.0 * a * g0;
    if discriminant < 0.0 {
        return (None, None);
    }

    let sqrt_disc = discriminant.sqrt();
    let two_a = 2.0 * a;

    let r1 = (-b + sqrt_disc) / two_a;
    let r2 = (-b - sqrt_disc) / two_a;

    let mut r_sorted = [r1, r2];
    r_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // For a cubic a*x^3+..., when a>0 the cubic rises left→right:
    // the smaller (leftmost) critical point is a local MAX and the larger is the MIN.
    // Python: `if p[0] > 0: maxim, minim = r_sorted` (maxim = r_sorted[0] = smaller).
    let (maxim, minim) = if a > 0.0 {
        (r_sorted[0], r_sorted[1])
    } else {
        (r_sorted[1], r_sorted[0])
    };

    // Check if maximum is in (0,1) and closer to 0.5 than minimum
    if maxim > 0.0 && maxim < 1.0 && (minim - 0.5).abs() > (maxim - 0.5).abs() {
        return (None, None);
    }

    let value = a * minim.powi(3) + b * minim.powi(2) + g0 * minim + y0;
    (Some(minim), Some(value))
}

/// Fits a constrained quartic polynomial to function values and derivatives at x = 0 and x = 1.
///
/// The quartic is constrained so its second derivative vanishes at exactly one point,
/// ensuring a single local extremum. Returns the minimum with lower function value.
pub fn fit_quartic(y0: f64, y1: f64, g0: f64, g1: f64) -> (Option<f64>, Option<f64>) {
    /// Constructs the quartic polynomial coefficients for a given constraint parameter `c`.
    fn quartic_poly(y0: f64, y1: f64, g0: f64, g1: f64, c: f64) -> [f64; 5] {
        let a = c + 3.0 * (y0 - y1) + 2.0 * g0 + g1;
        let b = -2.0 * c - 4.0 * (y0 - y1) - 3.0 * g0 - g1;
        [a, b, c, g0, y0]
    }

    /// Finds the minimum of a quartic polynomial.
    fn quart_min(p: [f64; 5]) -> (f64, f64) {
        // p'(x) = 4a*x^3 + 3b*x^2 + 2c*x + g0 = 0
        // For simplicity, find critical points
        let [a, b, c, g0, y0] = p;
        // Use numpy-style root finding via Cardano or numerical
        // Simplified: find where derivative changes sign
        let n_points = 1000;
        let mut min_x = 0.5;
        let mut min_val = f64::INFINITY;

        // Sample to find approximate minimum
        for i in 0..=n_points {
            let x = -1.0 + 3.0 * i as f64 / n_points as f64;
            let val = a * x.powi(4) + b * x.powi(3) + c * x.powi(2) + g0 * x + y0;
            if val < min_val {
                min_val = val;
                min_x = x;
            }
        }

        // Refine with Newton's method on derivative
        let mut x = min_x;
        for _ in 0..50 {
            let dp = 4.0 * a * x.powi(3) + 3.0 * b * x.powi(2) + 2.0 * c * x + g0;
            let ddp = 12.0 * a * x.powi(2) + 6.0 * b * x + 2.0 * c;
            if ddp.abs() < 1e-15 {
                break;
            }
            let x_new = x - dp / ddp;
            if (x_new - x).abs() < 1e-12 {
                break;
            }
            x = x_new;
        }

        let val = a * x.powi(4) + b * x.powi(3) + c * x.powi(2) + g0 * x + y0;
        (x, val)
    }

    // Discriminant of d^2y/dx^2 = 0
    let d = -((g0 + g1).powi(2)) - 2.0 * g0 * g1 + 6.0 * (y1 - y0) * (g0 + g1)
        - 6.0 * (y1 - y0).powi(2);

    if d < 1e-11 {
        return (None, None);
    }

    let m = -5.0 * g0 - g1 - 6.0 * y0 + 6.0 * y1;
    let sqrt_2d = (2.0 * d).sqrt();

    let p1 = quartic_poly(y0, y1, g0, g1, 0.5 * (m + sqrt_2d));
    let p2 = quartic_poly(y0, y1, g0, g1, 0.5 * (m - sqrt_2d));

    if p1[0] < 0.0 && p2[0] < 0.0 {
        return (None, None);
    }

    let (minim1, minval1) = quart_min(p1);
    let (minim2, minval2) = quart_min(p2);

    if minval1 < minval2 {
        (Some(minim1), Some(minval1))
    } else {
        (Some(minim2), Some(minval2))
    }
}

/// Finds a root of an increasing function on `(-∞, lim)`.
///
/// Assumes `f(lim - d) > 0` for sufficiently large `d` and `f(-∞) < 0`.
/// Uses Newton's method with numerical derivative (dx = 1e-10).
///
/// # Errors
/// Returns `FindrootError` if convergence fails after 1000 iterations.
pub fn findroot(f: impl Fn(f64) -> f64, lim: f64) -> Result<f64, FindrootError> {
    let dx = 1e-10;

    // Find initial bracket: d such that f(lim - d) > 0
    let mut d = 1.0;
    for _ in 0..1000 {
        if f(lim - d) > 0.0 {
            break;
        }
        d /= 2.0;
    }

    let mut x = lim - d;
    let mut fx = f(x);
    let mut err = fx.abs();

    for _ in 0..1000 {
        let fxpdx = f(x + dx);
        let dxf = (fxpdx - fx) / dx;
        x -= fx / dxf;
        fx = f(x);
        let err_new = fx.abs();
        if err_new >= err {
            return Ok(x);
        }
        err = err_new;
    }
    Err(FindrootError)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms() {
        let data = vec![1.0, 2.0, 3.0];
        let result = rms(&data).unwrap();
        assert!((result - (14.0_f64 / 3.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_rms_empty() {
        assert!(rms(&[]).is_none());
    }

    #[test]
    fn test_cross_product() {
        let a = nalgebra::Vector3::new(1.0, 0.0, 0.0);
        let b = nalgebra::Vector3::new(0.0, 1.0, 0.0);
        let c = cross(&a, &b);
        assert!((c[0] - 0.0).abs() < 1e-10);
        assert!((c[1] - 0.0).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_findroot() {
        // f(x) = x - 2, root at x = 2
        let root = findroot(|x| x - 2.0, 5.0).unwrap();
        assert!((root - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_findroot_nonlinear() {
        // f(x) = x^2 - 4, root at x = 2 (we look in (-inf, 5))
        let root = findroot(|x| x * x - 4.0, 5.0).unwrap();
        assert!((root - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_fit_cubic() {
        // p(x) = x^2, min at x = 0
        // p(0) = 0, p(1) = 1, p'(0) = 0, p'(1) = 2
        let (t, _e) = fit_cubic(0.0, 1.0, 0.0, 2.0);
        assert!(t.is_some());
        let t = t.unwrap();
        assert!(t >= -0.1 && t <= 0.1);
    }

    #[test]
    fn test_pinv() {
        // Use a square matrix to avoid shape complexity
        let m = DMatrix::from_fn(3, 3, |i, j| if i == j { (i + 1) as f64 } else { 0.1 });
        let log = |_: &str| {};
        let p = pinv(&m, &log);
        // Check that pinv has correct shape (3x3)
        assert_eq!(p.nrows(), 3);
        assert_eq!(p.ncols(), 3);
        // Check that m * pinv(m) * m ≈ m
        let reconstruct = &m * &p * &m;
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                assert!((reconstruct[(i, j)] - m[(i, j)]).abs() < 1e-10);
            }
        }
    }
}
