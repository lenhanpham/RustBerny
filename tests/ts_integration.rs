//! End-to-end transition-state search on analytic surfaces.
//!
//! These tests drive the full `Berny` optimizer (B-matrix, internal-coordinate
//! gradient transform, P-RFO step, geometry update, convergence) in TS mode and
//! confirm it climbs to a known stationary point that carries one negative
//! Hessian eigenvalue.

use rustberny::core::{Berny, BernyParams, Mode};
use rustberny::geometry::Geometry;

/// Cartesian gradient of `E(r) = 0.5 * k * (r - r0)^2` for a diatomic, where
/// `r` is the bond length. With `k < 0` the potential has a maximum (an
/// order-1 saddle along the single internal coordinate) at `r = r0`, so a
/// transition-state search must drive the bond length to `r0`.
fn diatomic_eg(coords: &[[f64; 3]], k: f64, r0: f64) -> (f64, Vec<Vec<f64>>) {
    let a = coords[0];
    let b = coords[1];
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let e = 0.5 * k * (r - r0) * (r - r0);
    let de_dr = k * (r - r0);
    let u = [d[0] / r, d[1] / r, d[2] / r];
    // dr/dB = u, dr/dA = -u
    let g = vec![
        vec![-de_dr * u[0], -de_dr * u[1], -de_dr * u[2]],
        vec![de_dr * u[0], de_dr * u[1], de_dr * u[2]],
    ];
    (e, g)
}

fn bond_length(geom: &Geometry) -> f64 {
    let a = geom.coords[0];
    let b = geom.coords[1];
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
}

#[test]
fn ts_search_climbs_diatomic_bond_maximum() {
    // Target stationary bond length r0 = 1.10 Å (in the optimizer's Å units),
    // potential maximum (negative curvature) along the bond.
    let r0 = 1.10_f64;
    let k = -0.5_f64;

    // Start away from the maximum so the search has to climb to it.
    let geom = Geometry::from_atoms(vec![("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.85])], None);

    let mut params = BernyParams::default();
    params.mode = Mode::Ts;
    params.trust = 0.2;

    let mut opt = Berny::new(geom, 200, params);

    let mut last = None;
    while let Some(g) = opt.next() {
        let coords: Vec<[f64; 3]> = g.coords.iter().map(|c| [c[0], c[1], c[2]]).collect();
        let (e, grad) = diatomic_eg(&coords, k, r0);
        last = Some(bond_length(&g));
        opt.send((e, grad, None));
    }

    assert!(opt.converged(), "TS search should converge on the diatomic maximum");
    let final_r = bond_length(opt.current_geom());
    assert!(
        (final_r - r0).abs() < 1e-3,
        "TS search should reach the bond maximum r0={r0}, got r={final_r}"
    );
    let _ = last;
}
