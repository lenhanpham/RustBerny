//! Solver interface for energy and gradient evaluation.

use crate::geometry::Geometry;

/// Wraps any energy function with finite-difference gradients.
pub struct GenericSolver<F>
where
    F: FnMut(&[(&str, [f64; 3])], Option<&[[f64; 3]; 3]>) -> f64,
{
    energy_fn: F,
    delta: f64,
}

impl<F> GenericSolver<F>
where
    F: FnMut(&[(&str, [f64; 3])], Option<&[[f64; 3]; 3]>) -> f64,
{
    /// Creates a new generic solver with finite-difference gradient.
    pub fn new(energy_fn: F, delta: f64) -> Self {
        Self { energy_fn, delta }
    }

    /// Computes energy and gradients for the given geometry.
    pub fn compute(&mut self, geom: &Geometry) -> (f64, Vec<Vec<f64>>) {
        let atoms: Vec<(&str, [f64; 3])> = geom
            .species
            .iter()
            .zip(geom.coords.iter())
            .map(|(s, c)| (s.as_str(), [c[0], c[1], c[2]]))
            .collect();
        let energy = (self.energy_fn)(&atoms, None);
        let gradients = self.numerical_gradient(geom);
        (energy, gradients)
    }

    /// Computes numerical gradient using 5-point finite difference.
    fn numerical_gradient(&mut self, geom: &Geometry) -> Vec<Vec<f64>> {
        let n = geom.len();
        let mut gradients = vec![vec![0.0; 3]; n];

        for i in 0..n {
            for j in 0..3 {
                let mut energies = [0.0; 5];
                for (k, &step) in [-2.0, -1.0, 1.0, 2.0].iter().enumerate() {
                    let mut geom_diff = geom.clone();
                    geom_diff.coords[i][j] += step * self.delta;
                    let atoms: Vec<(&str, [f64; 3])> = geom_diff
                        .species
                        .iter()
                        .zip(geom_diff.coords.iter())
                        .map(|(s, c)| (s.as_str(), [c[0], c[1], c[2]]))
                        .collect();
                    energies[k] = (self.energy_fn)(&atoms, None);
                }
                // 5-point finite difference
                gradients[i][j] = (energies[0] / 12.0 - 2.0 * energies[1] / 3.0
                    + 2.0 * energies[2] / 3.0
                    - energies[3] / 12.0)
                    / self.delta;
            }
        }
        gradients
    }
}
