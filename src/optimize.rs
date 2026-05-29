//! Convenience driver for geometry optimization.

use crate::geometry::Geometry;
use crate::core::Berny;

/// Optimizes a geometry using the given solver.
pub fn optimize(
    optimizer: &mut Berny,
    solver: &mut dyn FnMut(&[(&str, [f64; 3])], Option<&[[f64; 3]; 3]>) -> (f64, Vec<Vec<f64>>),
    trajectory: Option<&std::path::Path>,
) -> Geometry {
    use std::fs::OpenOptions;
    use std::io::Write;

    let mut traj_file =
        trajectory.and_then(|p| OpenOptions::new().create(true).write(true).open(p).ok());

    let mut geom = optimizer.next().unwrap();
    loop {
        let atoms: Vec<(&str, [f64; 3])> = geom
            .species
            .iter()
            .zip(geom.coords.iter())
            .map(|(s, c)| (s.as_str(), [c[0], c[1], c[2]]))
            .collect();
        let result = solver(&atoms, None);
        optimizer.send((result.0, result.1, None));

        if let Some(ref mut f) = traj_file {
            let _ = writeln!(f, "{}", geom.dump("xyz"));
        }

        match optimizer.next() {
            Some(g) => geom = g,
            None => break,
        }
    }
    geom
}
