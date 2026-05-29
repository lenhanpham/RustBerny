# RustBerny

Molecular geometry optimizer using rational function optimization (RFO) for energy minimization and partitioned RFO (P-RFO) for transition state search.

## Overview

`RustBerny` is a Rust implementation of redundant internal coordinate geometry optimization for molecular systems. It operates in redundant internal coordinates (bonds, angles, dihedrals) rather than Cartesian coordinates, providing:

- Better-conditioned Hessians
- More physically meaningful step sizes
- Faster convergence for molecular systems

The optimizer supports two modes:

- **Minimization (RFO):** Finds local minima on the potential energy surface
- **Transition State Search (P-RFO):** Finds first-order saddle points (transition states)

## Theory

### Internal Coordinates

The Wilson B-matrix maps Cartesian displacements to internal coordinate displacements:

```
dq = B · dx
```

Its pseudoinverse maps gradients back:

```
g_int = B_inv^T · g_cart
```

where `B_inv = B^T · pinv(B · B^T)`.

### Rational Function Optimization (RFO)

RFO constructs an augmented matrix that naturally handles both positive and negative curvature:

```
RFO = [[H, g],
       [g^T, 0]]
```

The step is computed from the eigenvector corresponding to the lowest eigenvalue of this matrix.

### Partitioned RFO (P-RFO) for Transition States

P-RFO partitions the Hessian eigenvectors into two groups:

1. **Transition vector** (lowest eigenvector): step *uphill* (maximize)
2. **Orthogonal complement**: step *downhill* (minimize)

This finds first-order saddle points where the Hessian has exactly one negative eigenvalue.

The algorithm follows Schlegel (1982):

> Schlegel, H. B. (1982). Optimization of equilibrium geometries and transition structures. *J. Comput. Chem.*, 3(2), 214-218.

### Hessian Updates

- **Minimization mode:** BFGS update with curvature guard (`dq · dg > 0`)
- **Transition state mode:** Flowchart SR1/BFGS/PSB selection (Birkholz & Schlegel 2016):
  - SR1 when `z·s / (|z||s|) < -0.1`
  - BFGS when `y·s / (|y||s|) > +0.1`
  - PSB otherwise (fallback)

### Trust Radius

The trust radius is adapted using Fletcher's criterion:

```
r = dE_actual / dE_predicted
if r < 0.25:  shrink (trust = ||dq|| / 4)
if r > 0.75:  grow (trust = 2 × trust)
```

### Convergence Criteria

**Minimization mode (4 criteria):**

| Criterion | Threshold |
|-----------|-----------|
| Gradient RMS | < 1.5e-4 a.u. |
| Gradient maximum | < 4.5e-4 a.u. |
| Step RMS | < 1.2e-3 a.u. |
| Step maximum | < 1.8e-3 a.u. |

**Transition state mode (5 criteria):** Same 4 + exactly 1 negative eigenvalue.

## Code Structure

```
src/
├── lib.rs          # Crate root with #![deny(missing_docs)]
├── species.rs      # Element properties (92 elements, hardcoded)
├── math.rs         # Math utilities (rms, pinv, findroot, fit_cubic, fit_quartic)
├── geometry.rs     # Geometry struct, I/O (xyz, aims), distance, connectivity
├── coords.rs       # Internal coordinates (Bond, Angle, Dihedral, B-matrix)
├── hessian.rs      # Hessian updates (BFGS, SR1/BFGS/PSB flowchart)
├── step.rs         # Quadratic steps (RFO, P-RFO with trust constraint)
├── trust.rs        # Trust radius update (Fletcher criterion)
├── core.rs         # Berny optimizer (state machine, main loop)
├── optimize.rs     # Convenience driver function
└── solvers.rs      # Solver interface with finite-difference gradients
```

## How to use RustBerny in your Rust codebase

Add to your `Cargo.toml`:

```toml
[dependencies]
rustberny = { path = "../RustBerny" }
```

## Usage

### Basic Minimization

```rust
use rustberny::geometry::Geometry;
use rustberny::core::{Berny, BernyParams, Mode};

// Create water molecule
let geom = Geometry::from_atoms(
    vec![
        ("O", [0.0, 0.0, 0.0]),
        ("H", [0.0, 0.757, 0.587]),
        ("H", [0.0, -0.757, 0.587]),
    ],
    None,
);

// Configure optimizer
let params = BernyParams {
    mode: Mode::Min,
    ..Default::default()
};

// Create optimizer
let mut optimizer = Berny::new(geom, 100, params);

// Optimization loop
for geom in optimizer {
    let atoms: Vec<(&str, [f64; 3])> = geom.species.iter()
        .zip(geom.coords.iter())
        .map(|(s, c)| (s.as_str(), [c[0], c[1], c[2]]))
        .collect();

    // Call your energy/gradient solver here
    let energy = 0.0; // Replace with actual energy
    let gradients = vec![[0.0; 3]; atoms.len()]; // Replace with actual gradients

    optimizer.send((energy, gradients, None));
}

if optimizer.converged() {
    println!("Optimization converged!");
}
```

### Transition State Search

```rust
use rustberny::core::{Berny, BernyParams, Mode};

let params = BernyParams {
    mode: Mode::Ts,
    ..Default::default()
};

let mut optimizer = Berny::new(geom, 200, params);

for geom in optimizer {
    // On first step, optionally provide Cartesian Hessian
    // from a frequency calculation
    let energy = 0.0;
    let gradients = vec![[0.0; 3]; geom.len()];
    let hessian: Option<Vec<Vec<f64>>> = None; // Provide Hessian here

    optimizer.send((energy, gradients, hessian));
}
```

### Using the Convenience Driver

```rust
use rustberny::geometry::Geometry;
use rustberny::core::{Berny, BernyParams};
use rustberny::optimize::optimize;

let geom = Geometry::from_file("molecule.xyz").unwrap();
let mut optimizer = Berny::new(geom, 100, BernyParams::default());

let final_geom = optimize(
    &mut optimizer,
    &mut |atoms, _lattice| {
        // Your energy/gradient calculation
        let energy = 0.0;
        let gradients = vec![[0.0; 3]; atoms.len()];
        (energy, gradients)
    },
    Some("trajectory.xyz".as_ref()),
);
```

## API Reference

### Core Types

#### `Berny`

The main optimizer struct. Implements `Iterator<Item = Geometry>`.

```rust
pub struct Berny { /* ... */ }

impl Berny {
    pub fn new(geom: Geometry, maxsteps: usize, params: BernyParams) -> Self;
    pub fn send(&mut self, result: (f64, Vec<Vec<f64>>, Option<Vec<Vec<f64>>>));
    pub fn converged(&self) -> bool;
    pub fn step(&self) -> usize;
    pub fn trust(&self) -> f64;
    pub fn current_geom(&self) -> &Geometry;
}
```

#### `BernyParams`

```rust
pub struct BernyParams {
    pub gradient_max: f64,    // 0.45e-3
    pub gradient_rms: f64,    // 0.15e-3
    pub step_max: f64,        // 1.8e-3
    pub step_rms: f64,        // 1.2e-3
    pub trust: f64,           // 0.3
    pub energy_noise: f64,    // 2e-8
    pub dihedral: bool,       // true
    pub superweak_dih: bool,  // false
    pub mode: Mode,           // Min
}
```

#### `Mode`

```rust
pub enum Mode {
    Min,  // Energy minimization
    Ts,   // Transition state search
}
```

### Geometry

```rust
pub struct Geometry {
    pub species: Vec<String>,
    pub coords: Vec<Vector3<f64>>,
    pub lattice: Option<Matrix3<f64>>,
}

impl Geometry {
    pub fn from_atoms(atoms: Vec<(&str, [f64; 3])>, lattice: Option<[[f64; 3]; 3]>) -> Self;
    pub fn len(&self) -> usize;
    pub fn formula(&self) -> String;
    pub fn dump(&self, fmt: &str) -> String;
    pub fn dist(&self) -> DMatrix<f64>;
    pub fn bondmatrix(&self, scale: f64) -> Vec<Vec<bool>>;
    pub fn rho(&self) -> DMatrix<f64>;
}

pub fn loads(s: &str, fmt: &str) -> Result<Geometry, String>;
pub fn readfile(path: &Path) -> Result<Geometry, String>;
```

### Element Properties

```rust
pub fn get_property(symbol: &str, property: &str) -> f64;
pub fn is_ghost(symbol: &str) -> bool;
```

Supported properties: `"number"`, `"name"`, `"covalent_radius"`, `"mass"`, `"vdw_radius"`

### Hessian Updates

```rust
pub fn update_bfgs(h: &DMatrix, dq: &DVector, dg: &DVector) -> DMatrix;
pub fn update_hessian_ts(h: &DMatrix, dq: &DVector, dg: &DVector) -> DMatrix;
```

### Quadratic Steps

```rust
pub fn quadratic_step(g: &DVector, h: &DMatrix, w: &DVector, trust: f64)
    -> (DVector, f64, bool);

pub fn quadratic_step_ts(g: &DVector, h: &DMatrix, w: &DVector, trust: f64)
    -> (DVector, f64, bool);
```

Returns `(step, predicted_energy_change, on_sphere)`.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Linear algebra (matrices, vectors, eigendecomposition) |
| `once_cell` | Lazy static initialization |
| `thiserror` | Custom error types |
| `log` | Logging facade |
| `serde` / `serde_json` | Serialization (trace output) |
| `num-complex` | Complex numbers (polynomial root-finding) |
| `itertools` | Iterator utilities (combinations) |

## Testing

```bash
cargo test
```

All 34 unit tests cover:
- Element property lookup
- Math utilities (rms, pinv, findroot, polynomial fitting)
- Geometry I/O and distance calculations
- Coordinate evaluation and B-matrix
- Hessian updates (BFGS, SR1/PSB)
- RFO and P-RFO step computation
- Trust radius updates

## References

1. Schlegel, H. B. (1982). Optimization of equilibrium geometries and transition structures. *J. Comput. Chem.*, 3(2), 214-218.

2. Birkholz, A. B., & Schlegel, H. B. (2016). Exploration of transition states using a modified quadratic steepest descent path. *Theor. Chem. Acc.*, 135, 84.

3. Fletcher, R. (1980). *Practical Methods of Optimization*. Wiley.

## License

MIT 
