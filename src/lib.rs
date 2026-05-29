//! Molecular geometry optimizer using rational function optimization.
//!
//! Provides redundant internal coordinate optimization for molecular systems,
//! supporting both energy minimization and transition state search.

#![deny(missing_docs)]

pub mod coords;
pub mod geometry;
pub mod hessian;
pub mod math;
pub mod core;
pub mod optimize;
pub mod solvers;
pub mod species;
pub mod step;
pub mod trust;
