//! Trust radius update using Fletcher's criterion.

/// Updates the trust radius based on predicted vs actual energy change.
///
/// # Arguments
/// * `trust` - Current trust radius
/// * `d_e` - Actual energy change
/// * `d_e_predicted` - Predicted energy change
/// * `dq_norm` - Norm of the step taken
/// * `energy_noise` - Estimated energy precision
///
/// # Returns
/// Updated trust radius.
pub fn update_trust(
    trust: f64,
    d_e: f64,
    d_e_predicted: f64,
    dq_norm: f64,
    energy_noise: f64,
) -> f64 {
    if d_e_predicted.abs() < 10.0 * energy_noise {
        // Below noise floor
        if (dq_norm - trust).abs() < 1e-10 {
            return 2.0 * trust;
        }
        return trust;
    }

    let r = if d_e != 0.0 { d_e / d_e_predicted } else { 1.0 };

    if r < 0.25 {
        dq_norm / 4.0
    } else if r > 0.75 && (dq_norm - trust).abs() < 1e-10 {
        2.0 * trust
    } else {
        trust
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_grow() {
        let new_trust = update_trust(0.3, -0.001, -0.001, 0.3, 2e-8);
        assert!((new_trust - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_trust_shrink() {
        // r = 0.1 < 0.25 → shrink
        let new_trust = update_trust(0.3, -0.0001, -0.001, 0.4, 2e-8);
        assert!((new_trust - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_trust_below_noise() {
        let new_trust = update_trust(0.3, 0.0, 1e-10, 0.3, 2e-8);
        assert!((new_trust - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_trust_keep() {
        let new_trust = update_trust(0.3, -0.0005, -0.001, 0.2, 2e-8);
        assert!((new_trust - 0.3).abs() < 1e-10);
    }
}
