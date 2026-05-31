//! Trust radius update using Fletcher's criterion.

/// Minimum trust radius.
const TRUST_MIN: f64 = 0.05;
/// Maximum trust radius.
const TRUST_MAX: f64 = 3.0;

/// # Arguments
/// * `trust` - Current trust radius
/// * `d_e` - Actual energy change
/// * `d_e_predicted` - Predicted energy change
/// * `dq_norm` - Norm of the step taken
/// * `energy_noise` - Estimated energy precision

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
            return (2.0 * trust).min(TRUST_MAX);
        }
        return trust;
    }

    let r = if d_e != 0.0 { d_e / d_e_predicted } else { 1.0 };

    let new_trust = if r < 0.25 {
        dq_norm / 4.0
    } else if r > 0.75 && (dq_norm - trust).abs() < 1e-10 {
        2.0 * trust
    } else {
        trust
    };

    new_trust.max(TRUST_MIN).min(TRUST_MAX)
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
    fn test_trust_grow_clamped() {
        // 2.0 * 2.5 = 5.0 > TRUST_MAX = 3.0
        let new_trust = update_trust(2.5, -0.001, -0.001, 2.5, 2e-8);
        assert!((new_trust - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trust_shrink() {
        // r = 0.1 < 0.25 → shrink
        let new_trust = update_trust(0.3, -0.0001, -0.001, 0.4, 2e-8);
        assert!((new_trust - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_trust_shrink_clamped() {
        // dq_norm/4 = 0.01/4 = 0.0025 < TRUST_MIN = 0.05
        let new_trust = update_trust(0.3, -0.0001, -0.001, 0.01, 2e-8);
        assert!((new_trust - 0.05).abs() < 1e-10);
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
