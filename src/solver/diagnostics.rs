use crate::state::{idx_inner, N};

/// Compute perturbation temperature θ = T - T_cond(y), where T_cond = 1 - y/(N-1).
pub fn compute_theta(temperature: &[f64], nx: usize) -> Vec<f64> {
    let mut theta = vec![0.0; nx * N];
    let n_minus_1 = (N - 1) as f64;
    for j in 0..N {
        let t_cond = 1.0 - j as f64 / n_minus_1;
        for i in 0..nx {
            let ii = idx_inner(i, j, nx);
            theta[ii] = temperature[ii] - t_cond;
        }
    }
    theta
}

/// Compute Nusselt number: Nu = 1 + <vy * θ> / diff.
/// The angle brackets denote a horizontal average over interior cells.
pub fn compute_nusselt(vy: &[f64], temperature: &[f64], diff: f64, nx: usize) -> f64 {
    let theta = compute_theta(temperature, nx);
    let mut sum = 0.0;
    let mut count = 0usize;
    for j in 1..(N - 1) {
        for i in 0..nx {
            let ii = idx_inner(i, j, nx);
            sum += vy[ii] * theta[ii];
            count += 1;
        }
    }
    let avg = if count > 0 { sum / count as f64 } else { 0.0 };
    1.0 + avg / diff
}

/// Compute volume-averaged kinetic energy: KE = 0.5 * <vx² + vy²>.
pub fn compute_kinetic_energy(vx: &[f64], vy: &[f64], nx: usize) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for j in 1..(N - 1) {
        for i in 0..nx {
            let ii = idx_inner(i, j, nx);
            sum += vx[ii] * vx[ii] + vy[ii] * vy[ii];
            count += 1;
        }
    }
    if count > 0 { 0.5 * sum / count as f64 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::N;

    #[test]
    fn test_theta_at_conduction_profile() {
        // If T = T_cond everywhere, theta should be zero
        let nx = N;
        let mut temperature = vec![0.0; nx * N];
        for j in 0..N {
            let t_cond = 1.0 - j as f64 / (N - 1) as f64;
            for i in 0..nx {
                temperature[idx_inner(i, j, nx)] = t_cond;
            }
        }
        let theta = compute_theta(&temperature, nx);
        let max = theta.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(max < 1e-12, "Theta should be zero at conduction profile, max={}", max);
    }

    #[test]
    fn test_theta_positive_anomaly() {
        let nx = N;
        let mut temperature = vec![0.0; nx * N];
        let mid = N / 2;
        for j in 0..N {
            let t_cond = 1.0 - j as f64 / (N - 1) as f64;
            for i in 0..nx {
                temperature[idx_inner(i, j, nx)] = t_cond;
            }
        }
        // Hot anomaly at midplane
        temperature[idx_inner(mid, mid, nx)] += 0.5;
        let theta = compute_theta(&temperature, nx);
        assert!((theta[idx_inner(mid, mid, nx)] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_nusselt_at_conduction() {
        // No flow → Nu = 1
        let nx = N;
        let vy = vec![0.0; nx * N];
        let mut temperature = vec![0.0; nx * N];
        for j in 0..N {
            let t_cond = 1.0 - j as f64 / (N - 1) as f64;
            for i in 0..nx {
                temperature[idx_inner(i, j, nx)] = t_cond;
            }
        }
        let nu = compute_nusselt(&vy, &temperature, 0.01, nx);
        assert!((nu - 1.0).abs() < 1e-10, "Nu should be 1.0 at conduction, got {}", nu);
    }

    #[test]
    fn test_nusselt_positive_convection() {
        // Upward flow (vy > 0) with positive theta → Nu > 1
        let nx = N;
        let mut vy = vec![0.0; nx * N];
        let mut temperature = vec![0.0; nx * N];
        for j in 0..N {
            let t_cond = 1.0 - j as f64 / (N - 1) as f64;
            for i in 0..nx {
                let ii = idx_inner(i, j, nx);
                temperature[ii] = t_cond + 0.1; // positive theta everywhere
                vy[ii] = 0.1; // upward flow
            }
        }
        let nu = compute_nusselt(&vy, &temperature, 0.01, nx);
        assert!(nu > 1.0, "Nu should be > 1 with convection, got {}", nu);
    }

    #[test]
    fn test_kinetic_energy_zero() {
        let nx = N;
        let vx = vec![0.0; nx * N];
        let vy = vec![0.0; nx * N];
        let ke = compute_kinetic_energy(&vx, &vy, nx);
        assert!(ke.abs() < 1e-15, "KE should be 0 with no flow, got {}", ke);
    }

    #[test]
    fn test_kinetic_energy_uniform_flow() {
        let nx = N;
        let vx = vec![1.0; nx * N];
        let vy = vec![0.0; nx * N];
        let ke = compute_kinetic_energy(&vx, &vy, nx);
        // Interior cells: vx=1 → KE = 0.5 * 1^2 = 0.5
        assert!((ke - 0.5).abs() < 1e-10, "KE should be 0.5, got {}", ke);
    }
}
