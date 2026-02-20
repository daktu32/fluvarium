use crate::state::{idx, idx_inner, N};

/// Inject large-scale temperature perturbation at convection-cell wavelengths.
/// Small-scale (per-cell) noise gets killed by projection and diffusion.
/// We need modes at wavelength ~nx/k (k=1..4) to seed convection cells.
/// Called once at initialization; convection is self-sustaining thereafter.
pub fn inject_thermal_perturbation(
    temperature: &mut [f64],
    rng: &mut crate::state::Xor128,
    amp: f64,
    nx: usize,
) {
    use std::f64::consts::{PI, TAU};
    let n_f = N as f64;
    let nx_f = nx as f64;
    // 4 random sinusoidal modes at convection scales
    let modes: [(f64, f64); 4] = [
        (1.0, rng.next_f64() * TAU),
        (2.0, rng.next_f64() * TAU),
        (3.0, rng.next_f64() * TAU),
        (4.0, rng.next_f64() * TAU),
    ];
    for j in 2..(N - 2) {
        // Envelope: zero at walls, max at midplane
        let y_env = (PI * j as f64 / (n_f - 1.0)).sin();
        for i in 0..nx {
            let ii = idx(i as i32, j as i32, nx);
            let mut perturbation = 0.0;
            for &(k, phase) in &modes {
                perturbation += (TAU * k * i as f64 / nx_f + phase).sin();
            }
            temperature[ii] += amp * y_env * perturbation * 0.25;
        }
    }
}

/// Volumetric heat source at bottom hot spot and Newtonian cooling at top.
/// This supplements the Dirichlet BCs which alone cannot drive heat into
/// the interior fast enough (wall velocity = 0, diff is small).
pub(super) fn inject_heat_source(temperature: &mut [f64], dt: f64, source_strength: f64, cool_rate: f64, nx: usize) {
    let center = (nx / 2) as f64;
    let sigma = (N / 24) as f64; // narrow: ~5 cells

    // Concentrated heat injection at bottom (rows 2-4)
    for j in 2..5 {
        let y_factor = 1.0 - (j - 2) as f64 / 3.0;
        for i in 0..nx {
            let dx = i as f64 - center;
            let g = (-dx * dx / (2.0 * sigma * sigma)).exp();
            let ii = idx_inner(i, j, nx);
            temperature[ii] += dt * source_strength * g * y_factor;
        }
    }

    // Newtonian cooling near top (rows N-7 to N-3): decay toward 0
    for j in (N - 7)..(N - 2) {
        let y_factor = 1.0 - ((N - 3) - j) as f64 / 5.0;
        for i in 0..nx {
            let ii = idx_inner(i, j, nx);
            temperature[ii] *= 1.0 - dt * cool_rate * y_factor;
        }
    }
}

/// Apply buoyancy force: hot fluid rises, cold fluid sinks.
/// vy += dt * buoyancy * (T - T_ambient), where T_ambient = bottom_base
pub(super) fn apply_buoyancy(vy: &mut [f64], temperature: &[f64], buoyancy: f64, dt: f64, bottom_base: f64, nx: usize) {
    let t_ambient = bottom_base;
    for j in 1..(N - 1) {
        for i in 0..nx {
            let ii = idx_inner(i, j, nx);
            vy[ii] += dt * buoyancy * (temperature[ii] - t_ambient);
        }
    }
}

/// Apply buoyancy from perturbation temperature (deviation from conduction profile).
/// vy += dt * buoyancy * (T - T_cond(y)), where T_cond = 1 - y/(N-1).
/// Used in benchmark mode where uniform Dirichlet BCs would otherwise cause
/// buoyancy to be absorbed by the pressure projection.
pub(super) fn apply_buoyancy_perturbation(vy: &mut [f64], temperature: &[f64], buoyancy: f64, dt: f64, nx: usize) {
    let n_minus_1 = (N - 1) as f64;
    for j in 1..(N - 1) {
        let t_cond = 1.0 - j as f64 / n_minus_1;
        for i in 0..nx {
            let ii = idx_inner(i, j, nx);
            vy[ii] += dt * buoyancy * (temperature[ii] - t_cond);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{idx, N};

    const BB: f64 = 0.15;

    #[test]
    fn test_perturbation_buoyancy_conduction_zero() {
        // If temperature exactly matches conduction profile, no buoyancy is added
        let mut vy = vec![0.0; N * N];
        let mut temperature = vec![0.0; N * N];
        for j in 0..N {
            let t_cond = 1.0 - j as f64 / (N - 1) as f64;
            for i in 0..N {
                temperature[idx(i as i32, j as i32, N)] = t_cond;
            }
        }
        apply_buoyancy_perturbation(&mut vy, &temperature, 10.0, 1.0, N);
        let max_vy: f64 = vy.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max_vy < 1e-12, "Should have zero buoyancy at conduction profile, got {}", max_vy);
    }

    #[test]
    fn test_perturbation_buoyancy_hot_anomaly() {
        // Hot anomaly above conduction profile → positive vy (upward)
        let mut vy = vec![0.0; N * N];
        let mut temperature = vec![0.0; N * N];
        let mid = N / 2;
        for j in 0..N {
            let t_cond = 1.0 - j as f64 / (N - 1) as f64;
            for i in 0..N {
                temperature[idx(i as i32, j as i32, N)] = t_cond;
            }
        }
        // Add hot anomaly at midplane
        temperature[idx((mid) as i32, mid as i32, N)] = 1.0; // much hotter than T_cond ~0.5
        apply_buoyancy_perturbation(&mut vy, &temperature, 1.0, 1.0, N);
        let vy_mid = vy[idx(mid as i32, mid as i32, N)];
        assert!(vy_mid > 0.0, "Hot anomaly should push upward, got {}", vy_mid);
    }

    #[test]
    fn test_buoyancy_creates_upward_velocity() {
        let mut vy = vec![0.0; N * N];
        let mut temperature = vec![0.5; N * N];

        // Create a hot spot in the interior
        let mid = N / 2;
        for j in (mid - 5)..(mid + 5) {
            for i in (mid - 8)..(mid + 8) {
                temperature[idx(i as i32, j as i32, N)] = 1.0;
            }
        }

        apply_buoyancy(&mut vy, &temperature, 1.0, 1.0, BB, N);

        // Hot spot should have stronger upward vy than ambient
        let hot_vy = vy[idx(mid as i32, mid as i32, N)];
        assert!(hot_vy > 0.0, "Hot spot should have positive vy (upward): got {}", hot_vy);

        let ambient_vy = vy[idx(10, (mid / 2) as i32, N)];
        assert!(hot_vy > ambient_vy, "Hot spot vy should exceed ambient vy");
    }
}
