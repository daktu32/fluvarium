use crate::state::{idx_inner, N};

/// Compute velocity magnitude into the temperature field as passive dye.
/// Normalized to [0, 1] by dividing by max speed.
pub fn compute_velocity_dye(state: &mut crate::state::SimState) {
    let nx = state.nx;
    let mut max_speed = 0.0_f64;
    for j in 1..(N - 1) {
        for i in 1..(nx - 1) {
            let ii = idx_inner(i, j, nx);
            let speed = (state.vx[ii] * state.vx[ii] + state.vy[ii] * state.vy[ii]).sqrt();
            max_speed = max_speed.max(speed);
        }
    }
    if max_speed < 1e-12 {
        state.temperature.fill(0.0);
        return;
    }
    for j in 0..N {
        for i in 0..nx {
            let ii = idx_inner(i, j, nx);
            let speed = (state.vx[ii] * state.vx[ii] + state.vy[ii] * state.vy[ii]).sqrt();
            state.temperature[ii] = (speed / max_speed).min(1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{SimState, N};

    #[test]
    fn test_compute_velocity_dye_zero_velocity() {
        let mut state = SimState::new_cavity(10, N);
        compute_velocity_dye(&mut state);
        assert!(state.temperature.iter().all(|&t| t == 0.0));
    }

    #[test]
    fn test_compute_velocity_dye_normalized() {
        let mut state = SimState::new_cavity(10, N);
        // Set some velocity at an interior cell
        state.vx[idx_inner(N / 2, N / 2, N)] = 1.0;
        compute_velocity_dye(&mut state);
        // Max should be 1.0
        let max_t = state.temperature.iter().cloned().fold(0.0_f64, f64::max);
        assert!((max_t - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_velocity_dye_bounded() {
        let mut state = SimState::new_cavity(10, N);
        // Set various velocities
        state.vx[idx_inner(10, 10, N)] = 0.5;
        state.vy[idx_inner(10, 10, N)] = 0.5;
        state.vx[idx_inner(20, 20, N)] = 1.0;
        compute_velocity_dye(&mut state);
        // All values should be in [0, 1]
        for &t in state.temperature.iter() {
            assert!(t >= 0.0 && t <= 1.0, "Temperature should be in [0,1], got {}", t);
        }
    }
}
