use crate::state::{idx, idx_inner, N};

/// Damp velocity inside the cylinder using fractional mask.
/// mask=1.0 -> velocity zeroed, mask=0.5 -> half damped, mask=0.0 -> unchanged.
pub fn apply_mask(state: &mut crate::state::SimState) {
    if let Some(ref mask) = state.mask {
        apply_mask_fields(&mut state.vx, &mut state.vy, mask);
    }
}

/// Damp velocity fields directly using a pre-extracted mask slice.
/// Used when the target fields are not `state.vx`/`state.vy` (e.g. `vx0`/`vy0`).
pub fn apply_mask_fields(vx: &mut [f64], vy: &mut [f64], mask: &[f64]) {
    for i in 0..mask.len() {
        let fluid = 1.0 - mask[i];
        vx[i] *= fluid;
        vy[i] *= fluid;
    }
}

/// Inject inflow velocity at left boundary for Karman flow.
/// Also adds small vy perturbation to trigger vortex shedding.
pub fn inject_inflow(state: &mut crate::state::SimState, inflow_vel: f64) {
    let nx = state.nx;
    for j in 0..N {
        state.vx[idx(0, j as i32, nx)] = inflow_vel;
        state.vx[idx(1, j as i32, nx)] = inflow_vel;
        state.vy[idx(0, j as i32, nx)] = 0.0;
        state.vy[idx(1, j as i32, nx)] = 0.0;
    }
    // No inflow perturbation -- wake perturbation is applied separately
    // in fluid_step_karman after the cylinder mask, closer to where the
    // physical instability actually develops.
}

/// Inject dye tracer at left inflow for Karman visualization.
/// Central band with Gaussian taper.
pub fn inject_dye(state: &mut crate::state::SimState) {
    let nx = state.nx;
    let center = (N / 2) as f64;
    let sigma = (N / 4) as f64;
    for j in 0..N {
        let dy = j as f64 - center;
        let g = (-dy * dy / (2.0 * sigma * sigma)).exp();
        state.temperature[idx(0, j as i32, nx)] = g;
        state.temperature[idx(1, j as i32, nx)] = g;
    }
}

/// Tiny wake perturbation to trigger vortex shedding.
/// Applied just behind the cylinder where the physical instability develops,
/// rather than at the inflow where it would pollute the entire domain.
pub fn inject_wake_perturbation(state: &mut crate::state::SimState, params: &super::SolverParams) {
    let nx = state.nx;
    let cx = params.cylinder_x as i32;
    let cy = params.cylinder_y as i32;
    let r = params.cylinder_radius as i32;
    let wake_x = cx + r + 2; // 2 cells behind cylinder surface
    let amp = params.inflow_vel * 0.01;
    let noise = state.rng.next_f64() * 2.0 - 1.0;
    // Antisymmetric kick: push vy opposite directions above/below centerline
    state.vy[idx(wake_x, cy + 1, nx)] += amp * (1.0 + 0.5 * noise);
    state.vy[idx(wake_x, cy - 1, nx)] -= amp * (1.0 + 0.5 * noise);
}

/// Damp dye (temperature field) inside the cylinder using the fractional mask.
pub fn damp_dye_in_cylinder(state: &mut crate::state::SimState) {
    if let Some(ref mask) = state.mask {
        for i in 0..mask.len() {
            state.temperature[i] *= 1.0 - mask[i];
        }
    }
}

/// Vorticity confinement: counteracts numerical diffusion by amplifying
/// existing vortical structures. Essential for Stable Fluids to sustain
/// vortex shedding (Fedkiw et al. 2001).
///
/// Uses `state.vorticity` (omega) and `state.vorticity_abs` (|omega|) as scratch buffers
/// to avoid per-frame allocation.
pub(super) fn vorticity_confinement(state: &mut crate::state::SimState, epsilon: f64, dt: f64) {
    let nx = state.nx;

    // Clear scratch buffers
    state.vorticity.iter_mut().for_each(|v| *v = 0.0);
    state.vorticity_abs.iter_mut().for_each(|v| *v = 0.0);

    // Compute vorticity: omega = dvy/dx - dvx/dy
    for j in 1..(N - 1) {
        for i in 1..(nx - 1) {
            let dvydx = (state.vy[idx_inner(i + 1, j, nx)] - state.vy[idx_inner(i - 1, j, nx)]) * 0.5;
            let dvxdy = (state.vx[idx_inner(i, j + 1, nx)] - state.vx[idx_inner(i, j - 1, nx)]) * 0.5;
            let w = dvydx - dvxdy;
            state.vorticity[idx_inner(i, j, nx)] = w;
            state.vorticity_abs[idx_inner(i, j, nx)] = w.abs();
        }
    }

    // Compute grad|omega| and apply confinement force: f = epsilon*dx*(N_hat x omega)
    for j in 2..(N - 2) {
        for i in 2..(nx - 2) {
            let eta_x = (state.vorticity_abs[idx_inner(i + 1, j, nx)] - state.vorticity_abs[idx_inner(i - 1, j, nx)]) * 0.5;
            let eta_y = (state.vorticity_abs[idx_inner(i, j + 1, nx)] - state.vorticity_abs[idx_inner(i, j - 1, nx)]) * 0.5;
            let len = (eta_x * eta_x + eta_y * eta_y).sqrt() + 1e-10;
            let norm_x = eta_x / len;
            let norm_y = eta_y / len;

            let w = state.vorticity[idx_inner(i, j, nx)];
            // 2D cross product: f_x = epsilon*ny*omega, f_y = -epsilon*nx*omega
            state.vx[idx_inner(i, j, nx)] += dt * epsilon * norm_y * w;
            state.vy[idx_inner(i, j, nx)] -= dt * epsilon * norm_x * w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{idx, SimState, N};
    use crate::solver::{fluid_step_karman, SolverParams};

    #[test]
    fn test_inject_inflow() {
        let mut state = SimState::new_karman(10, 0.1, 13.0, (N / 2) as f64, 5.0, N);
        state.vx.fill(0.0);
        inject_inflow(&mut state, 0.1);
        for j in 0..N {
            assert_eq!(state.vx[idx(0, j as i32, N)], 0.1);
            assert_eq!(state.vx[idx(1, j as i32, N)], 0.1);
            assert_eq!(state.vy[idx(0, j as i32, N)], 0.0);
        }
    }

    #[test]
    fn test_inject_dye() {
        let mut state = SimState::new_karman(10, 0.1, 13.0, (N / 2) as f64, 5.0, N);
        inject_dye(&mut state);
        // Center should have dye ~= 1.0
        let center_dye = state.temperature[idx(0, (N / 2) as i32, N)];
        assert!((center_dye - 1.0).abs() < 0.01, "Center dye should be ~1.0, got {}", center_dye);
        // Far from center should be near 0
        let edge_dye = state.temperature[idx(0, 0, N)];
        assert!(edge_dye < 0.2, "Edge dye should be small, got {}", edge_dye);
        assert!(edge_dye < center_dye, "Edge dye should be less than center");
    }

    #[test]
    fn test_apply_mask_damps_velocity() {
        let mut state = SimState::new_karman(10, 0.1, 13.0, (N / 2) as f64, 5.0, N);
        // Set nonzero velocity everywhere
        let mask = state.mask.as_ref().unwrap().clone();
        for i in 0..mask.len() {
            state.vx[i] = 1.0;
            state.vy[i] = 1.0;
        }
        apply_mask(&mut state);
        for i in 0..mask.len() {
            if mask[i] == 1.0 {
                // Fully solid: velocity should be zero
                assert_eq!(state.vx[i], 0.0);
                assert_eq!(state.vy[i], 0.0);
            } else if mask[i] == 0.0 {
                // Fully fluid: velocity unchanged
                assert_eq!(state.vx[i], 1.0);
                assert_eq!(state.vy[i], 1.0);
            } else {
                // Transition: partially damped
                assert!(state.vx[i] > 0.0 && state.vx[i] < 1.0,
                    "Edge cell mask={} should have partial velocity, got vx={}", mask[i], state.vx[i]);
            }
        }
    }

    #[test]
    fn test_apply_mask_preserves_outside() {
        let mut state = SimState::new_karman(10, 0.1, 13.0, (N / 2) as f64, 5.0, N);
        let mask = state.mask.as_ref().unwrap().clone();
        // Set known velocity outside
        for i in 0..mask.len() {
            if mask[i] == 0.0 {
                state.vx[i] = 0.5;
                state.vy[i] = 0.3;
            }
        }
        apply_mask(&mut state);
        for i in 0..mask.len() {
            if mask[i] == 0.0 {
                assert_eq!(state.vx[i], 0.5);
                assert_eq!(state.vy[i], 0.3);
            }
        }
    }

    #[test]
    fn test_apply_mask_noop_without_mask() {
        let mut state = SimState::new(10, 0.15, N);
        state.vx.fill(1.0);
        apply_mask(&mut state);
        assert!(state.vx.iter().all(|&v| v == 1.0), "No mask should leave velocity unchanged");
    }

    #[test]
    fn test_fluid_step_karman_no_panic() {
        let params = SolverParams::default_karman();
        let mut state = SimState::new_karman(100, params.inflow_vel, params.cylinder_x, params.cylinder_y, params.cylinder_radius, N);
        for _ in 0..5 {
            fluid_step_karman(&mut state, &params);
        }
    }

    #[test]
    fn test_karman_velocity_inside_cylinder_stays_zero() {
        let params = SolverParams::default_karman();
        let mut state = SimState::new_karman(10, params.inflow_vel, params.cylinder_x, params.cylinder_y, params.cylinder_radius, N);
        for _ in 0..10 {
            fluid_step_karman(&mut state, &params);
        }
        let mask = state.mask.as_ref().unwrap();
        for i in 0..mask.len() {
            if mask[i] == 1.0 {
                assert_eq!(state.vx[i], 0.0, "vx should be 0 inside cylinder");
                assert_eq!(state.vy[i], 0.0, "vy should be 0 inside cylinder");
            }
        }
    }

    #[test]
    fn test_karman_inflow_maintained() {
        let params = SolverParams::default_karman();
        let mut state = SimState::new_karman(10, params.inflow_vel, params.cylinder_x, params.cylinder_y, params.cylinder_radius, N);
        for _ in 0..5 {
            fluid_step_karman(&mut state, &params);
        }
        // After steps, left boundary should still have inflow velocity
        for j in 1..(N - 1) {
            let vx_left = state.vx[idx(0, j as i32, N)];
            // Inflow is maintained by inject + boundary
            assert!(vx_left > 0.0, "Left boundary vx should be positive at y={}: got {}", j, vx_left);
        }
    }

    #[test]
    fn test_karman_dye_advects_right() {
        let params = SolverParams::default_karman();
        let mut state = SimState::new_karman(10, params.inflow_vel, params.cylinder_x, params.cylinder_y, params.cylinder_radius, N);
        for _ in 0..50 {
            fluid_step_karman(&mut state, &params);
        }
        // Dye should have advected rightward past the cylinder
        let mid_y = (N / 2) as i32;
        let check_x = (params.cylinder_x + params.cylinder_radius + 10.0) as i32;
        let dye_past_cyl = state.temperature[idx(check_x, mid_y, N)];
        assert!(dye_past_cyl > 0.0, "Dye should have reached past cylinder at x={}: got {}", check_x, dye_past_cyl);
    }

    #[test]
    fn test_karman_particles_stay_in_domain() {
        let params = SolverParams::default_karman();
        let mut state = SimState::new_karman(100, params.inflow_vel, params.cylinder_x, params.cylinder_y, params.cylinder_radius, N);
        for _ in 0..20 {
            fluid_step_karman(&mut state, &params);
        }
        let n_f = N as f64;
        for i in 0..state.particles_x.len() {
            let px = state.particles_x[i];
            let py = state.particles_y[i];
            assert!(px >= 0.0 && px < n_f, "Particle {} x out of range: {}", i, px);
            assert!(py >= 2.0 && py <= n_f - 3.0, "Particle {} y out of range: {}", i, py);
        }
    }
}
