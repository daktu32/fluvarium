use crate::state::{idx, N};

/// Boundary condition handler for top/bottom walls only.
/// X-axis is periodic (wraps via idx).
///   - field_type 0 (scalar): Neumann (copy neighbor) at walls
///   - field_type 1 (vx): negate at walls (no-slip)
///   - field_type 2 (vy): negate at walls (no-slip + no-penetration)
///   - field_type 3 (temperature): top/bottom Dirichlet (hot bottom, cold top)
pub fn set_bnd(field_type: i32, x: &mut [f64], bottom_base: f64) {
    // Top/bottom walls (y boundaries)
    for i in 0..N {
        match field_type {
            1 | 2 => {
                x[idx(i as i32, 0)] = -x[idx(i as i32, 1)];
                x[idx(i as i32, (N - 1) as i32)] = -x[idx(i as i32, (N - 2) as i32)];
            }
            3 => {
                // Bottom: Gaussian hot spot centered at N/2
                let dx = i as f64 - (N / 2) as f64;
                let sigma = (N / 24) as f64;
                let hot = bottom_base + (1.0 - bottom_base) * (-dx * dx / (2.0 * sigma * sigma)).exp();
                x[idx(i as i32, 0)] = hot;
                x[idx(i as i32, 1)] = hot;
                // Top: cold
                x[idx(i as i32, (N - 1) as i32)] = 0.0;
                x[idx(i as i32, (N - 2) as i32)] = 0.0;
            }
            _ => {
                x[idx(i as i32, 0)] = x[idx(i as i32, 1)];
                x[idx(i as i32, (N - 1) as i32)] = x[idx(i as i32, (N - 2) as i32)];
            }
        }
    }
}

/// Gauss-Seidel iterative linear solver.
/// Solves: x[i,j] = (x0[i,j] + a * (neighbors)) / c
pub fn lin_solve(field_type: i32, x: &mut [f64], x0: &[f64], a: f64, c: f64, iter: usize, bottom_base: f64) {
    let c_inv = 1.0 / c;
    for _ in 0..iter {
        for j in 1..(N - 1) {
            for i in 0..N {
                let neighbors = x[idx(i as i32 - 1, j as i32)]
                    + x[idx(i as i32 + 1, j as i32)]
                    + x[idx(i as i32, j as i32 - 1)]
                    + x[idx(i as i32, j as i32 + 1)];
                x[idx(i as i32, j as i32)] = (x0[idx(i as i32, j as i32)] + a * neighbors) * c_inv;
            }
        }
        set_bnd(field_type, x, bottom_base);
    }
}

/// Diffusion step: spreads the field over time.
/// a = dt * diff * (N-2)^2, c = 1 + 4a
pub fn diffuse(field_type: i32, x: &mut [f64], x0: &[f64], diff: f64, dt: f64, iter: usize, bottom_base: f64) {
    let a = dt * diff * ((N - 2) as f64) * ((N - 2) as f64);
    let c = 1.0 + 4.0 * a;
    // Initialize x from x0
    x.copy_from_slice(x0);
    lin_solve(field_type, x, x0, a, c, iter, bottom_base);
}

/// Semi-Lagrangian advection: traces particles backwards through velocity field.
pub fn advect(field_type: i32, d: &mut [f64], d0: &[f64], vx: &[f64], vy: &[f64], dt: f64, bottom_base: f64) {
    let dt0 = dt * (N - 2) as f64;
    let n_f = N as f64;

    for j in 1..(N - 1) {
        for i in 0..N {
            let ii = idx(i as i32, j as i32);
            // Trace backwards
            let x = i as f64 - dt0 * vx[ii];
            let mut y = j as f64 - dt0 * vy[ii];

            // Clamp y to valid interior range
            if y < 0.5 {
                y = 0.5;
            }
            if y > n_f - 1.5 {
                y = n_f - 1.5;
            }

            // X wraps via idx (no clamping needed)
            let i0 = x.floor() as i32;
            let j0 = y.floor() as i32;
            let i1 = i0 + 1;
            let j1 = j0 + 1;
            let s1 = x - i0 as f64;
            let s0 = 1.0 - s1;
            let t1 = y - j0 as f64;
            let t0 = 1.0 - t1;

            d[ii] = s0 * (t0 * d0[idx(i0, j0)] + t1 * d0[idx(i0, j1)])
                + s1 * (t0 * d0[idx(i1, j0)] + t1 * d0[idx(i1, j1)]);
        }
    }
    set_bnd(field_type, d, bottom_base);
}

/// Pressure projection: enforces incompressibility (divergence-free velocity field).
pub fn project(vx: &mut [f64], vy: &mut [f64], p: &mut [f64], div: &mut [f64], iter: usize, bottom_base: f64) {
    let h = 1.0 / (N - 2) as f64;

    // Calculate divergence
    for j in 1..(N - 1) {
        for i in 0..N {
            div[idx(i as i32, j as i32)] = -0.5
                * h
                * (vx[idx(i as i32 + 1, j as i32)] - vx[idx(i as i32 - 1, j as i32)]
                    + vy[idx(i as i32, j as i32 + 1)] - vy[idx(i as i32, j as i32 - 1)]);
            p[idx(i as i32, j as i32)] = 0.0;
        }
    }
    set_bnd(0, div, bottom_base);
    set_bnd(0, p, bottom_base);

    // Solve for pressure
    lin_solve(0, p, div, 1.0, 4.0, iter, bottom_base);

    // Subtract pressure gradient from velocity
    for j in 1..(N - 1) {
        for i in 0..N {
            vx[idx(i as i32, j as i32)] -=
                0.5 * (p[idx(i as i32 + 1, j as i32)] - p[idx(i as i32 - 1, j as i32)]) / h;
            vy[idx(i as i32, j as i32)] -=
                0.5 * (p[idx(i as i32, j as i32 + 1)] - p[idx(i as i32, j as i32 - 1)]) / h;
        }
    }
    set_bnd(1, vx, bottom_base);
    set_bnd(2, vy, bottom_base);
}

/// Solver parameters for the fluid simulation.
#[derive(Clone)]
pub struct SolverParams {
    pub visc: f64,
    pub diff: f64,
    pub dt: f64,
    pub diffuse_iter: usize,
    pub project_iter: usize,
    pub heat_buoyancy: f64,
    pub noise_amp: f64,
    pub source_strength: f64,
    pub cool_rate: f64,
    pub bottom_base: f64,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            visc: 0.008,
            diff: 0.002,
            dt: 0.003,
            diffuse_iter: 20,
            project_iter: 30,
            heat_buoyancy: 8.0,
            noise_amp: 0.0,
            source_strength: 10.0,
            cool_rate: 8.0,
            bottom_base: 0.15,
        }
    }
}

impl From<&crate::config::PhysicsConfig> for SolverParams {
    fn from(cfg: &crate::config::PhysicsConfig) -> Self {
        Self {
            visc: cfg.visc,
            diff: cfg.diff,
            dt: cfg.dt,
            diffuse_iter: cfg.diffuse_iter,
            project_iter: cfg.project_iter,
            heat_buoyancy: cfg.heat_buoyancy,
            noise_amp: cfg.noise_amp,
            source_strength: cfg.source_strength,
            cool_rate: cfg.cool_rate,
            bottom_base: cfg.bottom_base,
        }
    }
}

/// Inject large-scale temperature perturbation at convection-cell wavelengths.
/// Small-scale (per-cell) noise gets killed by projection and diffusion.
/// We need modes at wavelength ~N/k (k=1..4) to seed/sustain convection cells.
fn inject_thermal_perturbation(
    temperature: &mut [f64],
    rng: &mut crate::state::Xor128,
    amp: f64,
) {
    use std::f64::consts::{PI, TAU};
    let n_f = N as f64;
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
        for i in 0..N {
            let ii = idx(i as i32, j as i32);
            let mut perturbation = 0.0;
            for &(k, phase) in &modes {
                perturbation += (TAU * k * i as f64 / n_f + phase).sin();
            }
            temperature[ii] += amp * y_env * perturbation * 0.25;
        }
    }
}

/// Volumetric heat source at bottom hot spot and Newtonian cooling at top.
/// This supplements the Dirichlet BCs which alone cannot drive heat into
/// the interior fast enough (wall velocity ≈ 0, diff is small).
fn inject_heat_source(temperature: &mut [f64], dt: f64, source_strength: f64, cool_rate: f64) {
    let center = (N / 2) as f64;
    let sigma = (N / 24) as f64; // narrow: ~5 cells

    // Concentrated heat injection at bottom (rows 2–4)
    for j in 2..5 {
        let y_factor = 1.0 - (j - 2) as f64 / 3.0;
        for i in 0..N {
            let dx = i as f64 - center;
            let g = (-dx * dx / (2.0 * sigma * sigma)).exp();
            let ii = idx(i as i32, j as i32);
            temperature[ii] += dt * source_strength * g * y_factor;
        }
    }

    // Newtonian cooling near top (rows N-7 to N-3): decay toward 0
    for j in (N - 7)..(N - 2) {
        let y_factor = 1.0 - ((N - 3) - j) as f64 / 5.0;
        for i in 0..N {
            let ii = idx(i as i32, j as i32);
            temperature[ii] *= 1.0 - dt * cool_rate * y_factor;
        }
    }
}

/// Apply buoyancy force: hot fluid rises, cold fluid sinks.
/// vy += dt * buoyancy * (T - T_ambient), where T_ambient = bottom_base
fn apply_buoyancy(vy: &mut [f64], temperature: &[f64], buoyancy: f64, dt: f64, bottom_base: f64) {
    let t_ambient = bottom_base;
    for j in 1..(N - 1) {
        for i in 0..N {
            let ii = idx(i as i32, j as i32);
            vy[ii] += dt * buoyancy * (temperature[ii] - t_ambient);
        }
    }
}

/// Advect particles through the velocity field using bilinear interpolation.
/// X wraps (periodic), Y reflects at walls.
fn advect_particles(state: &mut crate::state::SimState, dt: f64) {
    let dt0 = dt * (N - 2) as f64;
    let n_f = N as f64;

    for p in 0..state.particles_x.len() {
        let px = state.particles_x[p];
        let py = state.particles_y[p];

        // Bilinear interpolation of velocity at particle position
        let i0 = px.floor() as i32;
        let j0 = py.floor().max(0.0).min(n_f - 2.0) as i32;
        let j1 = j0 + 1;
        let sx = px - px.floor();
        let sy = py - j0 as f64;

        let vx_interp = (1.0 - sx) * (1.0 - sy) * state.vx[idx(i0, j0)]
            + sx * (1.0 - sy) * state.vx[idx(i0 + 1, j0)]
            + (1.0 - sx) * sy * state.vx[idx(i0, j1)]
            + sx * sy * state.vx[idx(i0 + 1, j1)];

        let vy_interp = (1.0 - sx) * (1.0 - sy) * state.vy[idx(i0, j0)]
            + sx * (1.0 - sy) * state.vy[idx(i0 + 1, j0)]
            + (1.0 - sx) * sy * state.vy[idx(i0, j1)]
            + sx * sy * state.vy[idx(i0 + 1, j1)];

        // Move particle forward
        let new_x = px + dt0 * vx_interp;
        let mut new_y = py + dt0 * vy_interp;

        // Y: ping-pong reflect within interior [y_min, y_max].
        // Keep particles outside the 2-row Dirichlet boundary zone where
        // no-slip makes velocity ≈ 0 and particles would get trapped.
        let y_min = 2.0;
        let y_max = n_f - 3.0;
        let y_range = y_max - y_min;
        if new_y < y_min || new_y > y_max {
            let mut t = (new_y - y_min) % (2.0 * y_range);
            if t < 0.0 {
                t += 2.0 * y_range;
            }
            new_y = if t <= y_range {
                y_min + t
            } else {
                y_max - (t - y_range)
            };
        }
        new_y = new_y.clamp(y_min, y_max);

        // X: wrap around (periodic)
        let new_x = ((new_x % n_f) + n_f) % n_f;

        state.particles_x[p] = new_x;
        state.particles_y[p] = new_y;
    }
}

/// Full fluid simulation step.
pub fn fluid_step(state: &mut crate::state::SimState, params: &SolverParams) {
    let dt = params.dt;
    let bb = params.bottom_base;

    // Diffuse velocity
    diffuse(1, &mut state.vx0, &state.vx, params.visc, dt, params.diffuse_iter, bb);
    diffuse(2, &mut state.vy0, &state.vy, params.visc, dt, params.diffuse_iter, bb);

    // Project to make diffused velocity divergence-free
    project(
        &mut state.vx0,
        &mut state.vy0,
        &mut state.work,
        &mut state.work2,
        params.project_iter,
        bb,
    );

    // Advect velocity
    advect(1, &mut state.vx, &state.vx0, &state.vx0, &state.vy0, dt, bb);
    advect(2, &mut state.vy, &state.vy0, &state.vx0, &state.vy0, dt, bb);

    // Apply buoyancy with dt scaling
    apply_buoyancy(&mut state.vy, &state.temperature, params.heat_buoyancy, dt, bb);

    // Project BEFORE temperature advection — velocity must be divergence-free
    project(
        &mut state.vx,
        &mut state.vy,
        &mut state.work,
        &mut state.work2,
        params.project_iter,
        bb,
    );

    // Diffuse + advect temperature with divergence-free velocity
    diffuse(3, &mut state.work, &state.temperature, params.diff, dt, params.diffuse_iter, bb);
    advect(3, &mut state.temperature, &state.work, &state.vx, &state.vy, dt, bb);

    // Inject large-scale temperature perturbation to seed/sustain convection cells
    if params.noise_amp > 0.0 {
        inject_thermal_perturbation(&mut state.temperature, &mut state.rng, params.noise_amp);
    }

    // Volumetric heat source at bottom hot spot + cooling at top.
    // Dirichlet BCs alone can't push heat into the interior (diff too small,
    // velocity zero at walls), so we inject heat directly into the first
    // few interior rows and apply Newtonian cooling near the top.
    inject_heat_source(&mut state.temperature, dt, params.source_strength, params.cool_rate);

    // Clamp temperature to physical bounds [0, 1]
    for t in state.temperature.iter_mut() {
        *t = t.clamp(0.0, 1.0);
    }

    // Advect particles through the divergence-free velocity field
    advect_particles(state, dt);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{idx, SimState, N, SIZE};

    const BB: f64 = 0.15;

    // === Sub-issue 3 tests: lin_solve, diffuse, set_bnd ===

    #[test]
    fn test_set_bnd_scalar_copies_neighbor() {
        let mut field = vec![0.0; SIZE];
        for i in 0..N {
            field[idx(i as i32, 1)] = 42.0;
            field[idx(i as i32, (N - 2) as i32)] = 99.0;
        }
        set_bnd(0, &mut field, BB);
        for i in 0..N {
            assert_eq!(field[idx(i as i32, 0)], 42.0, "Bottom boundary should copy y=1");
            assert_eq!(
                field[idx(i as i32, (N - 1) as i32)],
                99.0,
                "Top boundary should copy y=N-2"
            );
        }
    }

    #[test]
    fn test_set_bnd_vx_noslip() {
        let mut field = vec![0.0; SIZE];
        for i in 0..N {
            field[idx(i as i32, 1)] = 5.0;
            field[idx(i as i32, (N - 2) as i32)] = 3.0;
        }
        set_bnd(1, &mut field, BB);
        for i in 0..N {
            assert_eq!(field[idx(i as i32, 0)], -5.0, "vx should negate at bottom (no-slip)");
            assert_eq!(
                field[idx(i as i32, (N - 1) as i32)],
                -3.0,
                "vx should negate at top (no-slip)"
            );
        }
    }

    #[test]
    fn test_set_bnd_vy_negates() {
        let mut field = vec![0.0; SIZE];
        for i in 0..N {
            field[idx(i as i32, 1)] = 5.0;
            field[idx(i as i32, (N - 2) as i32)] = 3.0;
        }
        set_bnd(2, &mut field, BB);
        for i in 0..N {
            assert_eq!(field[idx(i as i32, 0)], -5.0, "vy should negate at bottom wall");
            assert_eq!(
                field[idx(i as i32, (N - 1) as i32)],
                -3.0,
                "vy should negate at top wall"
            );
        }
    }

    #[test]
    fn test_lin_solve_converges() {
        // Start with a spike and solve - should smooth out
        let mut x = vec![0.0; SIZE];
        let mut x0 = vec![0.0; SIZE];
        let mid = (N / 2) as i32;
        x0[idx(mid, mid)] = 100.0;
        x.copy_from_slice(&x0);

        lin_solve(0, &mut x, &x0, 1.0, 5.0, 20, BB);

        // The spike should have spread
        let center = x[idx(mid, mid)];
        let neighbor = x[idx(mid + 1, mid)];
        assert!(center > 0.0, "Center should still be positive");
        assert!(neighbor > 0.0, "Neighbors should get some value");
        assert!(center > neighbor, "Center should be larger than neighbor");
    }

    #[test]
    fn test_diffuse_smooths() {
        let mut x0 = vec![0.0; SIZE];
        let mut x = vec![0.0; SIZE];
        // Create sharp spike
        let mid = (N / 2) as i32;
        x0[idx(mid, mid)] = 100.0;

        diffuse(0, &mut x, &x0, 0.1, 0.1, 4, BB);

        // After diffusion, energy should spread
        let center = x[idx(mid, mid)];
        let neighbor = x[idx(mid + 1, mid)];
        assert!(center < 100.0, "Center should be less than original spike");
        assert!(neighbor > 0.0, "Neighbors should gain some value");
    }

    // === Sub-issue 4 tests: advect, project ===

    #[test]
    fn test_advect_zero_velocity_preserves() {
        let mut d0 = vec![0.0; SIZE];
        let mut d = vec![0.0; SIZE];
        let vx = vec![0.0; SIZE];
        let vy = vec![0.0; SIZE];

        // Set up a pattern
        for j in 1..(N - 1) {
            for i in 0..N {
                d0[idx(i as i32, j as i32)] = (i as f64) / N as f64;
            }
        }

        advect(0, &mut d, &d0, &vx, &vy, 0.1, BB);

        // With zero velocity, field should be (nearly) unchanged in interior
        for j in 2..(N - 2) {
            for i in 0..N {
                let orig = d0[idx(i as i32, j as i32)];
                let advected = d[idx(i as i32, j as i32)];
                assert!(
                    (orig - advected).abs() < 1e-10,
                    "Zero velocity should preserve field at ({}, {}): {} vs {}",
                    i,
                    j,
                    orig,
                    advected
                );
            }
        }
    }

    #[test]
    fn test_advect_uniform_field_unchanged() {
        let d0 = vec![5.0; SIZE];
        let mut d = vec![0.0; SIZE];
        let vx = vec![0.01; SIZE];
        let vy = vec![0.01; SIZE];

        advect(0, &mut d, &d0, &vx, &vy, 0.1, BB);

        // Uniform field should remain uniform regardless of velocity
        for j in 2..(N - 2) {
            for i in 0..N {
                let val = d[idx(i as i32, j as i32)];
                assert!(
                    (val - 5.0).abs() < 1e-6,
                    "Uniform field should stay uniform: got {} at ({}, {})",
                    val,
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_project_reduces_divergence() {
        let mut vx = vec![0.0; SIZE];
        let mut vy = vec![0.0; SIZE];
        let mut p = vec![0.0; SIZE];
        let mut div = vec![0.0; SIZE];

        // Create a source-sink pair (net divergence ≈ 0 for periodic compatibility)
        let cx = (N / 2) as i32;
        let cy = (N / 2) as i32;
        for j in 1..(N - 1) {
            for i in 0..N {
                let dx = i as f64 - cx as f64;
                let dy = j as f64 - cy as f64;
                let r2 = dx * dx + dy * dy;
                // Gaussian source at center
                let sigma = (N as f64 * N as f64) / 32.0;
                vx[idx(i as i32, j as i32)] = dx * 0.01 * (-r2 / sigma).exp();
                vy[idx(i as i32, j as i32)] = dy * 0.01 * (-r2 / sigma).exp();
            }
        }

        // Measure divergence before
        let mut div_before = 0.0;
        for j in 2..(N - 2) {
            for i in 0..N {
                let d = (vx[idx(i as i32 + 1, j as i32)] - vx[idx(i as i32 - 1, j as i32)])
                    + (vy[idx(i as i32, j as i32 + 1)] - vy[idx(i as i32, j as i32 - 1)]);
                div_before += d.abs();
            }
        }
        assert!(div_before > 0.0, "Should have some initial divergence");

        project(&mut vx, &mut vy, &mut p, &mut div, 40, BB);

        // Measure divergence after
        let mut div_after = 0.0;
        for j in 2..(N - 2) {
            for i in 0..N {
                let d = (vx[idx(i as i32 + 1, j as i32)] - vx[idx(i as i32 - 1, j as i32)])
                    + (vy[idx(i as i32, j as i32 + 1)] - vy[idx(i as i32, j as i32 - 1)]);
                div_after += d.abs();
            }
        }

        assert!(
            div_after < div_before,
            "Divergence should be reduced: before={}, after={}",
            div_before,
            div_after
        );
    }

    // === Sub-issue 5 tests: fluid_step, buoyancy, heat source ===

    #[test]
    fn test_fluid_step_no_panic() {
        let mut state = SimState::new(400, 0.15);
        let params = SolverParams::default();
        // Run a few steps - should not panic
        for _ in 0..3 {
            fluid_step(&mut state, &params);
        }
    }

    #[test]
    fn test_buoyancy_creates_upward_velocity() {
        let mut vy = vec![0.0; SIZE];
        let mut temperature = vec![0.5; SIZE];

        // Create a hot spot in the interior
        let mid = N / 2;
        for j in (mid - 5)..(mid + 5) {
            for i in (mid - 8)..(mid + 8) {
                temperature[idx(i as i32, j as i32)] = 1.0;
            }
        }

        apply_buoyancy(&mut vy, &temperature, 1.0, 1.0, BB);

        // Hot spot should have stronger upward vy than ambient
        let hot_vy = vy[idx(mid as i32, mid as i32)];
        assert!(hot_vy > 0.0, "Hot spot should have positive vy (upward): got {}", hot_vy);

        let ambient_vy = vy[idx(10, (mid / 2) as i32)];
        assert!(hot_vy > ambient_vy, "Hot spot vy should exceed ambient vy");
    }

    #[test]
    fn test_heat_source_boundaries() {
        let mut temperature = vec![0.5; SIZE];
        set_bnd(3, &mut temperature, BB);

        // Bottom: Gaussian hot spot at center (peak=1.0), base=BB at edges
        let center_t = temperature[idx((N / 2) as i32, 0)];
        assert!((center_t - 1.0).abs() < 1e-6, "Bottom center should be ~1.0, got {}", center_t);
        let edge_t = temperature[idx(0, 0)];
        assert!(edge_t >= BB - 0.01, "Bottom edge should be >= BB, got {}", edge_t);
        assert!(edge_t < center_t, "Bottom edge should be cooler than center, got {}", edge_t);

        // Top: cold
        for i in 0..N {
            assert!(temperature[idx(i as i32, (N - 1) as i32)].abs() < 1e-10, "y=N-1 should be 0.0");
            assert!(temperature[idx(i as i32, (N - 2) as i32)].abs() < 1e-10, "y=N-2 should be 0.0");
        }
        // y=2 should NOT be overwritten (remains 0.5)
        let t2 = temperature[idx(0, 2)];
        assert!((t2 - 0.5).abs() < 1e-10, "y=2 should remain interior value, got {}", t2);
    }

    // === Particle advection tests ===

    #[test]
    fn test_advect_particles_zero_velocity() {
        let mut state = SimState::new(400, 0.15);
        // Zero out velocity
        state.vx.fill(0.0);
        state.vy.fill(0.0);
        let orig_x = state.particles_x.clone();
        let orig_y = state.particles_y.clone();

        advect_particles(&mut state, 0.02);

        // Particles should not move
        for i in 0..state.particles_x.len() {
            assert!(
                (state.particles_x[i] - orig_x[i]).abs() < 1e-10,
                "Particle {} x moved with zero velocity",
                i
            );
            assert!(
                (state.particles_y[i] - orig_y[i]).abs() < 1e-10,
                "Particle {} y moved with zero velocity",
                i
            );
        }
    }

    #[test]
    fn test_advect_particles_uniform_velocity() {
        let mut state = SimState::new(400, 0.15);
        // Uniform rightward velocity
        state.vx.fill(0.01);
        state.vy.fill(0.0);
        let orig_x = state.particles_x.clone();

        advect_particles(&mut state, 0.02);

        // All particles should have moved right
        let dt0 = 0.02 * (N - 2) as f64;
        let expected_dx = dt0 * 0.01;
        for i in 0..state.particles_x.len() {
            let dx = ((state.particles_x[i] - orig_x[i]) + N as f64) % N as f64;
            assert!(
                (dx - expected_dx).abs() < 1e-6 || (dx - expected_dx + N as f64).abs() < 1e-6,
                "Particle {} dx={} expected ~{}",
                i, dx, expected_dx
            );
        }
    }

    #[test]
    fn test_advect_particles_stay_in_domain() {
        let mut state = SimState::new(400, 0.15);
        let params = SolverParams::default();

        // Run 50 steps with active flow
        for _ in 0..50 {
            fluid_step(&mut state, &params);
        }

        // All particles should still be in domain
        let n_f = N as f64;
        for i in 0..state.particles_x.len() {
            let px = state.particles_x[i];
            let py = state.particles_y[i];
            assert!(px >= 0.0 && px < n_f, "Particle {} x out of range: {}", i, px);
            assert!(py >= 2.0 && py <= n_f - 3.0, "Particle {} y out of range: {}", i, py);
        }
    }

    #[test]
    fn test_convection_maintains_gradient() {
        let mut state = SimState::new(400, 0.15);
        let params = SolverParams::default();

        // Run 100 steps
        for _ in 0..100 {
            fluid_step(&mut state, &params);
        }

        // Bottom boundary: Gaussian hot spot + cool base, avg should be above base
        let bottom_avg: f64 =
            (0..N).map(|x| state.temperature[idx(x as i32, 0)]).sum::<f64>() / N as f64;
        assert!(bottom_avg > BB, "Bottom avg should exceed base: {}", bottom_avg);

        // Top boundary should still be cold
        let top_avg: f64 =
            (0..N).map(|x| state.temperature[idx(x as i32, (N - 1) as i32)]).sum::<f64>() / N as f64;
        assert!(top_avg < 0.2, "Top should remain cold: {}", top_avg);

        // Interior mid-plane should have horizontal temperature variation (convection cells)
        let mid_y = N / 2;
        let temps: Vec<f64> =
            (0..N).map(|x| state.temperature[idx(x as i32, mid_y as i32)]).collect();
        let avg = temps.iter().sum::<f64>() / N as f64;
        let variance = temps.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / N as f64;
        assert!(
            variance > 1e-6,
            "Mid-plane should have horizontal variation (convection): variance={}",
            variance
        );
    }

    #[test]
    #[ignore] // diagnostic only — run with: cargo test test_diagnose -- --ignored --nocapture
    fn test_diagnose_vertical_convection() {
        let mut state = SimState::new(400, 0.15);
        let params = SolverParams::default();

        // Run 200 fluid steps
        for _ in 0..200 {
            fluid_step(&mut state, &params);
        }

        // --- Velocity magnitude diagnostics ---
        let mut max_vx: f64 = 0.0;
        let mut max_vy: f64 = 0.0;
        for j in 2..(N - 2) {
            for i in 0..N {
                let ii = idx(i as i32, j as i32);
                max_vx = max_vx.max(state.vx[ii].abs());
                max_vy = max_vy.max(state.vy[ii].abs());
            }
        }
        let ratio = if max_vx > 1e-30 { max_vy / max_vx } else { f64::NAN };
        eprintln!("=== Vertical Convection Diagnostics (after 200 steps) ===");
        eprintln!("max |vx| = {:.6e}", max_vx);
        eprintln!("max |vy| = {:.6e}", max_vy);
        eprintln!("ratio max_vy / max_vx = {:.4}", ratio);

        // --- Temperature profile diagnostics ---
        let y_quarter = N / 4;
        let y_mid = N / 2;
        let y_three_quarter = 3 * N / 4;

        let avg_temp_quarter: f64 = (0..N)
            .map(|x| state.temperature[idx(x as i32, y_quarter as i32)])
            .sum::<f64>()
            / N as f64;
        let avg_temp_mid: f64 = (0..N)
            .map(|x| state.temperature[idx(x as i32, y_mid as i32)])
            .sum::<f64>()
            / N as f64;
        let avg_temp_three_quarter: f64 = (0..N)
            .map(|x| state.temperature[idx(x as i32, y_three_quarter as i32)])
            .sum::<f64>()
            / N as f64;
        eprintln!(
            "avg T at y=N/4 (near bottom) = {:.6}",
            avg_temp_quarter
        );
        eprintln!("avg T at y=N/2 (middle)      = {:.6}", avg_temp_mid);
        eprintln!(
            "avg T at y=3N/4 (near top)   = {:.6}",
            avg_temp_three_quarter
        );

        // --- Horizontal temperature variance at midplane ---
        let mid_temps: Vec<f64> = (0..N)
            .map(|x| state.temperature[idx(x as i32, y_mid as i32)])
            .collect();
        let mid_avg = mid_temps.iter().sum::<f64>() / N as f64;
        let mid_variance =
            mid_temps.iter().map(|t| (t - mid_avg).powi(2)).sum::<f64>() / N as f64;
        eprintln!(
            "horizontal T variance at y=N/2 = {:.6e}",
            mid_variance
        );

        // --- Vertical velocity profile at x=N/2 ---
        let x_mid = (N / 2) as i32;
        eprintln!("vy profile at x=N/2 (sampled every 8 rows):");
        let mut j = 0;
        while j < N {
            let vy_val = state.vy[idx(x_mid, j as i32)];
            let t_val = state.temperature[idx(x_mid, j as i32)];
            eprintln!(
                "  y={:>4}  vy={:>+12.6e}  T={:.4}",
                j, vy_val, t_val
            );
            j += 8;
        }

        // --- Buoyancy vs projection survival analysis ---
        eprintln!("=== Buoyancy vs Projection Survival (step 201) ===");

        // Snapshot vy before buoyancy
        let vy_before = state.vy.clone();

        // Apply buoyancy manually
        apply_buoyancy(
            &mut state.vy,
            &state.temperature,
            params.heat_buoyancy,
            params.dt,
            params.bottom_base,
        );

        // Measure buoyancy contribution
        let mut max_buoyancy_delta: f64 = 0.0;
        for j in 2..(N - 2) {
            for i in 0..N {
                let ii = idx(i as i32, j as i32);
                let delta = (state.vy[ii] - vy_before[ii]).abs();
                max_buoyancy_delta = max_buoyancy_delta.max(delta);
            }
        }
        eprintln!(
            "max |vy_after_buoyancy - vy_before| = {:.6e}",
            max_buoyancy_delta
        );

        // Now apply projection
        project(
            &mut state.vx,
            &mut state.vy,
            &mut state.work,
            &mut state.work2,
            params.project_iter,
            params.bottom_base,
        );

        // Measure how much survived after projection
        let mut max_survived_delta: f64 = 0.0;
        for j in 2..(N - 2) {
            for i in 0..N {
                let ii = idx(i as i32, j as i32);
                let delta = (state.vy[ii] - vy_before[ii]).abs();
                max_survived_delta = max_survived_delta.max(delta);
            }
        }
        eprintln!(
            "max |vy_after_project - vy_before|  = {:.6e}",
            max_survived_delta
        );

        let survival_ratio = if max_buoyancy_delta > 1e-30 {
            max_survived_delta / max_buoyancy_delta
        } else {
            f64::NAN
        };
        eprintln!(
            "survival ratio (survived/applied)   = {:.6}",
            survival_ratio
        );

        // --- Horizontal structure: vy at y=N/2 across all x ---
        eprintln!("=== Horizontal vy structure at y=N/2 (sampled every 8 cols) ===");
        let mut vy_at_mid = Vec::new();
        for x in 0..N {
            vy_at_mid.push(state.vy[idx(x as i32, y_mid as i32)]);
        }
        let mut x = 0;
        while x < N {
            eprint!("  x={:>4} vy={:>+8.4} |", x, vy_at_mid[x]);
            x += 8;
        }
        eprintln!();

        // Count sign changes in vy at midplane (= number of convection cell boundaries)
        let mut sign_changes = 0;
        for x in 1..N {
            if vy_at_mid[x] * vy_at_mid[x - 1] < 0.0 {
                sign_changes += 1;
            }
        }
        eprintln!("vy sign changes at y=N/2: {} (≈ {} convection cells)", sign_changes, sign_changes / 2);

        // --- Average |vx| vs average |vy| in interior ---
        let mut sum_vx: f64 = 0.0;
        let mut sum_vy: f64 = 0.0;
        let mut count = 0usize;
        for j in 2..(N - 2) {
            for i in 0..N {
                let ii = idx(i as i32, j as i32);
                sum_vx += state.vx[ii].abs();
                sum_vy += state.vy[ii].abs();
                count += 1;
            }
        }
        eprintln!("avg |vx| = {:.6e}", sum_vx / count as f64);
        eprintln!("avg |vy| = {:.6e}", sum_vy / count as f64);
        eprintln!("ratio avg_vy / avg_vx = {:.4}", (sum_vy / count as f64) / (sum_vx / count as f64));

        // Diagnostic test: always passes
        assert!(true);
    }
}
