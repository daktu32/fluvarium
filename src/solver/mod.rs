mod boundary;
mod cavity;
mod core;
pub mod diagnostics;
mod karman;
mod kh;
mod params;
mod particle;
mod thermal;

// Re-export public API
pub use boundary::FieldType;
pub use params::SolverParams;
pub use thermal::inject_thermal_perturbation;

use crate::state::N;
use boundary::BoundaryConfig::{KarmanVortex, KelvinHelmholtz, LidDrivenCavity, RayleighBenard, RayleighBenardBenchmark};
use cavity::compute_velocity_dye;
use core::{advect, diffuse, project};
use karman::{apply_mask, apply_mask_fields, damp_dye_in_cylinder, inject_dye, inject_inflow, inject_wake_perturbation, vorticity_confinement};
use kh::reinject_shear;
use particle::{advect_particles, advect_particles_cavity, advect_particles_karman};
use thermal::{apply_buoyancy, apply_buoyancy_perturbation, inject_heat_source};

/// Full fluid simulation step (Rayleigh-Benard).
pub fn fluid_step(state: &mut crate::state::SimState, params: &SolverParams) {
    let dt = params.dt;
    let nx = state.nx;
    let bc = RayleighBenard { bottom_base: params.bottom_base };

    // Diffuse velocity
    diffuse(FieldType::Vx, &mut state.vx0, &state.vx, params.visc, dt, params.diffuse_iter, &bc, nx);
    diffuse(FieldType::Vy, &mut state.vy0, &state.vy, params.visc, dt, params.diffuse_iter, &bc, nx);

    // Project to make diffused velocity divergence-free
    project(
        &mut state.vx0,
        &mut state.vy0,
        &mut state.scratch_a,
        &mut state.scratch_b,
        params.project_iter,
        &bc,
        nx,
    );

    // Advect velocity
    advect(FieldType::Vx, &mut state.vx, &state.vx0, &state.vx0, &state.vy0, dt, &bc, nx);
    advect(FieldType::Vy, &mut state.vy, &state.vy0, &state.vx0, &state.vy0, dt, &bc, nx);

    // Diffuse + advect temperature BEFORE buoyancy so buoyancy uses T^(n+1)
    diffuse(FieldType::Temperature, &mut state.scratch_a, &state.temperature, params.diff, dt, params.diffuse_iter, &bc, nx);
    advect(FieldType::Temperature, &mut state.temperature, &state.scratch_a, &state.vx, &state.vy, dt, &bc, nx);

    // Volumetric heat source at bottom hot spot + cooling at top.
    inject_heat_source(&mut state.temperature, dt, params.source_strength, params.cool_rate, nx);

    // Apply buoyancy with T^(n+1) for better temporal coupling
    apply_buoyancy(&mut state.vy, &state.temperature, params.heat_buoyancy, dt, params.bottom_base, nx);

    // Project to make velocity divergence-free
    project(
        &mut state.vx,
        &mut state.vy,
        &mut state.scratch_a,
        &mut state.scratch_b,
        params.project_iter,
        &bc,
        nx,
    );

    // Note: thermal perturbation is now applied only at initialization (SimState::new),
    // not every step. Convection cells are self-sustaining after BC fix (#18).

    // Clamp temperature to physical bounds [0, 1]
    for t in state.temperature.iter_mut() {
        *t = t.clamp(0.0, 1.0);
    }

    // Advect particles through the divergence-free velocity field
    advect_particles(state, dt);
}

/// Full fluid simulation step for Rayleigh-Benard benchmark mode.
/// Uses uniform Dirichlet BCs and perturbation buoyancy (no Gaussian heat source).
pub fn fluid_step_benchmark(state: &mut crate::state::SimState, params: &SolverParams) {
    let dt = params.dt;
    let nx = state.nx;
    let bc = RayleighBenardBenchmark;

    // Diffuse velocity
    diffuse(FieldType::Vx, &mut state.vx0, &state.vx, params.visc, dt, params.diffuse_iter, &bc, nx);
    diffuse(FieldType::Vy, &mut state.vy0, &state.vy, params.visc, dt, params.diffuse_iter, &bc, nx);

    // Project to make diffused velocity divergence-free
    project(
        &mut state.vx0,
        &mut state.vy0,
        &mut state.scratch_a,
        &mut state.scratch_b,
        params.project_iter,
        &bc,
        nx,
    );

    // Advect velocity
    advect(FieldType::Vx, &mut state.vx, &state.vx0, &state.vx0, &state.vy0, dt, &bc, nx);
    advect(FieldType::Vy, &mut state.vy, &state.vy0, &state.vx0, &state.vy0, dt, &bc, nx);

    // Diffuse + advect temperature
    diffuse(FieldType::Temperature, &mut state.scratch_a, &state.temperature, params.diff, dt, params.diffuse_iter, &bc, nx);
    advect(FieldType::Temperature, &mut state.temperature, &state.scratch_a, &state.vx, &state.vy, dt, &bc, nx);

    // Perturbation buoyancy: vy += dt * B * (T - T_cond)
    apply_buoyancy_perturbation(&mut state.vy, &state.temperature, params.heat_buoyancy, dt, nx);

    // Project to make velocity divergence-free
    project(
        &mut state.vx,
        &mut state.vy,
        &mut state.scratch_a,
        &mut state.scratch_b,
        params.project_iter,
        &bc,
        nx,
    );

    // Clamp temperature to physical bounds [0, 1]
    for t in state.temperature.iter_mut() {
        *t = t.clamp(0.0, 1.0);
    }

    // Advect particles
    advect_particles(state, dt);
}

/// Full fluid simulation step for Karman vortex street.
pub fn fluid_step_karman(state: &mut crate::state::SimState, params: &SolverParams) {
    let dt = params.dt;
    let nx = state.nx;
    let bc = KarmanVortex { inflow_vel: params.inflow_vel };

    // Scale visc/diff by 1/(N-2) to fix the Re unit mismatch.
    // diffuse() computes a = dt * nu * (N-2)^2 where (N-2)^2 = 1/h^2 from the
    // Laplacian discretization. The displayed Re = U*D_grid/visc uses grid-cell
    // diameter D_grid, but physical D = D_grid*h = D_grid/(N-2). Compensating
    // one factor of (N-2) gives a = dt * visc * (N-2) -- moderate diffusion
    // yielding physical Re ~= displayed Re.
    let n2 = (N - 2) as f64;
    let visc_k = params.visc / n2;
    let diff_k = params.diff / n2;

    // 1. Inject inflow
    inject_inflow(state, params.inflow_vel);

    // 2. Diffuse velocity
    diffuse(FieldType::Vx, &mut state.vx0, &state.vx, visc_k, dt, params.diffuse_iter, &bc, nx);
    diffuse(FieldType::Vy, &mut state.vy0, &state.vy, visc_k, dt, params.diffuse_iter, &bc, nx);

    // 3. Project
    project(
        &mut state.vx0,
        &mut state.vy0,
        &mut state.scratch_a,
        &mut state.scratch_b,
        params.project_iter,
        &bc,
        nx,
    );

    // 4. Apply mask to projected velocity (vx0/vy0) before advection
    if let Some(ref mask) = state.mask {
        apply_mask_fields(&mut state.vx0, &mut state.vy0, mask);
    }

    // 5. Advect velocity
    advect(FieldType::Vx, &mut state.vx, &state.vx0, &state.vx0, &state.vy0, dt, &bc, nx);
    advect(FieldType::Vy, &mut state.vy, &state.vy0, &state.vx0, &state.vy0, dt, &bc, nx);

    // 6. Project
    project(
        &mut state.vx,
        &mut state.vy,
        &mut state.scratch_a,
        &mut state.scratch_b,
        params.project_iter,
        &bc,
        nx,
    );

    // 7. Apply mask
    apply_mask(state);

    // 7.5. Vorticity confinement -- counteract numerical diffusion
    vorticity_confinement(state, params.confinement, dt);
    apply_mask(state);

    // 7.6. Tiny wake perturbation to trigger vortex shedding.
    inject_wake_perturbation(state, params);

    // 8. Diffuse dye (using temperature field as dye)
    diffuse(FieldType::Scalar, &mut state.scratch_a, &state.temperature, diff_k, dt, params.diffuse_iter, &bc, nx);

    // 9. Advect dye
    advect(FieldType::Scalar, &mut state.temperature, &state.scratch_a, &state.vx, &state.vy, dt, &bc, nx);

    // 10. Inject dye at inflow
    inject_dye(state);

    // 11. Damp dye inside cylinder
    damp_dye_in_cylinder(state);

    // 12. Clamp dye
    for t in state.temperature.iter_mut() {
        *t = t.clamp(0.0, 1.0);
    }

    // 13. Advect particles
    advect_particles_karman(state, dt);
}

/// Full fluid simulation step for Kelvin-Helmholtz instability.
pub fn fluid_step_kh(state: &mut crate::state::SimState, params: &SolverParams) {
    let dt = params.dt;
    let nx = state.nx;
    let bc = KelvinHelmholtz;

    // 1. Reinject shear profile (counteract numerical diffusion)
    reinject_shear(state, params);

    // 2. Diffuse velocity
    diffuse(FieldType::Vx, &mut state.vx0, &state.vx, params.visc, dt, params.diffuse_iter, &bc, nx);
    diffuse(FieldType::Vy, &mut state.vy0, &state.vy, params.visc, dt, params.diffuse_iter, &bc, nx);

    // 3. Project (divergence-free)
    project(&mut state.vx0, &mut state.vy0, &mut state.scratch_a, &mut state.scratch_b, params.project_iter, &bc, nx);

    // 4. Advect velocity
    advect(FieldType::Vx, &mut state.vx, &state.vx0, &state.vx0, &state.vy0, dt, &bc, nx);
    advect(FieldType::Vy, &mut state.vy, &state.vy0, &state.vx0, &state.vy0, dt, &bc, nx);

    // 5. Project (divergence-free)
    project(&mut state.vx, &mut state.vy, &mut state.scratch_a, &mut state.scratch_b, params.project_iter, &bc, nx);

    // 5.5. Vorticity confinement -- counteract numerical diffusion
    if params.confinement > 0.0 {
        vorticity_confinement(state, params.confinement, dt);
    }

    // 6. Diffuse + advect dye (temperature as passive tracer)
    diffuse(FieldType::Scalar, &mut state.scratch_a, &state.temperature, params.diff, dt, params.diffuse_iter, &bc, nx);
    advect(FieldType::Scalar, &mut state.temperature, &state.scratch_a, &state.vx, &state.vy, dt, &bc, nx);

    // 7. Clamp dye
    for t in state.temperature.iter_mut() {
        *t = t.clamp(0.0, 1.0);
    }

    // 9. Advect particles (periodic X, reflected Y -- same as RB)
    advect_particles(state, dt);
}

/// Full fluid simulation step for Lid-Driven Cavity.
pub fn fluid_step_cavity(state: &mut crate::state::SimState, params: &SolverParams) {
    let dt = params.dt;
    let nx = state.nx;
    let bc = LidDrivenCavity { lid_velocity: params.lid_velocity };

    // 1. Diffuse velocity
    diffuse(FieldType::Vx, &mut state.vx0, &state.vx, params.visc, dt, params.diffuse_iter, &bc, nx);
    diffuse(FieldType::Vy, &mut state.vy0, &state.vy, params.visc, dt, params.diffuse_iter, &bc, nx);

    // 2. Project (divergence-free)
    project(&mut state.vx0, &mut state.vy0, &mut state.scratch_a, &mut state.scratch_b, params.project_iter, &bc, nx);

    // 3. Advect velocity
    advect(FieldType::Vx, &mut state.vx, &state.vx0, &state.vx0, &state.vy0, dt, &bc, nx);
    advect(FieldType::Vy, &mut state.vy, &state.vy0, &state.vx0, &state.vy0, dt, &bc, nx);

    // 4. Project again
    project(&mut state.vx, &mut state.vy, &mut state.scratch_a, &mut state.scratch_b, params.project_iter, &bc, nx);

    // 5. Compute velocity magnitude as visualization dye
    compute_velocity_dye(state);

    // 6. Advect particles (reflected X and Y -- solid walls on all sides)
    advect_particles_cavity(state, dt);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::boundary::BoundaryConfig;
    use crate::state::{idx, SimState, N};

    const BB: f64 = 0.15;

    fn rb_bc() -> BoundaryConfig {
        BoundaryConfig::RayleighBenard { bottom_base: BB }
    }

    #[test]
    fn test_fluid_step_no_panic() {
        let mut state = SimState::new(400, 0.15, N);
        let params = SolverParams::default();
        // Run a few steps - should not panic
        for _ in 0..3 {
            fluid_step(&mut state, &params);
        }
    }

    #[test]
    fn test_convection_maintains_gradient() {
        let mut state = SimState::new(400, 0.15, N);
        let params = SolverParams::default();

        // Run 100 steps
        for _ in 0..100 {
            fluid_step(&mut state, &params);
        }

        // Bottom boundary: Gaussian hot spot + cool base, avg should be above base
        let bottom_avg: f64 =
            (0..N).map(|x| state.temperature[idx(x as i32, 0, N)]).sum::<f64>() / N as f64;
        assert!(bottom_avg > BB, "Bottom avg should exceed base: {}", bottom_avg);

        // Top boundary should still be cold
        let top_avg: f64 =
            (0..N).map(|x| state.temperature[idx(x as i32, (N - 1) as i32, N)]).sum::<f64>() / N as f64;
        assert!(top_avg < 0.2, "Top should remain cold: {}", top_avg);

        // Interior mid-plane should have horizontal temperature variation (convection cells)
        let mid_y = N / 2;
        let temps: Vec<f64> =
            (0..N).map(|x| state.temperature[idx(x as i32, mid_y as i32, N)]).collect();
        let avg = temps.iter().sum::<f64>() / N as f64;
        let variance = temps.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / N as f64;
        assert!(
            variance > 1e-6,
            "Mid-plane should have horizontal variation (convection): variance={}",
            variance
        );
    }

    #[test]
    #[ignore] // diagnostic only -- run with: cargo test test_diagnose -- --ignored --nocapture
    fn test_diagnose_vertical_convection() {
        let mut state = SimState::new(400, 0.15, N);
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
                let ii = idx(i as i32, j as i32, N);
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
            .map(|x| state.temperature[idx(x as i32, y_quarter as i32, N)])
            .sum::<f64>()
            / N as f64;
        let avg_temp_mid: f64 = (0..N)
            .map(|x| state.temperature[idx(x as i32, y_mid as i32, N)])
            .sum::<f64>()
            / N as f64;
        let avg_temp_three_quarter: f64 = (0..N)
            .map(|x| state.temperature[idx(x as i32, y_three_quarter as i32, N)])
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
            .map(|x| state.temperature[idx(x as i32, y_mid as i32, N)])
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
            let vy_val = state.vy[idx(x_mid, j as i32, N)];
            let t_val = state.temperature[idx(x_mid, j as i32, N)];
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
            N,
        );

        // Measure buoyancy contribution
        let mut max_buoyancy_delta: f64 = 0.0;
        for j in 2..(N - 2) {
            for i in 0..N {
                let ii = idx(i as i32, j as i32, N);
                let delta = (state.vy[ii] - vy_before[ii]).abs();
                max_buoyancy_delta = max_buoyancy_delta.max(delta);
            }
        }
        eprintln!(
            "max |vy_after_buoyancy - vy_before| = {:.6e}",
            max_buoyancy_delta
        );

        // Now apply projection
        let bc = rb_bc();
        project(
            &mut state.vx,
            &mut state.vy,
            &mut state.scratch_a,
            &mut state.scratch_b,
            params.project_iter,
            &bc,
            N,
        );

        // Measure how much survived after projection
        let mut max_survived_delta: f64 = 0.0;
        for j in 2..(N - 2) {
            for i in 0..N {
                let ii = idx(i as i32, j as i32, N);
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
            vy_at_mid.push(state.vy[idx(x as i32, y_mid as i32, N)]);
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
        eprintln!("vy sign changes at y=N/2: {} (~= {} convection cells)", sign_changes, sign_changes / 2);

        // --- Average |vx| vs average |vy| in interior ---
        let mut sum_vx: f64 = 0.0;
        let mut sum_vy: f64 = 0.0;
        let mut count = 0usize;
        for j in 2..(N - 2) {
            for i in 0..N {
                let ii = idx(i as i32, j as i32, N);
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

    #[test]
    fn test_convection_self_sustaining_without_noise() {
        let mut state = SimState::new(400, 0.15, N);
        let mut params = SolverParams::default();

        // Initial 100 steps with perturbation to form convection cells
        for _ in 0..100 {
            fluid_step(&mut state, &params);
        }

        // Stop perturbation and run 1000 steps
        params.noise_amp = 0.0;
        for _ in 0..1000 {
            fluid_step(&mut state, &params);
        }

        // Mid-layer temperature variance should persist (convection is self-sustaining)
        let mid_y = N / 2;
        let temps: Vec<f64> = (0..N).map(|x| state.temperature[idx(x as i32, mid_y as i32, N)]).collect();
        let avg = temps.iter().sum::<f64>() / N as f64;
        let variance = temps.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / N as f64;
        assert!(variance > 1e-6, "Convection should persist without noise: variance={}", variance);
    }

    #[test]
    fn test_fluid_step_benchmark_no_panic() {
        let params = SolverParams::from_ra_pr(10000.0, 1.0);
        let mut state = SimState::new_benchmark(100, N);
        for _ in 0..10 {
            fluid_step_benchmark(&mut state, &params);
        }
    }

    #[test]
    fn test_benchmark_develops_convection() {
        let params = SolverParams::from_ra_pr(10000.0, 1.0);
        let mut state = SimState::new_benchmark(10, N);
        for _ in 0..200 {
            fluid_step_benchmark(&mut state, &params);
        }
        // After 200 steps, should have non-trivial flow
        let max_speed: f64 = state.vx.iter().zip(state.vy.iter())
            .map(|(vx, vy)| (vx * vx + vy * vy).sqrt())
            .fold(0.0_f64, f64::max);
        assert!(max_speed > 1e-6, "Benchmark should develop flow, max_speed={}", max_speed);
    }

    #[test]
    fn test_benchmark_boundary_preserved() {
        let params = SolverParams::from_ra_pr(10000.0, 1.0);
        let mut state = SimState::new_benchmark(10, N);
        for _ in 0..50 {
            fluid_step_benchmark(&mut state, &params);
        }
        // Bottom T should be 1.0 (Dirichlet)
        for i in 0..N {
            let t = state.temperature[idx(i as i32, 0, N)];
            assert!((t - 1.0).abs() < 0.1, "Bottom T should be ~1.0, got {} at x={}", t, i);
        }
        // Top T should be 0.0
        for i in 0..N {
            let t = state.temperature[idx(i as i32, (N - 1) as i32, N)];
            assert!(t < 0.1, "Top T should be ~0.0, got {} at x={}", t, i);
        }
    }

    #[test]
    fn test_qa_fluid_step_kh_no_panic() {
        let params = SolverParams::default_kh();
        let mut state = SimState::new_kh(100, &params, N);
        for _ in 0..10 {
            fluid_step_kh(&mut state, &params);
        }
        // If we reach here, no panic occurred
    }

    #[test]
    fn test_qa_kh_shear_maintained() {
        let params = SolverParams::default_kh();
        let mut state = SimState::new_kh(100, &params, N);
        for _ in 0..100 {
            fluid_step_kh(&mut state, &params);
        }
        // Average vx in top quarter should be positive
        let top_avg: f64 = (0..N)
            .map(|x| state.vx[idx(x as i32, (3 * N / 4) as i32, N)])
            .sum::<f64>() / N as f64;
        assert!(top_avg > 0.0, "Top quarter average vx should be positive after 100 steps, got {}", top_avg);
        // Average vx in bottom quarter should be negative
        let bot_avg: f64 = (0..N)
            .map(|x| state.vx[idx(x as i32, (N / 4) as i32, N)])
            .sum::<f64>() / N as f64;
        assert!(bot_avg < 0.0, "Bottom quarter average vx should be negative after 100 steps, got {}", bot_avg);
    }

    #[test]
    fn test_fluid_step_cavity_no_panic() {
        let params = SolverParams::default_cavity();
        let mut state = SimState::new_cavity(100, N);
        for _ in 0..10 {
            fluid_step_cavity(&mut state, &params);
        }
        // If we reach here, no panic occurred
    }

    #[test]
    fn test_cavity_develops_flow() {
        let params = SolverParams::default_cavity();
        let mut state = SimState::new_cavity(10, N);
        for _ in 0..50 {
            fluid_step_cavity(&mut state, &params);
        }
        // After 50 steps with lid driving, interior should have non-zero velocity
        let max_speed: f64 = state.vx.iter().zip(state.vy.iter())
            .map(|(vx, vy)| (vx * vx + vy * vy).sqrt())
            .fold(0.0_f64, f64::max);
        assert!(max_speed > 0.01, "Cavity should develop flow from lid driving, max_speed={}", max_speed);
    }
}
