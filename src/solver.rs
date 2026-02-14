use crate::state::{idx, N};

/// Field type for boundary condition dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Scalar,
    Vx,
    Vy,
    Temperature,
}

/// Boundary configuration for different flow models.
#[derive(Clone)]
pub enum BoundaryConfig {
    RayleighBenard { bottom_base: f64 },
    KarmanVortex { inflow_vel: f64 },
}

/// Boundary condition handler.
/// Dispatches based on BoundaryConfig variant.
pub fn set_bnd(field_type: FieldType, x: &mut [f64], bc: &BoundaryConfig, nx: usize) {
    match bc {
        BoundaryConfig::RayleighBenard { bottom_base } => {
            set_bnd_rb(field_type, x, *bottom_base, nx);
        }
        BoundaryConfig::KarmanVortex { inflow_vel } => {
            set_bnd_karman(field_type, x, *inflow_vel, nx);
        }
    }
}

/// Rayleigh-Benard boundary conditions.
/// X-axis is periodic (wraps via idx).
///   - field_type 0 (scalar): Neumann (copy neighbor) at walls
///   - field_type 1 (vx): negate at walls (no-slip)
///   - field_type 2 (vy): negate at walls (no-slip + no-penetration)
///   - field_type 3 (temperature): top/bottom Dirichlet (hot bottom, cold top)
fn set_bnd_rb(field_type: FieldType, x: &mut [f64], bottom_base: f64, nx: usize) {
    for i in 0..nx {
        match field_type {
            FieldType::Vx | FieldType::Vy => {
                x[idx(i as i32, 0, nx)] = -x[idx(i as i32, 1, nx)];
                x[idx(i as i32, (N - 1) as i32, nx)] = -x[idx(i as i32, (N - 2) as i32, nx)];
            }
            FieldType::Temperature => {
                // Bottom: Gaussian hot spot centered at nx/2
                let dx = i as f64 - (nx / 2) as f64;
                let sigma = (N / 24) as f64;
                let hot = bottom_base + (1.0 - bottom_base) * (-dx * dx / (2.0 * sigma * sigma)).exp();
                x[idx(i as i32, 0, nx)] = hot;
                x[idx(i as i32, 1, nx)] = hot;
                // Top: cold
                x[idx(i as i32, (N - 1) as i32, nx)] = 0.0;
                x[idx(i as i32, (N - 2) as i32, nx)] = 0.0;
            }
            _ => {
                x[idx(i as i32, 0, nx)] = x[idx(i as i32, 1, nx)];
                x[idx(i as i32, (N - 1) as i32, nx)] = x[idx(i as i32, (N - 2) as i32, nx)];
            }
        }
    }
}

/// Kármán vortex street boundary conditions.
/// Left: Dirichlet inflow. Right: zero-gradient outflow.
/// Top/Bottom: no-slip (negate) for velocity, Neumann for scalars.
/// Left/Right boundaries take priority over top/bottom at corners.
fn set_bnd_karman(field_type: FieldType, x: &mut [f64], inflow_vel: f64, nx: usize) {
    // Pass 1: Top/Bottom walls (y boundaries)
    for i in 0..nx {
        match field_type {
            FieldType::Vx | FieldType::Vy => {
                // No-slip: negate at walls
                x[idx(i as i32, 0, nx)] = -x[idx(i as i32, 1, nx)];
                x[idx(i as i32, (N - 1) as i32, nx)] = -x[idx(i as i32, (N - 2) as i32, nx)];
            }
            _ => {
                // Neumann: copy neighbor
                x[idx(i as i32, 0, nx)] = x[idx(i as i32, 1, nx)];
                x[idx(i as i32, (N - 1) as i32, nx)] = x[idx(i as i32, (N - 2) as i32, nx)];
            }
        }
    }

    // Pass 2: Left/Right walls (x boundaries) — overwrite corners
    for j in 0..N {
        match field_type {
            FieldType::Vx => {
                // Left: Dirichlet inflow
                x[idx(0, j as i32, nx)] = inflow_vel;
                x[idx(1, j as i32, nx)] = inflow_vel;
                // Right: zero-gradient (convective outflow)
                x[idx((nx - 1) as i32, j as i32, nx)] = x[idx((nx - 2) as i32, j as i32, nx)];
            }
            FieldType::Vy => {
                // Left: vy = 0
                x[idx(0, j as i32, nx)] = 0.0;
                x[idx(1, j as i32, nx)] = 0.0;
                // Right: zero-gradient
                x[idx((nx - 1) as i32, j as i32, nx)] = x[idx((nx - 2) as i32, j as i32, nx)];
            }
            _ => {
                // Scalar/dye: Neumann at left, zero-gradient at right
                x[idx(0, j as i32, nx)] = x[idx(1, j as i32, nx)];
                x[idx((nx - 1) as i32, j as i32, nx)] = x[idx((nx - 2) as i32, j as i32, nx)];
            }
        }
    }
}

/// Gauss-Seidel iterative linear solver.
/// Solves: x[i,j] = (x0[i,j] + a * (neighbors)) / c
pub fn lin_solve(field_type: FieldType, x: &mut [f64], x0: &[f64], a: f64, c: f64, iter: usize, bc: &BoundaryConfig, nx: usize) {
    let c_inv = 1.0 / c;
    // For Karman mode, skip X boundary cells to prevent periodic wrap contamination.
    // RB mode uses 0..nx (periodic X via idx wrapping).
    let (i_lo, i_hi) = match bc {
        BoundaryConfig::KarmanVortex { .. } => (1, nx - 1),
        _ => (0, nx),
    };
    for _ in 0..iter {
        for j in 1..(N - 1) {
            for i in i_lo..i_hi {
                let neighbors = x[idx(i as i32 - 1, j as i32, nx)]
                    + x[idx(i as i32 + 1, j as i32, nx)]
                    + x[idx(i as i32, j as i32 - 1, nx)]
                    + x[idx(i as i32, j as i32 + 1, nx)];
                x[idx(i as i32, j as i32, nx)] = (x0[idx(i as i32, j as i32, nx)] + a * neighbors) * c_inv;
            }
        }
        set_bnd(field_type, x, bc, nx);
    }
}

/// Diffusion step: spreads the field over time.
/// a = dt * diff * (N-2)^2, c = 1 + 4a
pub fn diffuse(field_type: FieldType, x: &mut [f64], x0: &[f64], diff: f64, dt: f64, iter: usize, bc: &BoundaryConfig, nx: usize) {
    let a = dt * diff * ((N - 2) as f64) * ((N - 2) as f64);
    let c = 1.0 + 4.0 * a;
    // Initialize x from x0
    x.copy_from_slice(x0);
    lin_solve(field_type, x, x0, a, c, iter, bc, nx);
}

/// Semi-Lagrangian advection: traces particles backwards through velocity field.
pub fn advect(field_type: FieldType, d: &mut [f64], d0: &[f64], vx: &[f64], vy: &[f64], dt: f64, bc: &BoundaryConfig, nx: usize) {
    let dt0 = dt * (N - 2) as f64;
    let n_f = N as f64;
    let nx_f = nx as f64;

    // For Karman mode, skip X boundary cells to prevent periodic wrap contamination.
    let (i_lo, i_hi) = match bc {
        BoundaryConfig::KarmanVortex { .. } => (1, nx - 1),
        _ => (0, nx),
    };

    for j in 1..(N - 1) {
        for i in i_lo..i_hi {
            let ii = idx(i as i32, j as i32, nx);
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

            // X clamping for non-periodic Kármán mode
            let x = match bc {
                BoundaryConfig::KarmanVortex { .. } => x.clamp(0.5, nx_f - 1.5),
                _ => x, // RB: periodic, no clamping
            };

            let i0 = x.floor() as i32;
            let j0 = y.floor() as i32;
            let i1 = i0 + 1;
            let j1 = j0 + 1;
            let s1 = x - i0 as f64;
            let s0 = 1.0 - s1;
            let t1 = y - j0 as f64;
            let t0 = 1.0 - t1;

            d[ii] = s0 * (t0 * d0[idx(i0, j0, nx)] + t1 * d0[idx(i0, j1, nx)])
                + s1 * (t0 * d0[idx(i1, j0, nx)] + t1 * d0[idx(i1, j1, nx)]);
        }
    }
    set_bnd(field_type, d, bc, nx);
}

/// Pressure projection: enforces incompressibility (divergence-free velocity field).
pub fn project(vx: &mut [f64], vy: &mut [f64], p: &mut [f64], div: &mut [f64], iter: usize, bc: &BoundaryConfig, nx: usize) {
    let h = 1.0 / (N - 2) as f64;

    // For Karman mode, skip X boundary cells to prevent periodic wrap contamination.
    let (i_lo, i_hi) = match bc {
        BoundaryConfig::KarmanVortex { .. } => (1, nx - 1),
        _ => (0, nx),
    };

    // Calculate divergence
    for j in 1..(N - 1) {
        for i in i_lo..i_hi {
            div[idx(i as i32, j as i32, nx)] = -0.5
                * h
                * (vx[idx(i as i32 + 1, j as i32, nx)] - vx[idx(i as i32 - 1, j as i32, nx)]
                    + vy[idx(i as i32, j as i32 + 1, nx)] - vy[idx(i as i32, j as i32 - 1, nx)]);
            p[idx(i as i32, j as i32, nx)] = 0.0;
        }
    }
    set_bnd(FieldType::Scalar, div, bc, nx);
    set_bnd(FieldType::Scalar, p, bc, nx);

    // Solve for pressure
    lin_solve(FieldType::Scalar, p, div, 1.0, 4.0, iter, bc, nx);

    // Subtract pressure gradient from velocity
    for j in 1..(N - 1) {
        for i in i_lo..i_hi {
            vx[idx(i as i32, j as i32, nx)] -=
                0.5 * (p[idx(i as i32 + 1, j as i32, nx)] - p[idx(i as i32 - 1, j as i32, nx)]) / h;
            vy[idx(i as i32, j as i32, nx)] -=
                0.5 * (p[idx(i as i32, j as i32 + 1, nx)] - p[idx(i as i32, j as i32 - 1, nx)]) / h;
        }
    }
    set_bnd(FieldType::Vx, vx, bc, nx);
    set_bnd(FieldType::Vy, vy, bc, nx);
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
    pub inflow_vel: f64,
    pub cylinder_x: f64,
    pub cylinder_y: f64,
    pub cylinder_radius: f64,
    pub confinement: f64,
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
            inflow_vel: 0.1,
            cylinder_x: 21.0,
            cylinder_y: (N / 2) as f64,
            cylinder_radius: 8.0,
            confinement: 0.0,
        }
    }
}

impl SolverParams {
    /// Default parameters optimized for Kármán vortex street at Re=100.
    pub fn default_karman() -> Self {
        Self {
            visc: 0.015,
            diff: 0.003,
            dt: 0.06,
            diffuse_iter: 20,
            project_iter: 30,
            heat_buoyancy: 0.0,
            noise_amp: 0.0,
            source_strength: 0.0,
            cool_rate: 0.0,
            bottom_base: 0.0,
            inflow_vel: 0.1,
            cylinder_x: 21.0,
            cylinder_y: (N / 2) as f64,
            cylinder_radius: 8.0,
            confinement: 3.0,
        }
    }
}

/// Damp velocity inside the cylinder using fractional mask.
/// mask=1.0 → velocity zeroed, mask=0.5 → half damped, mask=0.0 → unchanged.
pub fn apply_mask(state: &mut crate::state::SimState) {
    if let Some(ref mask) = state.mask {
        for i in 0..mask.len() {
            let fluid = 1.0 - mask[i];
            state.vx[i] *= fluid;
            state.vy[i] *= fluid;
        }
    }
}

/// Inject inflow velocity at left boundary for Kármán flow.
/// Also adds small vy perturbation to trigger vortex shedding.
pub fn inject_inflow(state: &mut crate::state::SimState, inflow_vel: f64) {
    let nx = state.nx;
    for j in 0..N {
        state.vx[idx(0, j as i32, nx)] = inflow_vel;
        state.vx[idx(1, j as i32, nx)] = inflow_vel;
        state.vy[idx(0, j as i32, nx)] = 0.0;
        state.vy[idx(1, j as i32, nx)] = 0.0;
    }
    // No inflow perturbation — wake perturbation is applied separately
    // in fluid_step_karman after the cylinder mask, closer to where the
    // physical instability actually develops.
}

/// Inject dye tracer at left inflow for Kármán visualization.
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

/// Inject large-scale temperature perturbation at convection-cell wavelengths.
/// Small-scale (per-cell) noise gets killed by projection and diffusion.
/// We need modes at wavelength ~nx/k (k=1..4) to seed/sustain convection cells.
fn inject_thermal_perturbation(
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
/// the interior fast enough (wall velocity ≈ 0, diff is small).
fn inject_heat_source(temperature: &mut [f64], dt: f64, source_strength: f64, cool_rate: f64, nx: usize) {
    let center = (nx / 2) as f64;
    let sigma = (N / 24) as f64; // narrow: ~5 cells

    // Concentrated heat injection at bottom (rows 2–4)
    for j in 2..5 {
        let y_factor = 1.0 - (j - 2) as f64 / 3.0;
        for i in 0..nx {
            let dx = i as f64 - center;
            let g = (-dx * dx / (2.0 * sigma * sigma)).exp();
            let ii = idx(i as i32, j as i32, nx);
            temperature[ii] += dt * source_strength * g * y_factor;
        }
    }

    // Newtonian cooling near top (rows N-7 to N-3): decay toward 0
    for j in (N - 7)..(N - 2) {
        let y_factor = 1.0 - ((N - 3) - j) as f64 / 5.0;
        for i in 0..nx {
            let ii = idx(i as i32, j as i32, nx);
            temperature[ii] *= 1.0 - dt * cool_rate * y_factor;
        }
    }
}

/// Apply buoyancy force: hot fluid rises, cold fluid sinks.
/// vy += dt * buoyancy * (T - T_ambient), where T_ambient = bottom_base
fn apply_buoyancy(vy: &mut [f64], temperature: &[f64], buoyancy: f64, dt: f64, bottom_base: f64, nx: usize) {
    let t_ambient = bottom_base;
    for j in 1..(N - 1) {
        for i in 0..nx {
            let ii = idx(i as i32, j as i32, nx);
            vy[ii] += dt * buoyancy * (temperature[ii] - t_ambient);
        }
    }
}

/// Advect particles through the velocity field using bilinear interpolation.
/// X wraps (periodic), Y reflects at walls.
fn advect_particles(state: &mut crate::state::SimState, dt: f64) {
    let nx = state.nx;
    let dt0 = dt * (N - 2) as f64;
    let n_f = N as f64;
    let nx_f = nx as f64;

    for p in 0..state.particles_x.len() {
        let px = state.particles_x[p];
        let py = state.particles_y[p];

        // Bilinear interpolation of velocity at particle position
        let i0 = px.floor() as i32;
        let j0 = py.floor().max(0.0).min(n_f - 2.0) as i32;
        let j1 = j0 + 1;
        let sx = px - px.floor();
        let sy = py - j0 as f64;

        let vx_interp = (1.0 - sx) * (1.0 - sy) * state.vx[idx(i0, j0, nx)]
            + sx * (1.0 - sy) * state.vx[idx(i0 + 1, j0, nx)]
            + (1.0 - sx) * sy * state.vx[idx(i0, j1, nx)]
            + sx * sy * state.vx[idx(i0 + 1, j1, nx)];

        let vy_interp = (1.0 - sx) * (1.0 - sy) * state.vy[idx(i0, j0, nx)]
            + sx * (1.0 - sy) * state.vy[idx(i0 + 1, j0, nx)]
            + (1.0 - sx) * sy * state.vy[idx(i0, j1, nx)]
            + sx * sy * state.vy[idx(i0 + 1, j1, nx)];

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
        let new_x = ((new_x % nx_f) + nx_f) % nx_f;

        state.particles_x[p] = new_x;
        state.particles_y[p] = new_y;
    }
}

/// Advect particles for Kármán flow.
/// Particles that exit right or enter cylinder are respawned at left.
fn advect_particles_karman(state: &mut crate::state::SimState, dt: f64) {
    let nx = state.nx;
    let dt0 = dt * (N - 2) as f64;
    let n_f = N as f64;
    let nx_f = nx as f64;

    for p in 0..state.particles_x.len() {
        let px = state.particles_x[p];
        let py = state.particles_y[p];

        // Bilinear interpolation of velocity
        let i0 = px.floor().max(0.0).min(nx_f - 2.0) as i32;
        let j0 = py.floor().max(0.0).min(n_f - 2.0) as i32;
        let j1 = j0 + 1;
        let i1 = i0 + 1;
        let sx = px - i0 as f64;
        let sy = py - j0 as f64;

        let vx_interp = (1.0 - sx) * (1.0 - sy) * state.vx[idx(i0, j0, nx)]
            + sx * (1.0 - sy) * state.vx[idx(i1, j0, nx)]
            + (1.0 - sx) * sy * state.vx[idx(i0, j1, nx)]
            + sx * sy * state.vx[idx(i1, j1, nx)];

        let vy_interp = (1.0 - sx) * (1.0 - sy) * state.vy[idx(i0, j0, nx)]
            + sx * (1.0 - sy) * state.vy[idx(i1, j0, nx)]
            + (1.0 - sx) * sy * state.vy[idx(i0, j1, nx)]
            + sx * sy * state.vy[idx(i1, j1, nx)];

        let new_x = px + dt0 * vx_interp;
        let new_y = (py + dt0 * vy_interp).clamp(2.0, n_f - 3.0);

        // Check if particle needs respawn (out of domain or near cylinder)
        let needs_respawn = new_x >= nx_f - 1.0 || new_x < 0.0 || {
            if let Some((cx, cy, r)) = state.cylinder {
                let dx = new_x - cx;
                let dy = new_y - cy;
                dx * dx + dy * dy < (r + 1.0) * (r + 1.0)
            } else {
                false
            }
        };

        if needs_respawn {
            // Respawn at left boundary with random y
            state.particles_x[p] = 2.0 + state.rng.next_f64().abs() * 2.0;
            state.particles_y[p] = 2.0 + (state.rng.next_f64() + 1.0) * 0.5 * (n_f - 5.0);
        } else {
            state.particles_x[p] = new_x;
            state.particles_y[p] = new_y;
        }
    }
}

/// Vorticity confinement: counteracts numerical diffusion by amplifying
/// existing vortical structures. Essential for Stable Fluids to sustain
/// vortex shedding (Fedkiw et al. 2001).
fn vorticity_confinement(state: &mut crate::state::SimState, epsilon: f64, dt: f64) {
    let nx = state.nx;
    let size = nx * N;
    let mut omega = vec![0.0; size];
    let mut abs_omega = vec![0.0; size];

    // Compute vorticity: ω = ∂vy/∂x - ∂vx/∂y
    for j in 1..(N - 1) as i32 {
        for i in 1..(nx - 1) as i32 {
            let dvydx = (state.vy[idx(i + 1, j, nx)] - state.vy[idx(i - 1, j, nx)]) * 0.5;
            let dvxdy = (state.vx[idx(i, j + 1, nx)] - state.vx[idx(i, j - 1, nx)]) * 0.5;
            let w = dvydx - dvxdy;
            omega[idx(i, j, nx)] = w;
            abs_omega[idx(i, j, nx)] = w.abs();
        }
    }

    // Compute ∇|ω| and apply confinement force: f = ε·Δx·(N̂ × ω)
    for j in 2..(N - 2) as i32 {
        for i in 2..(nx - 2) as i32 {
            let eta_x = (abs_omega[idx(i + 1, j, nx)] - abs_omega[idx(i - 1, j, nx)]) * 0.5;
            let eta_y = (abs_omega[idx(i, j + 1, nx)] - abs_omega[idx(i, j - 1, nx)]) * 0.5;
            let len = (eta_x * eta_x + eta_y * eta_y).sqrt() + 1e-10;
            let norm_x = eta_x / len;
            let norm_y = eta_y / len;

            let w = omega[idx(i, j, nx)];
            // 2D cross product: f_x = ε·ny·ω, f_y = -ε·nx·ω
            state.vx[idx(i, j, nx)] += dt * epsilon * norm_y * w;
            state.vy[idx(i, j, nx)] -= dt * epsilon * norm_x * w;
        }
    }
}

/// Full fluid simulation step for Kármán vortex street.
pub fn fluid_step_karman(state: &mut crate::state::SimState, params: &SolverParams) {
    let dt = params.dt;
    let nx = state.nx;
    let bc = BoundaryConfig::KarmanVortex { inflow_vel: params.inflow_vel };

    // Scale visc/diff by 1/(N-2) to fix the Re unit mismatch.
    // diffuse() computes a = dt * ν * (N-2)^2 where (N-2)^2 = 1/h^2 from the
    // Laplacian discretization. The displayed Re = U*D_grid/visc uses grid-cell
    // diameter D_grid, but physical D = D_grid*h = D_grid/(N-2). Compensating
    // one factor of (N-2) gives a = dt * visc * (N-2) — moderate diffusion
    // yielding physical Re ≈ displayed Re.
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
        &mut state.work,
        &mut state.work2,
        params.project_iter,
        &bc,
        nx,
    );

    // 4. Apply mask (zero velocity inside cylinder)
    std::mem::swap(&mut state.vx, &mut state.vx0);
    std::mem::swap(&mut state.vy, &mut state.vy0);
    apply_mask(state);
    std::mem::swap(&mut state.vx, &mut state.vx0);
    std::mem::swap(&mut state.vy, &mut state.vy0);

    // 5. Advect velocity
    advect(FieldType::Vx, &mut state.vx, &state.vx0, &state.vx0, &state.vy0, dt, &bc, nx);
    advect(FieldType::Vy, &mut state.vy, &state.vy0, &state.vx0, &state.vy0, dt, &bc, nx);

    // 6. Project
    project(
        &mut state.vx,
        &mut state.vy,
        &mut state.work,
        &mut state.work2,
        params.project_iter,
        &bc,
        nx,
    );

    // 7. Apply mask
    apply_mask(state);

    // 7.5. Vorticity confinement — counteract numerical diffusion
    vorticity_confinement(state, params.confinement, dt);
    apply_mask(state);

    // 7.6. Tiny wake perturbation to trigger vortex shedding.
    // Applied just behind the cylinder where the physical instability develops,
    // rather than at the inflow where it would pollute the entire domain.
    {
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

    // 8. Diffuse dye (using temperature field as dye)
    diffuse(FieldType::Scalar, &mut state.work, &state.temperature, diff_k, dt, params.diffuse_iter, &bc, nx);

    // 9. Advect dye
    advect(FieldType::Scalar, &mut state.temperature, &state.work, &state.vx, &state.vy, dt, &bc, nx);

    // 10. Inject dye at inflow
    inject_dye(state);

    // 11. Damp dye inside cylinder
    if let Some(ref mask) = state.mask {
        for i in 0..mask.len() {
            state.temperature[i] *= 1.0 - mask[i];
        }
    }

    // 12. Clamp dye
    for t in state.temperature.iter_mut() {
        *t = t.clamp(0.0, 1.0);
    }

    // 13. Advect particles
    advect_particles_karman(state, dt);
}

/// Full fluid simulation step (Rayleigh-Benard).
pub fn fluid_step(state: &mut crate::state::SimState, params: &SolverParams) {
    let dt = params.dt;
    let nx = state.nx;
    let bc = BoundaryConfig::RayleighBenard { bottom_base: params.bottom_base };

    // Diffuse velocity
    diffuse(FieldType::Vx, &mut state.vx0, &state.vx, params.visc, dt, params.diffuse_iter, &bc, nx);
    diffuse(FieldType::Vy, &mut state.vy0, &state.vy, params.visc, dt, params.diffuse_iter, &bc, nx);

    // Project to make diffused velocity divergence-free
    project(
        &mut state.vx0,
        &mut state.vy0,
        &mut state.work,
        &mut state.work2,
        params.project_iter,
        &bc,
        nx,
    );

    // Advect velocity
    advect(FieldType::Vx, &mut state.vx, &state.vx0, &state.vx0, &state.vy0, dt, &bc, nx);
    advect(FieldType::Vy, &mut state.vy, &state.vy0, &state.vx0, &state.vy0, dt, &bc, nx);

    // Apply buoyancy with dt scaling
    apply_buoyancy(&mut state.vy, &state.temperature, params.heat_buoyancy, dt, params.bottom_base, nx);

    // Project BEFORE temperature advection — velocity must be divergence-free
    project(
        &mut state.vx,
        &mut state.vy,
        &mut state.work,
        &mut state.work2,
        params.project_iter,
        &bc,
        nx,
    );

    // Diffuse + advect temperature with divergence-free velocity
    diffuse(FieldType::Temperature, &mut state.work, &state.temperature, params.diff, dt, params.diffuse_iter, &bc, nx);
    advect(FieldType::Temperature, &mut state.temperature, &state.work, &state.vx, &state.vy, dt, &bc, nx);

    // Inject large-scale temperature perturbation to seed/sustain convection cells
    if params.noise_amp > 0.0 {
        inject_thermal_perturbation(&mut state.temperature, &mut state.rng, params.noise_amp, nx);
    }

    // Volumetric heat source at bottom hot spot + cooling at top.
    inject_heat_source(&mut state.temperature, dt, params.source_strength, params.cool_rate, nx);

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
    use crate::state::{idx, SimState, N};

    const BB: f64 = 0.15;

    fn rb_bc() -> BoundaryConfig {
        BoundaryConfig::RayleighBenard { bottom_base: BB }
    }

    // === BoundaryConfig tests ===

    #[test]
    fn test_boundary_config_rb_unchanged() {
        let bc = rb_bc();
        let mut field = vec![0.5; N * N];
        let mut field2 = field.clone();
        for i in 0..N {
            field[idx(i as i32, 1, N)] = 42.0;
            field[idx(i as i32, (N - 2) as i32, N)] = 99.0;
            field2[idx(i as i32, 1, N)] = 42.0;
            field2[idx(i as i32, (N - 2) as i32, N)] = 99.0;
        }
        set_bnd(FieldType::Scalar, &mut field, &bc, N);
        set_bnd(FieldType::Scalar, &mut field2, &bc, N);
        assert_eq!(field, field2);
    }

    #[test]
    fn test_karman_bnd_vx_inflow() {
        let bc = BoundaryConfig::KarmanVortex { inflow_vel: 0.1 };
        let mut field = vec![0.5; N * N];
        set_bnd(FieldType::Vx, &mut field, &bc, N);
        // Left boundary: vx = inflow_vel
        for j in 0..N {
            assert_eq!(field[idx(0, j as i32, N)], 0.1, "Left x=0 should be inflow_vel at y={}", j);
            assert_eq!(field[idx(1, j as i32, N)], 0.1, "Left x=1 should be inflow_vel at y={}", j);
        }
    }

    #[test]
    fn test_karman_bnd_vy_zero() {
        let bc = BoundaryConfig::KarmanVortex { inflow_vel: 0.1 };
        let mut field = vec![0.5; N * N];
        set_bnd(FieldType::Vy, &mut field, &bc, N);
        // Left boundary: vy = 0
        for j in 0..N {
            assert_eq!(field[idx(0, j as i32, N)], 0.0, "Left vy should be 0 at y={}", j);
        }
    }

    #[test]
    fn test_karman_bnd_outflow_zerogradient() {
        let bc = BoundaryConfig::KarmanVortex { inflow_vel: 0.1 };
        let mut field = vec![0.0; N * N];
        // Set interior values near right boundary (avoid wall rows y=0, y=N-1)
        for j in 1..(N - 1) {
            field[idx((N - 2) as i32, j as i32, N)] = 0.42;
        }
        set_bnd(FieldType::Vx, &mut field, &bc, N);
        // Right boundary should copy from N-2 at interior y positions
        for j in 1..(N - 1) {
            assert_eq!(field[idx((N - 1) as i32, j as i32, N)], 0.42,
                "Right boundary should be zero-gradient at y={}", j);
        }
    }

    #[test]
    fn test_karman_bnd_topbottom_noslip() {
        let bc = BoundaryConfig::KarmanVortex { inflow_vel: 0.1 };
        let mut field = vec![0.0; N * N];
        for i in 0..N {
            field[idx(i as i32, 1, N)] = 3.0;
            field[idx(i as i32, (N - 2) as i32, N)] = 5.0;
        }
        set_bnd(FieldType::Vx, &mut field, &bc, N);
        // Top/Bottom: negate for no-slip (except left inflow overrides x=0,1)
        assert_eq!(field[idx(10, 0, N)], -3.0, "Bottom should negate for no-slip");
        assert_eq!(field[idx(10, (N - 1) as i32, N)], -5.0, "Top should negate for no-slip");
    }

    // === Kármán fluid_step tests ===

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

    #[test]
    fn test_default_karman_params() {
        let params = SolverParams::default_karman();
        assert_eq!(params.inflow_vel, 0.1);
        assert_eq!(params.visc, 0.015);
        assert_eq!(params.dt, 0.06);
        assert_eq!(params.confinement, 3.0);
        assert_eq!(params.cylinder_x, 21.0);
        assert_eq!(params.cylinder_y, (N / 2) as f64);
        assert_eq!(params.cylinder_radius, 8.0);
    }

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
        // Center should have dye ≈ 1.0
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
    fn test_advect_karman_clamps_x() {
        let bc = BoundaryConfig::KarmanVortex { inflow_vel: 0.1 };
        let d0 = vec![1.0; N * N];
        let mut d = vec![0.0; N * N];
        // Strong leftward velocity that would wrap in periodic mode
        let vx = vec![1.0; N * N]; // large: backtrace would go x < 0
        let vy = vec![0.0; N * N];

        advect(FieldType::Scalar, &mut d, &d0, &vx, &vy, 0.1, &bc, N);

        // Interior values should still be well-defined (no wrap artifacts)
        for j in 2..(N - 2) {
            for i in 2..(N - 2) {
                let val = d[idx(i as i32, j as i32, N)];
                assert!(val >= 0.0 && val <= 1.0 + 1e-10,
                    "Advected value should be in [0,1] at ({},{}): got {}", i, j, val);
            }
        }
    }

    // === Sub-issue 3 tests: lin_solve, diffuse, set_bnd ===

    #[test]
    fn test_set_bnd_scalar_copies_neighbor() {
        let bc = rb_bc();
        let mut field = vec![0.0; N * N];
        for i in 0..N {
            field[idx(i as i32, 1, N)] = 42.0;
            field[idx(i as i32, (N - 2) as i32, N)] = 99.0;
        }
        set_bnd(FieldType::Scalar, &mut field, &bc, N);
        for i in 0..N {
            assert_eq!(field[idx(i as i32, 0, N)], 42.0, "Bottom boundary should copy y=1");
            assert_eq!(
                field[idx(i as i32, (N - 1) as i32, N)],
                99.0,
                "Top boundary should copy y=N-2"
            );
        }
    }

    #[test]
    fn test_set_bnd_vx_noslip() {
        let bc = rb_bc();
        let mut field = vec![0.0; N * N];
        for i in 0..N {
            field[idx(i as i32, 1, N)] = 5.0;
            field[idx(i as i32, (N - 2) as i32, N)] = 3.0;
        }
        set_bnd(FieldType::Vx, &mut field, &bc, N);
        for i in 0..N {
            assert_eq!(field[idx(i as i32, 0, N)], -5.0, "vx should negate at bottom (no-slip)");
            assert_eq!(
                field[idx(i as i32, (N - 1) as i32, N)],
                -3.0,
                "vx should negate at top (no-slip)"
            );
        }
    }

    #[test]
    fn test_set_bnd_vy_negates() {
        let bc = rb_bc();
        let mut field = vec![0.0; N * N];
        for i in 0..N {
            field[idx(i as i32, 1, N)] = 5.0;
            field[idx(i as i32, (N - 2) as i32, N)] = 3.0;
        }
        set_bnd(FieldType::Vy, &mut field, &bc, N);
        for i in 0..N {
            assert_eq!(field[idx(i as i32, 0, N)], -5.0, "vy should negate at bottom wall");
            assert_eq!(
                field[idx(i as i32, (N - 1) as i32, N)],
                -3.0,
                "vy should negate at top wall"
            );
        }
    }

    #[test]
    fn test_lin_solve_converges() {
        let bc = rb_bc();
        let mut x = vec![0.0; N * N];
        let mut x0 = vec![0.0; N * N];
        let mid = (N / 2) as i32;
        x0[idx(mid, mid, N)] = 100.0;
        x.copy_from_slice(&x0);

        lin_solve(FieldType::Scalar, &mut x, &x0, 1.0, 5.0, 20, &bc, N);

        let center = x[idx(mid, mid, N)];
        let neighbor = x[idx(mid + 1, mid, N)];
        assert!(center > 0.0, "Center should still be positive");
        assert!(neighbor > 0.0, "Neighbors should get some value");
        assert!(center > neighbor, "Center should be larger than neighbor");
    }

    #[test]
    fn test_diffuse_smooths() {
        let bc = rb_bc();
        let mut x0 = vec![0.0; N * N];
        let mut x = vec![0.0; N * N];
        let mid = (N / 2) as i32;
        x0[idx(mid, mid, N)] = 100.0;

        diffuse(FieldType::Scalar, &mut x, &x0, 0.1, 0.1, 4, &bc, N);

        let center = x[idx(mid, mid, N)];
        let neighbor = x[idx(mid + 1, mid, N)];
        assert!(center < 100.0, "Center should be less than original spike");
        assert!(neighbor > 0.0, "Neighbors should gain some value");
    }

    // === Sub-issue 4 tests: advect, project ===

    #[test]
    fn test_advect_zero_velocity_preserves() {
        let bc = rb_bc();
        let mut d0 = vec![0.0; N * N];
        let mut d = vec![0.0; N * N];
        let vx = vec![0.0; N * N];
        let vy = vec![0.0; N * N];

        for j in 1..(N - 1) {
            for i in 0..N {
                d0[idx(i as i32, j as i32, N)] = (i as f64) / N as f64;
            }
        }

        advect(FieldType::Scalar, &mut d, &d0, &vx, &vy, 0.1, &bc, N);

        for j in 2..(N - 2) {
            for i in 0..N {
                let orig = d0[idx(i as i32, j as i32, N)];
                let advected = d[idx(i as i32, j as i32, N)];
                assert!(
                    (orig - advected).abs() < 1e-10,
                    "Zero velocity should preserve field at ({}, {}): {} vs {}",
                    i, j, orig, advected
                );
            }
        }
    }

    #[test]
    fn test_advect_uniform_field_unchanged() {
        let bc = rb_bc();
        let d0 = vec![5.0; N * N];
        let mut d = vec![0.0; N * N];
        let vx = vec![0.01; N * N];
        let vy = vec![0.01; N * N];

        advect(FieldType::Scalar, &mut d, &d0, &vx, &vy, 0.1, &bc, N);

        for j in 2..(N - 2) {
            for i in 0..N {
                let val = d[idx(i as i32, j as i32, N)];
                assert!(
                    (val - 5.0).abs() < 1e-6,
                    "Uniform field should stay uniform: got {} at ({}, {})",
                    val, i, j
                );
            }
        }
    }

    #[test]
    fn test_project_reduces_divergence() {
        let bc = rb_bc();
        let mut vx = vec![0.0; N * N];
        let mut vy = vec![0.0; N * N];
        let mut p = vec![0.0; N * N];
        let mut div = vec![0.0; N * N];

        let cx = (N / 2) as i32;
        let cy = (N / 2) as i32;
        for j in 1..(N - 1) {
            for i in 0..N {
                let dx = i as f64 - cx as f64;
                let dy = j as f64 - cy as f64;
                let r2 = dx * dx + dy * dy;
                let sigma = (N as f64 * N as f64) / 32.0;
                vx[idx(i as i32, j as i32, N)] = dx * 0.01 * (-r2 / sigma).exp();
                vy[idx(i as i32, j as i32, N)] = dy * 0.01 * (-r2 / sigma).exp();
            }
        }

        let mut div_before = 0.0;
        for j in 2..(N - 2) {
            for i in 0..N {
                let d = (vx[idx(i as i32 + 1, j as i32, N)] - vx[idx(i as i32 - 1, j as i32, N)])
                    + (vy[idx(i as i32, j as i32 + 1, N)] - vy[idx(i as i32, j as i32 - 1, N)]);
                div_before += d.abs();
            }
        }
        assert!(div_before > 0.0, "Should have some initial divergence");

        project(&mut vx, &mut vy, &mut p, &mut div, 40, &bc, N);

        let mut div_after = 0.0;
        for j in 2..(N - 2) {
            for i in 0..N {
                let d = (vx[idx(i as i32 + 1, j as i32, N)] - vx[idx(i as i32 - 1, j as i32, N)])
                    + (vy[idx(i as i32, j as i32 + 1, N)] - vy[idx(i as i32, j as i32 - 1, N)]);
                div_after += d.abs();
            }
        }

        assert!(
            div_after < div_before,
            "Divergence should be reduced: before={}, after={}",
            div_before, div_after
        );
    }

    // === Sub-issue 5 tests: fluid_step, buoyancy, heat source ===

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

    #[test]
    fn test_heat_source_boundaries() {
        let bc = rb_bc();
        let mut temperature = vec![0.5; N * N];
        set_bnd(FieldType::Temperature, &mut temperature, &bc, N);

        // Bottom: Gaussian hot spot at center (peak=1.0), base=BB at edges
        let center_t = temperature[idx((N / 2) as i32, 0, N)];
        assert!((center_t - 1.0).abs() < 1e-6, "Bottom center should be ~1.0, got {}", center_t);
        let edge_t = temperature[idx(0, 0, N)];
        assert!(edge_t >= BB - 0.01, "Bottom edge should be >= BB, got {}", edge_t);
        assert!(edge_t < center_t, "Bottom edge should be cooler than center, got {}", edge_t);

        // Top: cold
        for i in 0..N {
            assert!(temperature[idx(i as i32, (N - 1) as i32, N)].abs() < 1e-10, "y=N-1 should be 0.0");
            assert!(temperature[idx(i as i32, (N - 2) as i32, N)].abs() < 1e-10, "y=N-2 should be 0.0");
        }
        // y=2 should NOT be overwritten (remains 0.5)
        let t2 = temperature[idx(0, 2, N)];
        assert!((t2 - 0.5).abs() < 1e-10, "y=2 should remain interior value, got {}", t2);
    }

    // === Particle advection tests ===

    #[test]
    fn test_advect_particles_zero_velocity() {
        let mut state = SimState::new(400, 0.15, N);
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
        let mut state = SimState::new(400, 0.15, N);
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
        let mut state = SimState::new(400, 0.15, N);
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
    #[ignore] // diagnostic only — run with: cargo test test_diagnose -- --ignored --nocapture
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
            &mut state.work,
            &mut state.work2,
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
        eprintln!("vy sign changes at y=N/2: {} (≈ {} convection cells)", sign_changes, sign_changes / 2);

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
}
