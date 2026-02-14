#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FluidModel {
    RayleighBenard,
    KarmanVortex,
}

impl Default for FluidModel {
    fn default() -> Self {
        Self::RayleighBenard
    }
}

pub const N: usize = 80;
pub const TRAIL_LEN: usize = 8;

pub struct Xor128 {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

pub struct SimState {
    pub nx: usize,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub vx0: Vec<f64>,
    pub vy0: Vec<f64>,
    pub temperature: Vec<f64>,
    /// General-purpose scratch buffer (used for pressure solve / diffused temperature).
    pub scratch_a: Vec<f64>,
    /// General-purpose scratch buffer (used for divergence field).
    pub scratch_b: Vec<f64>,
    /// Scratch buffer for vorticity values (omega = dvy/dx - dvx/dy).
    pub vorticity: Vec<f64>,
    /// Scratch buffer for absolute vorticity (|omega|) used in confinement.
    pub vorticity_abs: Vec<f64>,
    pub rng: Xor128,
    pub particles_x: Vec<f64>,
    pub particles_y: Vec<f64>,
    /// Fractional cylinder mask: 1.0 = fully solid, 0.0 = fully fluid.
    pub mask: Option<Vec<f64>>,
    /// Cylinder geometry (cx, cy, radius) for smooth rendering.
    pub cylinder: Option<(f64, f64, f64)>,
    /// Ring buffer of past particle positions for trail rendering.
    pub trail_xs: Vec<Vec<f64>>,
    pub trail_ys: Vec<Vec<f64>>,
    trail_cursor: usize,
    trail_count: usize,
}

/// Convert 2D coordinates to 1D index with wrapping.
/// x wraps mod nx (horizontal), y wraps mod N (vertical).
pub fn idx(x: i32, y: i32, nx: usize) -> usize {
    let x = ((x % nx as i32) + nx as i32) as usize % nx;
    let y = ((y % N as i32) + N as i32) as usize % N;
    y * nx + x
}

/// Fast index for interior cells where x,y are guaranteed in-bounds.
/// Skips mod wrapping — use only when 0 <= x < nx and 0 <= y < N.
#[inline(always)]
pub const fn idx_inner(x: usize, y: usize, nx: usize) -> usize {
    y * nx + x
}

impl Xor128 {
    pub fn new(seed: u32) -> Self {
        Self {
            x: seed,
            y: seed.wrapping_mul(1812433253).wrapping_add(1),
            z: seed.wrapping_mul(1812433253).wrapping_mul(2).wrapping_add(2),
            w: seed.wrapping_mul(1812433253).wrapping_mul(3).wrapping_add(3),
        }
    }

    pub fn next(&mut self) -> u32 {
        let t = self.x ^ (self.x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        self.w = self.w ^ (self.w >> 19) ^ (t ^ (t >> 8));
        self.w
    }

    /// Returns a float in [-1.0, 1.0)
    pub fn next_f64(&mut self) -> f64 {
        (self.next() as f64 / u32::MAX as f64) * 2.0 - 1.0
    }
}

pub struct FrameSnapshot {
    pub nx: usize,
    pub temperature: Vec<f64>,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub particles_x: Vec<f64>,
    pub particles_y: Vec<f64>,
    /// Cylinder geometry (cx, cy, radius) for smooth rendering.
    pub cylinder: Option<(f64, f64, f64)>,
    /// Past particle positions ordered oldest to newest.
    pub trail_xs: Vec<Vec<f64>>,
    pub trail_ys: Vec<Vec<f64>>,
}

impl FrameSnapshot {
    /// Pre-allocate a snapshot buffer matching the given simulation dimensions.
    pub fn new_empty(nx: usize, n: usize, particle_count: usize) -> Self {
        FrameSnapshot {
            nx,
            temperature: vec![0.0; nx * n],
            vx: vec![0.0; nx * n],
            vy: vec![0.0; nx * n],
            particles_x: vec![0.0; particle_count],
            particles_y: vec![0.0; particle_count],
            cylinder: None,
            trail_xs: vec![Vec::new(); TRAIL_LEN],
            trail_ys: vec![Vec::new(); TRAIL_LEN],
        }
    }
}

impl SimState {
    /// Record current particle positions into the trail ring buffer.
    fn push_trail(&mut self) {
        self.trail_xs[self.trail_cursor] = self.particles_x.clone();
        self.trail_ys[self.trail_cursor] = self.particles_y.clone();
        self.trail_cursor = (self.trail_cursor + 1) % TRAIL_LEN;
        if self.trail_count < TRAIL_LEN {
            self.trail_count += 1;
        }
    }

    /// Allocating snapshot convenience method (used in tests).
    #[cfg(any(test, debug_assertions))]
    pub fn snapshot(&mut self) -> FrameSnapshot {
        self.push_trail();

        // Extract trail in chronological order (oldest first)
        let mut txs = Vec::with_capacity(self.trail_count);
        let mut tys = Vec::with_capacity(self.trail_count);
        for i in 0..self.trail_count {
            let slot = if self.trail_count < TRAIL_LEN {
                i
            } else {
                (self.trail_cursor + i) % TRAIL_LEN
            };
            txs.push(self.trail_xs[slot].clone());
            tys.push(self.trail_ys[slot].clone());
        }

        FrameSnapshot {
            nx: self.nx,
            temperature: self.temperature.clone(),
            vx: self.vx.clone(),
            vy: self.vy.clone(),
            particles_x: self.particles_x.clone(),
            particles_y: self.particles_y.clone(),
            cylinder: self.cylinder,
            trail_xs: txs,
            trail_ys: tys,
        }
    }

    /// Copy current state into a pre-allocated snapshot, avoiding allocation.
    pub fn snapshot_into(&mut self, dst: &mut FrameSnapshot) {
        self.push_trail();

        dst.temperature.copy_from_slice(&self.temperature);
        dst.vx.copy_from_slice(&self.vx);
        dst.vy.copy_from_slice(&self.vy);
        dst.particles_x.copy_from_slice(&self.particles_x);
        dst.particles_y.copy_from_slice(&self.particles_y);

        // Copy trails in chronological order (oldest first).
        // Resize dst.trail_xs/trail_ys to match trail_count (renderer uses .len()).
        let count = self.trail_count;
        dst.trail_xs.resize_with(count, Vec::new);
        dst.trail_ys.resize_with(count, Vec::new);
        for i in 0..count {
            let slot = if self.trail_count < TRAIL_LEN {
                i
            } else {
                (self.trail_cursor + i) % TRAIL_LEN
            };
            let src_x = &self.trail_xs[slot];
            let src_y = &self.trail_ys[slot];
            dst.trail_xs[i].resize(src_x.len(), 0.0);
            dst.trail_ys[i].resize(src_y.len(), 0.0);
            dst.trail_xs[i].copy_from_slice(src_x);
            dst.trail_ys[i].copy_from_slice(src_y);
        }

        dst.nx = self.nx;
        dst.cylinder = self.cylinder;
    }

    pub fn new(num_particles: usize, bottom_base: f64, nx: usize) -> Self {
        let size = nx * N;
        let mut rng = Xor128::new(42);

        let mut temperature = vec![0.0; size];
        let mut vx = vec![0.0; size];
        let mut vy = vec![0.0; size];

        // Initial temperature: Gaussian hot spot at bottom center, cold top.
        // Bottom BC has a localized heat source (see solver::set_bnd field_type 3).
        let sigma = (N / 24) as f64;
        let center = (nx / 2) as f64;
        for y in 0..N {
            let y_frac = y as f64 / (N - 1) as f64;
            for x in 0..nx {
                let dx = x as f64 - center;
                let hot = bottom_base + (1.0 - bottom_base) * (-dx * dx / (2.0 * sigma * sigma)).exp();
                let t_base = hot * (1.0 - y_frac); // gradient from hot-spot profile to 0
                let noise = rng.next_f64() * 0.02;
                temperature[idx(x as i32, y as i32, nx)] = (t_base + noise).clamp(0.0, 1.0);
            }
        }

        // Small perturbation to velocity to break symmetry
        let perturbation = 1e-5;
        for y in 0..N {
            for x in 0..nx {
                let i = idx(x as i32, y as i32, nx);
                vx[i] = perturbation * rng.next_f64();
                vy[i] = perturbation * rng.next_f64();
            }
        }

        // Initialize particles at random positions in the active interior
        // (outside the 2-row Dirichlet boundary zone where velocity ≈ 0)
        let mut particles_x = Vec::with_capacity(num_particles);
        let mut particles_y = Vec::with_capacity(num_particles);
        for _ in 0..num_particles {
            let px = 2.0 + (rng.next_f64() + 1.0) * 0.5 * (nx as f64 - 5.0);
            let py = 2.0 + (rng.next_f64() + 1.0) * 0.5 * (N as f64 - 5.0);
            particles_x.push(px);
            particles_y.push(py);
        }

        // Seed convection cells with large-scale thermal perturbation (initial only).
        // Convection is self-sustaining after BC-driven heat source takes over.
        crate::solver::inject_thermal_perturbation(&mut temperature, &mut rng, 0.05, nx);

        Self {
            nx,
            vx,
            vy,
            vx0: vec![0.0; size],
            vy0: vec![0.0; size],
            temperature,
            scratch_a: vec![0.0; size],
            scratch_b: vec![0.0; size],
            vorticity: vec![0.0; size],
            vorticity_abs: vec![0.0; size],
            rng,
            particles_x,
            particles_y,
            mask: None,
            cylinder: None,
            trail_xs: vec![Vec::new(); TRAIL_LEN],
            trail_ys: vec![Vec::new(); TRAIL_LEN],
            trail_cursor: 0,
            trail_count: 0,
        }
    }

    pub fn new_karman(num_particles: usize, inflow_vel: f64, cx: f64, cy: f64, radius: f64, nx: usize) -> Self {
        let size = nx * N;
        let mut rng = Xor128::new(42);
        let mask = build_cylinder_mask(cx, cy, radius, nx);

        let mut vx = vec![inflow_vel; size];
        let mut vy = vec![0.0; size];
        let temperature = vec![0.0; size];

        // Small random vy perturbation to trigger vortex shedding
        for j in 0..N {
            for i in 0..nx {
                let ii = idx(i as i32, j as i32, nx);
                vy[ii] = 1e-4 * rng.next_f64();
                // Damp velocity inside/near cylinder
                let solid = mask[ii];
                vx[ii] *= 1.0 - solid;
                vy[ii] *= 1.0 - solid;
            }
        }

        // Particles: random positions in left quarter, avoiding cylinder
        let mut particles_x = Vec::with_capacity(num_particles);
        let mut particles_y = Vec::with_capacity(num_particles);
        let nx_f = nx as f64;
        let n_f = N as f64;
        for _ in 0..num_particles {
            loop {
                let px = 2.0 + (rng.next_f64() + 1.0) * 0.5 * (nx_f * 0.25);
                let py = 2.0 + (rng.next_f64() + 1.0) * 0.5 * (n_f - 5.0);
                let gi = (px as usize).min(nx - 1);
                let gj = (py as usize).min(N - 1);
                if mask[gj * nx + gi] < 0.5 {
                    particles_x.push(px);
                    particles_y.push(py);
                    break;
                }
            }
        }

        Self {
            nx,
            vx,
            vy,
            vx0: vec![0.0; size],
            vy0: vec![0.0; size],
            temperature,
            scratch_a: vec![0.0; size],
            scratch_b: vec![0.0; size],
            vorticity: vec![0.0; size],
            vorticity_abs: vec![0.0; size],
            rng,
            particles_x,
            particles_y,
            mask: Some(mask),
            cylinder: Some((cx, cy, radius)),
            trail_xs: vec![Vec::new(); TRAIL_LEN],
            trail_ys: vec![Vec::new(); TRAIL_LEN],
            trail_cursor: 0,
            trail_count: 0,
        }
    }
}

/// Build a fractional mask for a circular cylinder obstacle.
/// 1.0 = fully solid, 0.0 = fully fluid, with a 1-cell-wide smooth transition.
pub fn build_cylinder_mask(cx: f64, cy: f64, radius: f64, nx: usize) -> Vec<f64> {
    let mut mask = vec![0.0; nx * N];
    for j in 0..N {
        for i in 0..nx {
            let dx = i as f64 - cx;
            let dy = j as f64 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            // Smooth transition over 1 cell width
            mask[j * nx + i] = (radius + 0.5 - dist).clamp(0.0, 1.0);
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_size() {
        assert_eq!(N, 80);
    }

    #[test]
    fn test_idx_basic() {
        assert_eq!(idx(0, 0, N), 0);
        assert_eq!(idx(1, 0, N), 1);
        assert_eq!(idx(0, 1, N), N);
        assert_eq!(idx((N - 1) as i32, (N - 1) as i32, N), N * N - 1);
    }

    #[test]
    fn test_idx_wrap_positive() {
        assert_eq!(idx(N as i32, 0, N), idx(0, 0, N));
        assert_eq!(idx(0, N as i32, N), idx(0, 0, N));
        assert_eq!(idx(N as i32 + 3, N as i32 + 5, N), idx(3, 5, N));
    }

    #[test]
    fn test_idx_wrap_negative() {
        assert_eq!(idx(-1, 0, N), idx(N as i32 - 1, 0, N));
        assert_eq!(idx(0, -1, N), idx(0, N as i32 - 1, N));
        assert_eq!(idx(-3, -5, N), idx(N as i32 - 3, N as i32 - 5, N));
    }

    #[test]
    fn test_idx_nonsquare() {
        let nx = 256;
        assert_eq!(idx(0, 0, nx), 0);
        assert_eq!(idx(1, 0, nx), 1);
        assert_eq!(idx(0, 1, nx), nx);
        assert_eq!(idx((nx - 1) as i32, (N - 1) as i32, nx), N * nx - 1);
        // X wraps mod nx
        assert_eq!(idx(nx as i32, 0, nx), idx(0, 0, nx));
        // Y wraps mod N
        assert_eq!(idx(0, N as i32, nx), idx(0, 0, nx));
    }

    #[test]
    fn test_initial_temperature_bottom_hot() {
        let state = SimState::new(400, 0.15, N);
        // Bottom center (hot spot) should be near 1.0
        let center_t = state.temperature[idx((N / 2) as i32, 0, N)];
        assert!(center_t > 0.9, "Bottom center should be hot (near 1.0), got {}", center_t);
        // Bottom edges should be at base temperature
        let edge_t = state.temperature[idx(0, 0, N)];
        assert!(edge_t < center_t, "Bottom edge should be cooler than center, got {}", edge_t);
    }

    #[test]
    fn test_initial_temperature_top_cold() {
        let state = SimState::new(400, 0.15, N);
        for x in 0..N {
            let t = state.temperature[idx(x as i32, (N - 1) as i32, N)];
            assert!(t < 0.1, "Top should be cold (near 0.0), got {}", t);
        }
    }

    #[test]
    fn test_initial_temperature_gradient() {
        let state = SimState::new(400, 0.15, N);
        // Average temperature should decrease from bottom to top
        let avg_bottom: f64 = (0..N).map(|x| state.temperature[idx(x as i32, 0, N)]).sum::<f64>() / N as f64;
        let avg_mid: f64 = (0..N).map(|x| state.temperature[idx(x as i32, (N / 2) as i32, N)]).sum::<f64>() / N as f64;
        let avg_top: f64 = (0..N).map(|x| state.temperature[idx(x as i32, (N - 1) as i32, N)]).sum::<f64>() / N as f64;
        assert!(avg_bottom > avg_mid, "Bottom should be hotter than middle");
        assert!(avg_mid > avg_top, "Middle should be hotter than top");
    }

    #[test]
    fn test_initial_velocity_near_zero() {
        let state = SimState::new(400, 0.15, N);
        let max_v: f64 = state.vx.iter().chain(state.vy.iter())
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(max_v < 1e-3, "Initial velocity should be near zero, max was {}", max_v);
        // But not exactly zero (perturbation applied)
        assert!(max_v > 0.0, "Velocity should have some perturbation");
    }

    #[test]
    fn test_all_fields_correct_size() {
        let state = SimState::new(400, 0.15, N);
        assert_eq!(state.vx.len(), N * N);
        assert_eq!(state.vy.len(), N * N);
        assert_eq!(state.vx0.len(), N * N);
        assert_eq!(state.vy0.len(), N * N);
        assert_eq!(state.temperature.len(), N * N);
        assert_eq!(state.scratch_a.len(), N * N);
        assert_eq!(state.scratch_b.len(), N * N);
        assert_eq!(state.vorticity.len(), N * N);
        assert_eq!(state.vorticity_abs.len(), N * N);
    }

    #[test]
    fn test_xor128_deterministic() {
        let mut rng1 = Xor128::new(42);
        let mut rng2 = Xor128::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_xor128_range() {
        let mut rng = Xor128::new(42);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= -1.0 && v < 1.0, "next_f64 out of range: {}", v);
        }
    }

    #[test]
    fn test_fluid_model_default_is_rb() {
        assert_eq!(FluidModel::default(), FluidModel::RayleighBenard);
    }

    #[test]
    fn test_particle_count() {
        let state = SimState::new(400, 0.15, N);
        assert_eq!(state.particles_x.len(), 400);
        assert_eq!(state.particles_y.len(), 400);
    }

    #[test]
    fn test_simstate_new_has_no_mask() {
        let state = SimState::new(10, 0.15, N);
        assert!(state.mask.is_none());
    }

    #[test]
    fn test_build_cylinder_mask_basic() {
        let mask = build_cylinder_mask(21.0, (N / 2) as f64, 8.0, N);
        assert_eq!(mask.len(), N * N);
        // Center should be fully solid
        assert_eq!(mask[(N / 2) * N + 21], 1.0, "Center should be fully solid");
        // Far corner should be fully fluid
        assert_eq!(mask[0], 0.0, "Origin should be fully fluid");
    }

    #[test]
    fn test_build_cylinder_mask_radius() {
        let mask = build_cylinder_mask((N / 2) as f64, (N / 2) as f64, 20.0, N);
        // Count fully solid cells (mask == 1.0)
        let count: usize = mask.iter().filter(|&&v| v == 1.0).count();
        // Slightly fewer than π r² ≈ 1257 due to smooth boundary
        assert!(count > 1100 && count < 1300, "Fully solid cell count {} not near π*400", count);
    }

    #[test]
    fn test_build_cylinder_mask_smooth_edge() {
        let mask = build_cylinder_mask((N / 2) as f64, (N / 2) as f64, 20.0, N);
        // There should be cells with fractional values (0 < mask < 1)
        let edge_count = mask.iter().filter(|&&v| v > 0.0 && v < 1.0).count();
        assert!(edge_count > 0, "Should have smooth edge cells, got {}", edge_count);
    }

    #[test]
    fn test_new_karman_initial_conditions() {
        let state = SimState::new_karman(100, 0.1, 13.0, (N / 2) as f64, 5.0, N);
        assert!(state.mask.is_some());
        assert_eq!(state.particles_x.len(), 100);
        // Temperature (dye) should be 0 everywhere
        assert!(state.temperature.iter().all(|&t| t == 0.0));
    }

    #[test]
    fn test_new_karman_velocity_inside_cylinder_zero() {
        let state = SimState::new_karman(10, 0.1, 21.0, (N / 2) as f64, 8.0, N);
        let mask = state.mask.as_ref().unwrap();
        for j in 0..N {
            for i in 0..N {
                let ii = idx(i as i32, j as i32, N);
                if mask[ii] == 1.0 {
                    assert_eq!(state.vx[ii], 0.0, "vx inside cylinder should be 0 at ({},{})", i, j);
                    assert_eq!(state.vy[ii], 0.0, "vy inside cylinder should be 0 at ({},{})", i, j);
                }
            }
        }
    }

    #[test]
    fn test_particles_in_domain() {
        let state = SimState::new(400, 0.15, N);
        for i in 0..400 {
            let px = state.particles_x[i];
            let py = state.particles_y[i];
            assert!(px >= 0.0 && px < N as f64, "particle x out of range: {}", px);
            assert!(py >= 2.0 && py <= (N - 3) as f64, "particle y out of range: {}", py);
        }
    }

    #[test]
    fn test_trail_initial_empty() {
        let state = SimState::new(10, 0.15, N);
        assert_eq!(state.trail_count, 0);
        assert_eq!(state.trail_cursor, 0);
        assert_eq!(state.trail_xs.len(), TRAIL_LEN);
    }

    #[test]
    fn test_trail_push_increments() {
        let mut state = SimState::new(10, 0.15, N);
        let snap = state.snapshot();
        assert_eq!(snap.trail_xs.len(), 1);
        assert_eq!(snap.trail_ys.len(), 1);
        assert_eq!(snap.trail_xs[0].len(), 10);
    }

    #[test]
    fn test_trail_ring_buffer_wraps() {
        let mut state = SimState::new(2, 0.15, N);
        // Fill the ring buffer beyond capacity
        for _ in 0..TRAIL_LEN + 3 {
            let _ = state.snapshot();
        }
        // trail_count should be capped at TRAIL_LEN
        let snap = state.snapshot();
        assert_eq!(snap.trail_xs.len(), TRAIL_LEN);
    }

    #[test]
    fn test_trail_chronological_order() {
        let mut state = SimState::new(1, 0.15, N);
        // Take snapshots while moving the particle
        for i in 0..4 {
            state.particles_x[0] = i as f64 * 10.0;
            let _ = state.snapshot();
        }
        // Next snapshot should have trail in order: 0, 10, 20, 30, current
        state.particles_x[0] = 40.0;
        let snap = state.snapshot();
        for i in 1..snap.trail_xs.len() {
            assert!(
                snap.trail_xs[i][0] >= snap.trail_xs[i - 1][0],
                "Trail should be oldest-first: idx {} ({}) < idx {} ({})",
                i, snap.trail_xs[i][0], i - 1, snap.trail_xs[i - 1][0]
            );
        }
    }

    #[test]
    fn test_simstate_karman_nonsquare() {
        let nx = 256;
        let state = SimState::new_karman(100, 0.1, 21.0, (N / 2) as f64, 8.0, nx);
        assert_eq!(state.nx, nx);
        assert_eq!(state.vx.len(), nx * N);
        assert_eq!(state.vy.len(), nx * N);
        assert_eq!(state.temperature.len(), nx * N);
        assert_eq!(state.mask.as_ref().unwrap().len(), nx * N);
    }

    #[test]
    fn test_new_empty_dimensions() {
        let snap = FrameSnapshot::new_empty(N, N, 100);
        assert_eq!(snap.nx, N);
        assert_eq!(snap.temperature.len(), N * N);
        assert_eq!(snap.vx.len(), N * N);
        assert_eq!(snap.vy.len(), N * N);
        assert_eq!(snap.particles_x.len(), 100);
        assert_eq!(snap.particles_y.len(), 100);
        assert_eq!(snap.trail_xs.len(), TRAIL_LEN);
        assert!(snap.cylinder.is_none());
    }

    #[test]
    fn test_snapshot_into_matches_snapshot() {
        let mut state = SimState::new(10, 0.15, N);
        // Run a few snapshots to populate trails
        for _ in 0..3 {
            let _ = state.snapshot();
        }

        // Take a snapshot via both methods from the same state
        let snap_clone = state.snapshot();
        // Reset trail state to match (snapshot() calls push_trail, so we need a fresh state)
        let mut state2 = SimState::new(10, 0.15, N);
        for _ in 0..3 {
            let _ = state2.snapshot();
        }
        let mut dst = FrameSnapshot::new_empty(N, N, 10);
        state2.snapshot_into(&mut dst);

        assert_eq!(dst.nx, snap_clone.nx);
        assert_eq!(dst.temperature, snap_clone.temperature);
        assert_eq!(dst.vx, snap_clone.vx);
        assert_eq!(dst.vy, snap_clone.vy);
        assert_eq!(dst.particles_x, snap_clone.particles_x);
        assert_eq!(dst.particles_y, snap_clone.particles_y);
        assert_eq!(dst.cylinder, snap_clone.cylinder);
        assert_eq!(dst.trail_xs.len(), snap_clone.trail_xs.len());
        assert_eq!(dst.trail_ys.len(), snap_clone.trail_ys.len());
    }

    #[test]
    fn test_snapshot_into_reuse_buffer() {
        let mut state = SimState::new(10, 0.15, N);
        let mut dst = FrameSnapshot::new_empty(N, N, 10);

        // Call snapshot_into multiple times to verify buffer reuse
        for _ in 0..TRAIL_LEN + 3 {
            state.snapshot_into(&mut dst);
        }
        // After TRAIL_LEN+3 calls, trail_xs should have TRAIL_LEN entries
        assert_eq!(dst.trail_xs.len(), TRAIL_LEN);
    }
}
