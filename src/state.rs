pub const N: usize = 128;
pub const SIZE: usize = N * N;
pub struct Xor128 {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

pub struct SimState {
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub vx0: Vec<f64>,
    pub vy0: Vec<f64>,
    pub temperature: Vec<f64>,
    pub work: Vec<f64>,
    pub work2: Vec<f64>,
    pub rng: Xor128,
    pub particles_x: Vec<f64>,
    pub particles_y: Vec<f64>,
}

/// Convert 2D coordinates to 1D index with wrapping (mod N).
pub fn idx(x: i32, y: i32) -> usize {
    let x = ((x % N as i32) + N as i32) as usize % N;
    let y = ((y % N as i32) + N as i32) as usize % N;
    y * N + x
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
    pub temperature: Vec<f64>,
    pub particles_x: Vec<f64>,
    pub particles_y: Vec<f64>,
}

impl SimState {
    pub fn snapshot(&self) -> FrameSnapshot {
        FrameSnapshot {
            temperature: self.temperature.clone(),
            particles_x: self.particles_x.clone(),
            particles_y: self.particles_y.clone(),
        }
    }

    pub fn new(num_particles: usize, bottom_base: f64) -> Self {
        let mut rng = Xor128::new(42);

        let mut temperature = vec![0.0; SIZE];
        let mut vx = vec![0.0; SIZE];
        let mut vy = vec![0.0; SIZE];

        // Initial temperature: Gaussian hot spot at bottom center, cold top.
        // Bottom BC has a localized heat source (see solver::set_bnd field_type 3).
        let sigma = (N / 24) as f64;
        let center = (N / 2) as f64;
        for y in 0..N {
            let y_frac = y as f64 / (N - 1) as f64;
            for x in 0..N {
                let dx = x as f64 - center;
                let hot = bottom_base + (1.0 - bottom_base) * (-dx * dx / (2.0 * sigma * sigma)).exp();
                let t_base = hot * (1.0 - y_frac); // gradient from hot-spot profile to 0
                let noise = rng.next_f64() * 0.02;
                temperature[idx(x as i32, y as i32)] = (t_base + noise).clamp(0.0, 1.0);
            }
        }

        // Small perturbation to velocity to break symmetry
        let perturbation = 1e-5;
        for y in 0..N {
            for x in 0..N {
                let i = idx(x as i32, y as i32);
                vx[i] = perturbation * rng.next_f64();
                vy[i] = perturbation * rng.next_f64();
            }
        }

        // Initialize particles at random positions in the active interior
        // (outside the 2-row Dirichlet boundary zone where velocity ≈ 0)
        let mut particles_x = Vec::with_capacity(num_particles);
        let mut particles_y = Vec::with_capacity(num_particles);
        for _ in 0..num_particles {
            let px = 2.0 + (rng.next_f64() + 1.0) * 0.5 * (N as f64 - 5.0);
            let py = 2.0 + (rng.next_f64() + 1.0) * 0.5 * (N as f64 - 5.0);
            particles_x.push(px);
            particles_y.push(py);
        }

        Self {
            vx,
            vy,
            vx0: vec![0.0; SIZE],
            vy0: vec![0.0; SIZE],
            temperature,
            work: vec![0.0; SIZE],
            work2: vec![0.0; SIZE],
            rng,
            particles_x,
            particles_y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_size() {
        assert_eq!(N, 128);
        assert_eq!(SIZE, 128 * 128);
    }

    #[test]
    fn test_idx_basic() {
        assert_eq!(idx(0, 0), 0);
        assert_eq!(idx(1, 0), 1);
        assert_eq!(idx(0, 1), N);
        assert_eq!(idx((N - 1) as i32, (N - 1) as i32), SIZE - 1);
    }

    #[test]
    fn test_idx_wrap_positive() {
        assert_eq!(idx(N as i32, 0), idx(0, 0));
        assert_eq!(idx(0, N as i32), idx(0, 0));
        assert_eq!(idx(N as i32 + 3, N as i32 + 5), idx(3, 5));
    }

    #[test]
    fn test_idx_wrap_negative() {
        assert_eq!(idx(-1, 0), idx(N as i32 - 1, 0));
        assert_eq!(idx(0, -1), idx(0, N as i32 - 1));
        assert_eq!(idx(-3, -5), idx(N as i32 - 3, N as i32 - 5));
    }

    #[test]
    fn test_initial_temperature_bottom_hot() {
        let state = SimState::new(400, 0.15);
        // Bottom center (hot spot) should be near 1.0
        let center_t = state.temperature[idx((N / 2) as i32, 0)];
        assert!(center_t > 0.9, "Bottom center should be hot (near 1.0), got {}", center_t);
        // Bottom edges should be at base temperature
        let edge_t = state.temperature[idx(0, 0)];
        assert!(edge_t < center_t, "Bottom edge should be cooler than center, got {}", edge_t);
    }

    #[test]
    fn test_initial_temperature_top_cold() {
        let state = SimState::new(400, 0.15);
        for x in 0..N {
            let t = state.temperature[idx(x as i32, (N - 1) as i32)];
            assert!(t < 0.1, "Top should be cold (near 0.0), got {}", t);
        }
    }

    #[test]
    fn test_initial_temperature_gradient() {
        let state = SimState::new(400, 0.15);
        // Average temperature should decrease from bottom to top
        let avg_bottom: f64 = (0..N).map(|x| state.temperature[idx(x as i32, 0)]).sum::<f64>() / N as f64;
        let avg_mid: f64 = (0..N).map(|x| state.temperature[idx(x as i32, (N / 2) as i32)]).sum::<f64>() / N as f64;
        let avg_top: f64 = (0..N).map(|x| state.temperature[idx(x as i32, (N - 1) as i32)]).sum::<f64>() / N as f64;
        assert!(avg_bottom > avg_mid, "Bottom should be hotter than middle");
        assert!(avg_mid > avg_top, "Middle should be hotter than top");
    }

    #[test]
    fn test_initial_velocity_near_zero() {
        let state = SimState::new(400, 0.15);
        let max_v: f64 = state.vx.iter().chain(state.vy.iter())
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(max_v < 1e-3, "Initial velocity should be near zero, max was {}", max_v);
        // But not exactly zero (perturbation applied)
        assert!(max_v > 0.0, "Velocity should have some perturbation");
    }

    #[test]
    fn test_all_fields_correct_size() {
        let state = SimState::new(400, 0.15);
        assert_eq!(state.vx.len(), SIZE);
        assert_eq!(state.vy.len(), SIZE);
        assert_eq!(state.vx0.len(), SIZE);
        assert_eq!(state.vy0.len(), SIZE);
        assert_eq!(state.temperature.len(), SIZE);
        assert_eq!(state.work.len(), SIZE);
        assert_eq!(state.work2.len(), SIZE);
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
    fn test_particle_count() {
        let state = SimState::new(400, 0.15);
        assert_eq!(state.particles_x.len(), 400);
        assert_eq!(state.particles_y.len(), 400);
    }

    #[test]
    fn test_particles_in_domain() {
        let state = SimState::new(400, 0.15);
        for i in 0..400 {
            let px = state.particles_x[i];
            let py = state.particles_y[i];
            assert!(px >= 0.0 && px < N as f64, "particle x out of range: {}", px);
            assert!(py >= 2.0 && py <= (N - 3) as f64, "particle y out of range: {}", py);
        }
    }

}
