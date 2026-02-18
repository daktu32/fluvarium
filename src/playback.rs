// Playback controller for spmodel-rs .spg output.

use crate::spherical::{SphericalParticleSystem, SphericalSnapshot, PARTICLE_COUNT};
use crate::spgrid::{SpgFrame, SpgReader};

pub struct PlaybackState {
    frames: Vec<SpgFrame>,
    gauss_nodes: Vec<f64>,
    /// Global (vmin, vmax) per field index, computed across all frames.
    global_ranges: Vec<(f64, f64)>,
    pub im: usize,
    pub jm: usize,
    pub field_names: Vec<String>,
    pub field_index: usize,
    pub current_frame: usize,
    pub playing: bool,
    pub speed: f64,
    accumulator: f64,
    pub model_name: String,
    /// Index of "u_cos" field in the frame fields list.
    u_cos_index: Option<usize>,
    /// Index of "v_cos" field in the frame fields list.
    v_cos_index: Option<usize>,
    /// Particle system (only if velocity fields are present).
    pub particles: Option<SphericalParticleSystem>,
    /// Model dt (time between consecutive steps in simulation units).
    model_dt: f64,
    /// Output interval in steps.
    output_interval: u64,
}

impl PlaybackState {
    pub fn from_reader(reader: &SpgReader) -> std::io::Result<Self> {
        let frames = reader.read_all_frames()?;
        let im = reader.manifest.grid.im;
        let jm = reader.manifest.grid.jm;
        let field_names = reader.manifest.fields.clone();

        // Compute gauss nodes from nm
        let nm = reader.manifest.grid.nm;
        let gauss_nodes = compute_gauss_nodes(nm, jm);

        // Pre-compute global min/max per field across all frames
        let nfields = field_names.len();
        let mut global_ranges = vec![(f64::INFINITY, f64::NEG_INFINITY); nfields];
        for frame in &frames {
            for (fi, (_name, data)) in frame.fields.iter().enumerate() {
                if fi < nfields {
                    for &v in data {
                        if v < global_ranges[fi].0 {
                            global_ranges[fi].0 = v;
                        }
                        if v > global_ranges[fi].1 {
                            global_ranges[fi].1 = v;
                        }
                    }
                }
            }
        }
        // Apply diverging symmetric centering
        for r in &mut global_ranges {
            if r.0 < 0.0 && r.1 > 0.0 {
                let abs_max = r.0.abs().max(r.1.abs());
                r.0 = -abs_max;
                r.1 = abs_max;
            }
        }

        // Pad range for fields with small variation relative to their mean.
        // e.g. geopotential phi ≈ 100 ± 0.08 → without padding the tiny
        // fluctuation fills the full color spectrum.  Expanding the range
        // to at least 1% of |mean| keeps the variation visually subtle.
        for r in &mut global_ranges {
            let mean = (r.0 + r.1) * 0.5;
            let current_range = r.1 - r.0;
            let min_range = mean.abs() * 0.01;
            if current_range < min_range && mean.abs() > 1e-10 {
                r.0 = mean - min_range * 0.5;
                r.1 = mean + min_range * 0.5;
            }
        }

        // Detect velocity fields
        let u_cos_index = field_names.iter().position(|n| n == "u_cos");
        let v_cos_index = field_names.iter().position(|n| n == "v_cos");
        let particles = if u_cos_index.is_some() && v_cos_index.is_some() {
            Some(SphericalParticleSystem::new(PARTICLE_COUNT))
        } else {
            None
        };

        let model_dt = reader.manifest.time.dt;
        let output_interval = reader.manifest.time.output_interval;

        Ok(Self {
            frames,
            gauss_nodes,
            global_ranges,
            im,
            jm,
            field_names,
            field_index: 0,
            current_frame: 0,
            playing: true,
            speed: 1.0,
            accumulator: 0.0,
            model_name: reader.manifest.model.clone(),
            u_cos_index,
            v_cos_index,
            particles,
            model_dt,
            output_interval,
        })
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn gauss_nodes(&self) -> &[f64] {
        &self.gauss_nodes
    }

    pub fn tick(&mut self, dt_seconds: f64) {
        if !self.playing || self.frames.is_empty() {
            return;
        }
        self.accumulator += dt_seconds * self.speed * 30.0; // 30 frames/sec base rate
        while self.accumulator >= 1.0 {
            self.accumulator -= 1.0;
            let prev_frame = self.current_frame;
            if self.current_frame + 1 < self.frames.len() {
                self.current_frame += 1;
            } else {
                self.current_frame = 0; // loop
            }
            if self.current_frame != prev_frame {
                self.advect_particles();
            }
        }
    }

    fn advect_particles(&mut self) {
        let (Some(ui), Some(vi)) = (self.u_cos_index, self.v_cos_index) else {
            return;
        };
        let Some(ref mut ps) = self.particles else {
            return;
        };

        let frame = &self.frames[self.current_frame];
        if ui >= frame.fields.len() || vi >= frame.fields.len() {
            return;
        }

        let cu_data = &frame.fields[ui].1;
        let cv_data = &frame.fields[vi].1;
        let sim_dt = self.model_dt * self.output_interval as f64;

        ps.advect(cu_data, cv_data, self.im, self.jm, &self.gauss_nodes, sim_dt);
    }

    pub fn toggle_play(&mut self) {
        self.playing = !self.playing;
    }

    pub fn toggle_particles(&mut self) {
        if let Some(ref mut ps) = self.particles {
            ps.enabled = !ps.enabled;
        }
    }

    pub fn has_particles(&self) -> bool {
        self.particles.is_some()
    }

    pub fn particles_enabled(&self) -> bool {
        self.particles.as_ref().map_or(false, |ps| ps.enabled)
    }

    pub fn step_forward(&mut self) {
        if self.current_frame + 1 < self.frames.len() {
            self.current_frame += 1;
            self.advect_particles();
        }
    }

    pub fn step_backward(&mut self) {
        if self.current_frame > 0 {
            self.current_frame -= 1;
        }
    }

    pub fn speed_up(&mut self) {
        self.speed = (self.speed * 2.0).min(64.0);
    }

    pub fn speed_down(&mut self) {
        self.speed = (self.speed * 0.5).max(0.0625);
    }

    pub fn next_field(&mut self) {
        if !self.field_names.is_empty() {
            self.field_index = (self.field_index + 1) % self.field_names.len();
        }
    }

    pub fn snapshot(&self) -> SphericalSnapshot {
        let frame = &self.frames[self.current_frame];
        let field_data = if self.field_index < frame.fields.len() {
            frame.fields[self.field_index].1.clone()
        } else {
            vec![0.0; self.im * self.jm]
        };
        let field_name = if self.field_index < frame.fields.len() {
            frame.fields[self.field_index].0.clone()
        } else {
            "?".to_string()
        };
        let global_range = self.global_ranges.get(self.field_index).copied();
        SphericalSnapshot {
            im: self.im,
            jm: self.jm,
            step: frame.step,
            time: frame.time,
            field_name,
            field_data,
            field_names: self.field_names.clone(),
            field_index: self.field_index,
            global_range,
        }
    }
}

/// Compute gauss nodes (μ = sinφ, ascending) via Newton iteration on Legendre polynomials.
fn compute_gauss_nodes(nm: usize, jm: usize) -> Vec<f64> {
    let mut nodes = Vec::with_capacity(jm);
    let n = jm;

    for i in 0..n {
        // Initial guess (Chebyshev-like)
        let mut x = ((4 * i + 3) as f64 * std::f64::consts::PI / (4 * n + 2) as f64).cos();

        // Newton iteration on P_n(x)
        for _ in 0..50 {
            let (pn, dpn) = legendre_pn(n, x);
            let dx = pn / dpn;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        nodes.push(x);
    }

    // Sort ascending (south to north)
    nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Verify these are indeed gauss nodes for Legendre polynomials
    // For spectral transforms, we need the jm nodes for the jm Gauss quadrature points
    let _ = nm; // nm used indirectly (jm is already the number of gauss points)

    nodes
}

/// Evaluate P_n(x) and P'_n(x) via recurrence.
fn legendre_pn(n: usize, x: f64) -> (f64, f64) {
    let mut p0 = 1.0;
    let mut p1 = x;
    for k in 2..=n {
        let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
        p0 = p1;
        p1 = p2;
    }
    let dp = n as f64 * (p0 - x * p1) / (1.0 - x * x);
    (p1, dp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_nodes_count() {
        let nodes = compute_gauss_nodes(42, 64);
        assert_eq!(nodes.len(), 64);
    }

    #[test]
    fn test_gauss_nodes_ascending() {
        let nodes = compute_gauss_nodes(42, 64);
        for i in 1..nodes.len() {
            assert!(nodes[i] > nodes[i - 1], "nodes should be ascending");
        }
    }

    #[test]
    fn test_gauss_nodes_in_range() {
        let nodes = compute_gauss_nodes(42, 64);
        for &mu in &nodes {
            assert!(mu > -1.0 && mu < 1.0, "mu should be in (-1, 1), got {mu}");
        }
    }

    #[test]
    fn test_gauss_nodes_symmetric() {
        let nodes = compute_gauss_nodes(42, 64);
        let n = nodes.len();
        for i in 0..n / 2 {
            let sum = nodes[i] + nodes[n - 1 - i];
            assert!(sum.abs() < 1e-12, "nodes should be symmetric, got sum={sum}");
        }
    }
}
