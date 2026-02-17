// Playback controller for spmodel-rs .spg output.

use crate::spherical::SphericalSnapshot;
use crate::spgrid::{SpgFrame, SpgReader};

pub struct PlaybackState {
    frames: Vec<SpgFrame>,
    gauss_nodes: Vec<f64>,
    pub im: usize,
    pub jm: usize,
    pub field_names: Vec<String>,
    pub field_index: usize,
    pub current_frame: usize,
    pub playing: bool,
    pub speed: f64,
    accumulator: f64,
    pub model_name: String,
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

        Ok(Self {
            frames,
            gauss_nodes,
            im,
            jm,
            field_names,
            field_index: 0,
            current_frame: 0,
            playing: true,
            speed: 1.0,
            accumulator: 0.0,
            model_name: reader.manifest.model.clone(),
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
            if self.current_frame + 1 < self.frames.len() {
                self.current_frame += 1;
            } else {
                self.current_frame = 0; // loop
            }
        }
    }

    pub fn toggle_play(&mut self) {
        self.playing = !self.playing;
    }

    pub fn step_forward(&mut self) {
        if self.current_frame + 1 < self.frames.len() {
            self.current_frame += 1;
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
        SphericalSnapshot {
            im: self.im,
            jm: self.jm,
            step: frame.step,
            time: frame.time,
            field_name,
            field_data,
            field_names: self.field_names.clone(),
            field_index: self.field_index,
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
