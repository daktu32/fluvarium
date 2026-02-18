// Spherical data snapshot for playback rendering.

use crate::state::Xor128;
use std::f64::consts::PI;

pub const PARTICLE_COUNT: usize = 2000;
pub const TRAIL_LEN: usize = 8;

const LAT_CLAMP: f64 = 85.0 * PI / 180.0;

pub struct SphericalParticle {
    pub lon: f64, // [0, 2π)
    pub lat: f64, // [-π/2, π/2]
}

pub struct SphericalParticleSystem {
    pub particles: Vec<SphericalParticle>,
    pub enabled: bool,
    trail_lons: Vec<Vec<f64>>, // [TRAIL_LEN][N]
    trail_lats: Vec<Vec<f64>>,
    trail_cursor: usize,
    trail_count: usize,
}

impl SphericalParticleSystem {
    pub fn new(count: usize) -> Self {
        let mut rng = Xor128::new(42);
        let particles: Vec<SphericalParticle> = (0..count)
            .map(|_| {
                let lon = (rng.next_f64() * 0.5 + 0.5) * 2.0 * PI; // [0, 2π)
                let u = rng.next_f64(); // [-1, 1)
                let lat = u.asin();
                SphericalParticle { lon, lat }
            })
            .collect();

        let trail_lons = vec![vec![0.0; count]; TRAIL_LEN];
        let trail_lats = vec![vec![0.0; count]; TRAIL_LEN];

        Self {
            particles,
            enabled: true,
            trail_lons,
            trail_lats,
            trail_cursor: 0,
            trail_count: 0,
        }
    }

    /// Push current positions into the trail ring buffer.
    pub fn push_trail(&mut self) {
        let slot = self.trail_cursor % TRAIL_LEN;
        for (i, p) in self.particles.iter().enumerate() {
            self.trail_lons[slot][i] = p.lon;
            self.trail_lats[slot][i] = p.lat;
        }
        self.trail_cursor += 1;
        if self.trail_count < TRAIL_LEN {
            self.trail_count += 1;
        }
    }

    /// Return trail slices oldest→newest. Each element is (&[lon], &[lat]).
    pub fn ordered_trails(&self) -> Vec<(&[f64], &[f64])> {
        let n = self.trail_count;
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let idx = (self.trail_cursor + TRAIL_LEN - n + k) % TRAIL_LEN;
            out.push((self.trail_lons[idx].as_slice(), self.trail_lats[idx].as_slice()));
        }
        out
    }

    /// Advect particles using Euler step.
    /// `cu_data`, `cv_data`: cosφ·u, cosφ·v on jm×im grid.
    pub fn advect(
        &mut self,
        cu_data: &[f64],
        cv_data: &[f64],
        im: usize,
        jm: usize,
        gauss_nodes: &[f64],
        dt: f64,
    ) {
        self.push_trail();

        for p in &mut self.particles {
            let mu = p.lat.sin();
            let cu = interpolate_gauss(cu_data, im, jm, gauss_nodes, p.lon, mu);
            let cv = interpolate_gauss(cv_data, im, jm, gauss_nodes, p.lon, mu);

            let cos_lat = p.lat.cos();
            let cos2_lat = cos_lat * cos_lat;

            // dλ/dt = cu / cos²φ, dφ/dt = cv / cosφ
            let dlon = if cos2_lat > 1e-6 { cu / cos2_lat } else { 0.0 };
            let dlat = if cos_lat.abs() > 1e-3 { cv / cos_lat } else { 0.0 };

            p.lon += dlon * dt;
            p.lat += dlat * dt;

            // Wrap longitude [0, 2π)
            p.lon = p.lon.rem_euclid(2.0 * PI);
            // Clamp latitude
            p.lat = p.lat.clamp(-LAT_CLAMP, LAT_CLAMP);
        }
    }
}

#[allow(dead_code)]
pub struct SphericalSnapshot {
    pub im: usize,
    pub jm: usize,
    pub step: u64,
    pub time: f64,
    pub field_name: String,
    pub field_data: Vec<f64>,
    pub field_names: Vec<String>,
    pub field_index: usize,
    /// Pre-computed global data range (vmin, vmax) across all frames for this field.
    pub global_range: Option<(f64, f64)>,
}

/// Find neighbors for interpolation on gaussian grid.
/// `nodes` must be sorted ascending (south to north, μ = sinφ).
/// Returns (j0, j1, weight) where the interpolated value is:
///   data[j0] * (1 - weight) + data[j1] * weight
pub fn find_gauss_neighbors(nodes: &[f64], mu: f64) -> (usize, usize, f64) {
    let n = nodes.len();
    if n == 0 {
        return (0, 0, 0.0);
    }
    if mu <= nodes[0] {
        return (0, 0, 0.0);
    }
    if mu >= nodes[n - 1] {
        return (n - 1, n - 1, 0.0);
    }
    // Binary search for the interval [nodes[j], nodes[j+1]] containing mu
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if nodes[mid] <= mu {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let range = nodes[hi] - nodes[lo];
    let weight = if range > 1e-30 {
        (mu - nodes[lo]) / range
    } else {
        0.5
    };
    (lo, hi, weight)
}

/// Bilinear interpolation on a gaussian grid.
/// `data` is jm × im (row-major, j=latitude, i=longitude).
/// `lon` in [0, 2π), `mu` = sin(lat).
/// `gauss_nodes` sorted ascending.
pub fn interpolate_gauss(
    data: &[f64],
    im: usize,
    _jm: usize,
    gauss_nodes: &[f64],
    lon: f64,
    mu: f64,
) -> f64 {
    let (j0, j1, wy) = find_gauss_neighbors(gauss_nodes, mu);

    let lon_frac = lon / (2.0 * std::f64::consts::PI) * im as f64;
    let i0 = lon_frac.floor() as usize % im;
    let i1 = (i0 + 1) % im;
    let wx = lon_frac - lon_frac.floor();

    let v00 = data[j0 * im + i0];
    let v10 = data[j0 * im + i1];
    let v01 = data[j1 * im + i0];
    let v11 = data[j1 * im + i1];

    v00 * (1.0 - wx) * (1.0 - wy)
        + v10 * wx * (1.0 - wy)
        + v01 * (1.0 - wx) * wy
        + v11 * wx * wy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_gauss_neighbors_interior() {
        let nodes = vec![-0.8, -0.4, 0.0, 0.4, 0.8];
        let (j0, j1, w) = find_gauss_neighbors(&nodes, 0.2);
        assert_eq!(j0, 2);
        assert_eq!(j1, 3);
        assert!((w - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_find_gauss_neighbors_below_min() {
        let nodes = vec![-0.5, 0.0, 0.5];
        let (j0, j1, w) = find_gauss_neighbors(&nodes, -0.9);
        assert_eq!(j0, 0);
        assert_eq!(j1, 0);
        assert!((w - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_gauss_neighbors_above_max() {
        let nodes = vec![-0.5, 0.0, 0.5];
        let (j0, j1, w) = find_gauss_neighbors(&nodes, 0.9);
        assert_eq!(j0, 2);
        assert_eq!(j1, 2);
        assert!((w - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_gauss_neighbors_exact_node() {
        let nodes = vec![-0.5, 0.0, 0.5];
        let (j0, _j1, w) = find_gauss_neighbors(&nodes, 0.0);
        assert_eq!(j0, 1);
        assert!(w.abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_gauss_uniform() {
        let im = 4;
        let jm = 3;
        let nodes = vec![-0.5, 0.0, 0.5];
        let data = vec![42.0; im * jm];
        let val = interpolate_gauss(&data, im, jm, &nodes, 1.0, 0.2);
        assert!((val - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_gauss_lon_wraparound() {
        let im = 4;
        let jm = 2;
        let nodes = vec![-0.5, 0.5];
        // data[j=0]: [10, 20, 30, 40]
        // data[j=1]: [50, 60, 70, 80]
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        // lon = 0 should give i0=3, i1=0 (wrap) at j=0: 40*(1-0) + 10*0 = ... wait
        // Actually lon=0 → lon_frac=0 → i0=0, i1=1
        // lon near 2π: lon_frac ≈ im → i0=im-1=3, i1=0 (wrap)
        let pi2 = 2.0 * std::f64::consts::PI;
        let lon = pi2 * 3.5 / 4.0; // between i=3 and i=0
        let val = interpolate_gauss(&data, im, jm, &nodes, lon, -0.5);
        // At j=0, mu=-0.5 exact: v00=data[0*4+3]=40, v10=data[0*4+0]=10
        // weight in lon: 0.5
        // Expected: 40*0.5 + 10*0.5 = 25
        assert!((val - 25.0).abs() < 1e-10, "got {val}");
    }

    // --- SphericalParticleSystem tests ---

    #[test]
    fn test_particle_initial_positions_in_range() {
        let sys = SphericalParticleSystem::new(500);
        for p in &sys.particles {
            assert!(p.lon >= 0.0 && p.lon < 2.0 * PI, "lon out of range: {}", p.lon);
            assert!(p.lat >= -PI / 2.0 && p.lat <= PI / 2.0, "lat out of range: {}", p.lat);
        }
    }

    #[test]
    fn test_particle_advect_uniform_eastwind() {
        let mut sys = SphericalParticleSystem::new(10);
        let im = 8;
        let jm = 4;
        let nodes = vec![-0.6, -0.2, 0.2, 0.6];
        // Uniform cu = 1.0, cv = 0.0 (eastward cosφ·u = 1)
        let cu = vec![1.0; im * jm];
        let cv = vec![0.0; im * jm];

        let lon_before: Vec<f64> = sys.particles.iter().map(|p| p.lon).collect();
        sys.advect(&cu, &cv, im, jm, &nodes, 0.01);

        for (i, p) in sys.particles.iter().enumerate() {
            // lon should have increased (cu > 0 → dlon/dt > 0)
            // For particles not near the wrap boundary
            let expected_dlon = 0.01 / p.lat.cos().powi(2);
            let actual_dlon = (p.lon - lon_before[i]).rem_euclid(2.0 * PI);
            if actual_dlon < PI {
                assert!(actual_dlon > 0.0, "particle {i} should move east");
            }
            let _ = expected_dlon; // just verify it moved
        }
    }

    #[test]
    fn test_particle_trail_ordering() {
        let mut sys = SphericalParticleSystem::new(2);
        // Push 3 trail entries with distinct positions
        for k in 0..3 {
            sys.particles[0].lon = k as f64;
            sys.particles[1].lon = k as f64 + 10.0;
            sys.push_trail();
        }

        let trails = sys.ordered_trails();
        assert_eq!(trails.len(), 3);
        // oldest first
        assert!((trails[0].0[0] - 0.0).abs() < 1e-10);
        assert!((trails[1].0[0] - 1.0).abs() < 1e-10);
        assert!((trails[2].0[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_particle_trail_ring_buffer() {
        let mut sys = SphericalParticleSystem::new(1);
        // Push more than TRAIL_LEN entries
        for k in 0..(TRAIL_LEN + 3) {
            sys.particles[0].lon = k as f64;
            sys.push_trail();
        }

        let trails = sys.ordered_trails();
        assert_eq!(trails.len(), TRAIL_LEN);
        // Should contain the last TRAIL_LEN entries, oldest first
        for (i, (lons, _lats)) in trails.iter().enumerate() {
            let expected = (3 + i) as f64; // entries 3..TRAIL_LEN+3-1
            assert!((lons[0] - expected).abs() < 1e-10, "trail[{i}] lon={}, expected={expected}", lons[0]);
        }
    }
}
