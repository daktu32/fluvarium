// Spherical data snapshot for playback rendering.

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
}
