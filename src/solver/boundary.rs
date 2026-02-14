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

impl BoundaryConfig {
    /// Whether the X-axis uses periodic (wrapping) boundaries.
    pub fn periodic_x(&self) -> bool {
        matches!(self, BoundaryConfig::RayleighBenard { .. })
    }

    /// X iteration range for interior loops.
    /// RayleighBenard: full width `(0, nx)` (periodic wrap handled by `idx`).
    /// KarmanVortex: skip boundary columns `(1, nx-1)`.
    pub fn x_range(&self, nx: usize) -> (usize, usize) {
        match self {
            BoundaryConfig::KarmanVortex { .. } => (1, nx - 1),
            _ => (0, nx),
        }
    }
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
///   - `FieldType::Scalar`: Neumann (copy neighbor) at walls
///   - `FieldType::Vx`: negate at walls (no-slip)
///   - `FieldType::Vy`: negate at walls (no-slip + no-penetration)
///   - `FieldType::Temperature`: top/bottom Dirichlet (hot bottom, cold top)
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
                // Top: cold
                x[idx(i as i32, (N - 1) as i32, nx)] = 0.0;
            }
            _ => {
                x[idx(i as i32, 0, nx)] = x[idx(i as i32, 1, nx)];
                x[idx(i as i32, (N - 1) as i32, nx)] = x[idx(i as i32, (N - 2) as i32, nx)];
            }
        }
    }
}

/// Karman vortex street boundary conditions.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{idx, N};

    const BB: f64 = 0.15;

    fn rb_bc() -> BoundaryConfig {
        BoundaryConfig::RayleighBenard { bottom_base: BB }
    }

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
            assert!((temperature[idx(i as i32, (N - 2) as i32, N)] - 0.5).abs() < 1e-10, "y=N-2 should remain interior value");
        }
        // y=2 should NOT be overwritten (remains 0.5)
        let t2 = temperature[idx(0, 2, N)];
        assert!((t2 - 0.5).abs() < 1e-10, "y=2 should remain interior value, got {}", t2);
    }
}
