use crate::state::{idx, idx_inner, N};
use super::boundary::{BoundaryConfig, FieldType, set_bnd};

/// Gauss-Seidel iterative linear solver.
/// Solves: x[i,j] = (x0[i,j] + a * (neighbors)) / c
pub fn lin_solve(field_type: FieldType, x: &mut [f64], x0: &[f64], a: f64, c: f64, iter: usize, bc: &BoundaryConfig, nx: usize) {
    let c_inv = 1.0 / c;
    // All modes iterate over the full X range; set_bnd overwrites boundary
    // cells after each iteration, so including them is harmless and keeps
    // the solver uniform across models.
    let periodic_x = bc.periodic_x();
    for _ in 0..iter {
        for j in 1..(N - 1) {
            for i in 0..nx {
                let im1 = if periodic_x {
                    if i == 0 { nx - 1 } else { i - 1 }
                } else {
                    i.saturating_sub(1)
                };
                let ip1 = if periodic_x {
                    if i == nx - 1 { 0 } else { i + 1 }
                } else {
                    (i + 1).min(nx - 1)
                };
                let neighbors = x[idx_inner(im1, j, nx)]
                    + x[idx_inner(ip1, j, nx)]
                    + x[idx_inner(i, j - 1, nx)]
                    + x[idx_inner(i, j + 1, nx)];
                x[idx_inner(i, j, nx)] = (x0[idx_inner(i, j, nx)] + a * neighbors) * c_inv;
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
    let (i_lo, i_hi) = bc.x_range(nx);
    let periodic_x = bc.periodic_x();

    for j in 1..(N - 1) {
        for i in i_lo..i_hi {
            let ii = idx_inner(i, j, nx);
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

            // X clamping for non-periodic Karman mode
            let x = match bc {
                BoundaryConfig::KarmanVortex { .. } => x.clamp(0.5, nx_f - 1.5),
                _ => x, // RB: periodic, no clamping
            };

            let j0 = y.floor() as usize;
            let j1 = j0 + 1;
            let s1 = x - x.floor();
            let s0 = 1.0 - s1;
            let t1 = y - j0 as f64;
            let t0 = 1.0 - t1;

            if periodic_x {
                let i0 = x.floor() as i32;
                let i1 = i0 + 1;
                d[ii] = s0 * (t0 * d0[idx(i0, j0 as i32, nx)] + t1 * d0[idx(i0, j1 as i32, nx)])
                    + s1 * (t0 * d0[idx(i1, j0 as i32, nx)] + t1 * d0[idx(i1, j1 as i32, nx)]);
            } else {
                let i0 = x.floor() as usize;
                let i1 = i0 + 1;
                d[ii] = s0 * (t0 * d0[idx_inner(i0, j0, nx)] + t1 * d0[idx_inner(i0, j1, nx)])
                    + s1 * (t0 * d0[idx_inner(i1, j0, nx)] + t1 * d0[idx_inner(i1, j1, nx)]);
            }
        }
    }
    set_bnd(field_type, d, bc, nx);
}

/// Pressure projection: enforces incompressibility (divergence-free velocity field).
pub fn project(vx: &mut [f64], vy: &mut [f64], p: &mut [f64], div: &mut [f64], iter: usize, bc: &BoundaryConfig, nx: usize) {
    let h = 1.0 / (N - 2) as f64;

    // For Karman mode, skip X boundary cells to prevent periodic wrap contamination.
    let (i_lo, i_hi) = bc.x_range(nx);
    let periodic_x = bc.periodic_x();

    // Calculate divergence
    for j in 1..(N - 1) {
        for i in i_lo..i_hi {
            let im1 = if periodic_x && i == 0 { nx - 1 } else { i - 1 };
            let ip1 = if periodic_x && i == nx - 1 { 0 } else { i + 1 };
            div[idx_inner(i, j, nx)] = -0.5
                * h
                * (vx[idx_inner(ip1, j, nx)] - vx[idx_inner(im1, j, nx)]
                    + vy[idx_inner(i, j + 1, nx)] - vy[idx_inner(i, j - 1, nx)]);
            p[idx_inner(i, j, nx)] = 0.0;
        }
    }
    set_bnd(FieldType::Scalar, div, bc, nx);
    set_bnd(FieldType::Scalar, p, bc, nx);

    // Solve for pressure
    lin_solve(FieldType::Scalar, p, div, 1.0, 4.0, iter, bc, nx);

    // Subtract pressure gradient from velocity
    for j in 1..(N - 1) {
        for i in i_lo..i_hi {
            let im1 = if periodic_x && i == 0 { nx - 1 } else { i - 1 };
            let ip1 = if periodic_x && i == nx - 1 { 0 } else { i + 1 };
            vx[idx_inner(i, j, nx)] -=
                0.5 * (p[idx_inner(ip1, j, nx)] - p[idx_inner(im1, j, nx)]) / h;
            vy[idx_inner(i, j, nx)] -=
                0.5 * (p[idx_inner(i, j + 1, nx)] - p[idx_inner(i, j - 1, nx)]) / h;
        }
    }
    set_bnd(FieldType::Vx, vx, bc, nx);
    set_bnd(FieldType::Vy, vy, bc, nx);
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
}
