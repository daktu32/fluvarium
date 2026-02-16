use crate::state::N;

/// Solver parameters for the fluid simulation.
#[derive(Clone)]
pub struct SolverParams {
    pub visc: f64,
    pub diff: f64,
    pub dt: f64,
    pub diffuse_iter: usize,
    pub project_iter: usize,
    pub heat_buoyancy: f64,
    #[allow(dead_code)] // reserved for future use
    pub noise_amp: f64,
    pub source_strength: f64,
    pub cool_rate: f64,
    pub bottom_base: f64,
    pub inflow_vel: f64,
    pub cylinder_x: f64,
    pub cylinder_y: f64,
    pub cylinder_radius: f64,
    pub confinement: f64,
    pub shear_velocity: f64,
    pub shear_relax: f64,
    pub shear_thickness: f64,
    pub lid_velocity: f64,
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
            shear_velocity: 0.0,
            shear_relax: 0.0,
            shear_thickness: 3.0,
            lid_velocity: 0.0,
        }
    }
}

impl SolverParams {
    /// Default parameters optimized for Karman vortex street at Re=100.
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
            shear_velocity: 0.0,
            shear_relax: 0.0,
            shear_thickness: 3.0,
            lid_velocity: 0.0,
        }
    }

    /// Default parameters for Kelvin-Helmholtz instability.
    pub fn default_kh() -> Self {
        Self {
            visc: 0.001,
            diff: 0.0005,
            dt: 0.05,
            diffuse_iter: 20,
            project_iter: 30,
            heat_buoyancy: 0.0,
            noise_amp: 0.0,
            source_strength: 0.0,
            cool_rate: 0.0,
            bottom_base: 0.0,
            inflow_vel: 0.0,
            cylinder_x: 0.0,
            cylinder_y: 0.0,
            cylinder_radius: 0.0,
            confinement: 0.0,
            shear_velocity: 0.08,
            shear_relax: 1.0,
            shear_thickness: 3.0,
            lid_velocity: 0.0,
        }
    }

    /// Default parameters for Lid-Driven Cavity flow.
    pub fn default_cavity() -> Self {
        Self {
            visc: 0.01,
            diff: 0.001,
            dt: 0.05,
            diffuse_iter: 20,
            project_iter: 30,
            lid_velocity: 1.0,
            heat_buoyancy: 0.0,
            noise_amp: 0.0,
            source_strength: 0.0,
            cool_rate: 0.0,
            bottom_base: 0.0,
            inflow_vel: 0.0,
            cylinder_x: 0.0,
            cylinder_y: 0.0,
            cylinder_radius: 0.0,
            confinement: 0.0,
            shear_velocity: 0.0,
            shear_relax: 0.0,
            shear_thickness: 3.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::N;

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
    fn test_default_cavity_params() {
        let params = SolverParams::default_cavity();
        assert_eq!(params.visc, 0.01);
        assert_eq!(params.diff, 0.001);
        assert_eq!(params.dt, 0.05);
        assert_eq!(params.lid_velocity, 1.0);
        assert_eq!(params.diffuse_iter, 20);
        assert_eq!(params.project_iter, 30);
        assert_eq!(params.heat_buoyancy, 0.0);
        assert_eq!(params.inflow_vel, 0.0);
        assert_eq!(params.shear_velocity, 0.0);
    }
}
