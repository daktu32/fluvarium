use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::renderer;
use crate::solver::{self, SolverParams};
use crate::state::{FluidModel, FrameSnapshot, SimState, N};

/// Compute simulation grid width from window dimensions and model.
/// Karman: NX = N * (win_w / win_h), clamped to at least N.
/// RB: always N (tiles handle horizontal extent).
pub fn compute_sim_nx(win_w: usize, win_h: usize, model: FluidModel) -> usize {
    match model {
        FluidModel::KarmanVortex => {
            let aspect = win_w as f64 / win_h as f64;
            let nx = (N as f64 * aspect).round() as usize;
            nx.max(N)
        }
        _ => N,
    }
}

/// Create a SimState for the given model, dispatching to the appropriate constructor.
pub fn create_sim_state(
    model: FluidModel,
    params: &SolverParams,
    num_particles: usize,
    nx: usize,
) -> SimState {
    match model {
        FluidModel::KarmanVortex => SimState::new_karman(
            num_particles,
            params.inflow_vel,
            params.cylinder_x,
            params.cylinder_y,
            params.cylinder_radius,
            nx,
        ),
        FluidModel::KelvinHelmholtz => SimState::new_kh(num_particles, params, nx),
        FluidModel::LidDrivenCavity => SimState::new_cavity(num_particles, nx),
        FluidModel::RayleighBenard if params.benchmark_mode => SimState::new_benchmark(num_particles, nx),
        _ => SimState::new(num_particles, params.bottom_base, nx),
    }
}

/// Per-model parameter storage with save/restore on model switch.
pub struct ModelParams {
    pub rb: SolverParams,
    pub karman: SolverParams,
    pub kh: SolverParams,
    pub cavity: SolverParams,
}

impl ModelParams {
    pub fn new() -> Self {
        Self {
            rb: SolverParams::default(),
            karman: SolverParams::default_karman(),
            kh: SolverParams::default_kh(),
            cavity: SolverParams::default_cavity(),
        }
    }

    pub fn get(&self, model: FluidModel) -> &SolverParams {
        match model {
            FluidModel::KarmanVortex => &self.karman,
            FluidModel::KelvinHelmholtz => &self.kh,
            FluidModel::LidDrivenCavity => &self.cavity,
            _ => &self.rb,
        }
    }

    /// Save current params for old_model, cycle to next model, return (new_model, restored_params).
    pub fn save_and_switch(
        &mut self,
        current: &SolverParams,
        old_model: FluidModel,
    ) -> (FluidModel, SolverParams) {
        match old_model {
            FluidModel::RayleighBenard => self.rb = current.clone(),
            FluidModel::KarmanVortex => self.karman = current.clone(),
            FluidModel::KelvinHelmholtz => self.kh = current.clone(),
            FluidModel::LidDrivenCavity => self.cavity = current.clone(),
        }
        // Cycle: RB -> Karman -> KH -> Cavity -> RB
        let new_model = match old_model {
            FluidModel::RayleighBenard => FluidModel::KarmanVortex,
            FluidModel::KarmanVortex => FluidModel::KelvinHelmholtz,
            FluidModel::KelvinHelmholtz => FluidModel::LidDrivenCavity,
            FluidModel::LidDrivenCavity => FluidModel::RayleighBenard,
        };
        let restored = self.get(new_model).clone();
        (new_model, restored)
    }
}

/// Channels connecting the main (render) thread to the physics thread.
pub struct PhysicsChannels {
    pub param_tx: mpsc::Sender<SolverParams>,
    pub reset_tx: mpsc::Sender<(FluidModel, SimState)>,
    pub snap_rx: mpsc::Receiver<FrameSnapshot>,
    pub snap_return_tx: mpsc::Sender<FrameSnapshot>,
}

/// Spawn the physics simulation thread and return its channels + join handle.
pub fn spawn_physics_thread(
    model: FluidModel,
    params: SolverParams,
    num_particles: usize,
    sim_nx: usize,
    steps_per_frame: usize,
    running: Arc<AtomicBool>,
    export_path: Option<String>,
) -> (PhysicsChannels, std::thread::JoinHandle<()>) {
    let (param_tx, param_rx) = mpsc::channel::<SolverParams>();
    let (reset_tx, reset_rx) = mpsc::channel::<(FluidModel, SimState)>();
    let (snap_tx, snap_rx) = mpsc::sync_channel::<FrameSnapshot>(1);
    let (snap_return_tx, snap_return_rx) = mpsc::channel::<FrameSnapshot>();

    let handle = std::thread::spawn(move || {
        let mut cur_model = model;
        let mut sim = create_sim_state(cur_model, &params, num_particles, sim_nx);
        let mut params = params;
        let mut snap_buf = FrameSnapshot::new_empty(sim.nx, N, num_particles);
        let mut step_count: u64 = 0;

        // Setup NetCDF exporter for benchmark mode
        let mut writer = if params.benchmark_mode {
            export_path.as_ref().and_then(|path| {
                create_benchmark_writer(path, sim.nx).ok()
            })
        } else {
            None
        };
        let export_interval: u64 = 100; // export every 100 steps

        while running.load(Ordering::SeqCst) {
            while let Ok((new_model, new_sim)) = reset_rx.try_recv() {
                cur_model = new_model;
                sim = new_sim;
                snap_buf = FrameSnapshot::new_empty(sim.nx, N, num_particles);
            }
            while let Ok(new_params) = param_rx.try_recv() {
                params = new_params;
            }
            for _ in 0..steps_per_frame {
                match cur_model {
                    FluidModel::KarmanVortex => solver::fluid_step_karman(&mut sim, &params),
                    FluidModel::KelvinHelmholtz => solver::fluid_step_kh(&mut sim, &params),
                    FluidModel::LidDrivenCavity => solver::fluid_step_cavity(&mut sim, &params),
                    FluidModel::RayleighBenard if params.benchmark_mode => {
                        solver::fluid_step_benchmark(&mut sim, &params);
                    }
                    _ => solver::fluid_step(&mut sim, &params),
                }
                step_count += 1;

                // Benchmark export + diagnostics
                if params.benchmark_mode && step_count % export_interval == 0 {
                    let nu = solver::diagnostics::compute_nusselt(
                        &sim.vy, &sim.temperature, params.diff, sim.nx,
                    );
                    let ke = solver::diagnostics::compute_kinetic_energy(
                        &sim.vx, &sim.vy, sim.nx,
                    );
                    let time = step_count as f64 * params.dt;
                    eprintln!("step={} t={:.4} Nu={:.4} KE={:.6e}", step_count, time, nu, ke);

                    if let Some(ref mut w) = writer {
                        let _ = write_benchmark_frame(
                            w, &sim, &params, step_count, time,
                        );
                    }
                }
            }
            sim.snapshot_into(&mut snap_buf);
            if snap_tx.send(snap_buf).is_err() {
                break;
            }
            let expected_len = sim.nx * N;
            snap_buf = snap_return_rx
                .try_recv()
                .ok()
                .filter(|b| b.temperature.len() == expected_len)
                .unwrap_or_else(|| FrameSnapshot::new_empty(sim.nx, N, num_particles));
        }

        // Finalize writer
        if let Some(w) = writer {
            let _ = w.finish();
            eprintln!("Export complete.");
        }
    });

    let channels = PhysicsChannels {
        param_tx,
        reset_tx,
        snap_rx,
        snap_return_tx,
    };
    (channels, handle)
}

/// Create a GtoolWriter for benchmark mode NetCDF export.
fn create_benchmark_writer(
    path: &str,
    nx: usize,
) -> Result<gtool_rs::writer::GtoolWriter, Box<dyn std::error::Error>> {
    use gtool_rs::types::*;

    let grid = GridSpec {
        grid_type: GridType::Channel,
        im: nx,
        jm: N,
        nm: 0,
    };
    let time = TimeInfo {
        dt: 0.003,
        output_interval: 100,
    };
    let mut writer = gtool_rs::writer::GtoolWriter::create(path, &grid, "fludarium_rb_benchmark", &time)?;

    // Channel domain: x is periodic [0, nx), z is [0, 1] (normalized height)
    writer.set_channel_domain(nx as f64, 1.0)?;

    // Define fields
    writer.define_field(&FieldMeta {
        name: "theta".to_string(),
        units: "1".to_string(),
        long_name: "Perturbation temperature (T - T_cond)".to_string(),
    })?;
    writer.define_field(&FieldMeta {
        name: "zeta".to_string(),
        units: "1/s".to_string(),
        long_name: "Vorticity (dvy/dx - dvx/dy)".to_string(),
    })?;
    writer.define_field(&FieldMeta {
        name: "psi".to_string(),
        units: "1".to_string(),
        long_name: "Stream function".to_string(),
    })?;

    Ok(writer)
}

/// Write a single benchmark frame (theta, zeta, psi) to the NetCDF file.
fn write_benchmark_frame(
    writer: &mut gtool_rs::writer::GtoolWriter,
    sim: &SimState,
    _params: &SolverParams,
    step: u64,
    time: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = sim.nx;

    // Compute theta
    let theta = solver::diagnostics::compute_theta(&sim.temperature, nx);

    // Compute vorticity
    let (zeta, _, _) = renderer::compute_vorticity(&sim.vx, &sim.vy, nx);

    // Compute stream function
    let (psi, _, _) = renderer::compute_stream_function(&sim.vx, nx);

    let frame = gtool_rs::types::Frame {
        step,
        time,
        fields: vec![
            ("theta".to_string(), theta),
            ("zeta".to_string(), zeta),
            ("psi".to_string(), psi),
        ],
    };
    writer.write_frame(&frame)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_params_save_and_switch() {
        let mut mp = ModelParams::new();
        let mut current = mp.get(FluidModel::KarmanVortex).clone();
        // Modify a Karman param
        current.visc = 0.999;
        // Switch from Karman -> KH (4-model cycle: RB -> Karman -> KH -> Cavity -> RB)
        let (new_model, restored) = mp.save_and_switch(&current, FluidModel::KarmanVortex);
        assert!(matches!(new_model, FluidModel::KelvinHelmholtz));
        // Restored should be KH defaults
        assert!((restored.visc - SolverParams::default_kh().visc).abs() < 1e-10);
        // Saved Karman params should have our modification
        assert!((mp.karman.visc - 0.999).abs() < 1e-10);
        // Switch from KH -> Cavity
        let (cavity_model, cavity_params) = mp.save_and_switch(&restored, new_model);
        assert!(matches!(cavity_model, FluidModel::LidDrivenCavity));
        assert!((cavity_params.visc - SolverParams::default_cavity().visc).abs() < 1e-10);
        // Switch from Cavity -> RB
        let (rb_model, rb_params) = mp.save_and_switch(&cavity_params, cavity_model);
        assert!(matches!(rb_model, FluidModel::RayleighBenard));
        assert!((rb_params.visc - SolverParams::default().visc).abs() < 1e-10);
        // Switch from RB -> Karman, should get our modified Karman visc back
        let (back_model, back_params) = mp.save_and_switch(&rb_params, rb_model);
        assert!(matches!(back_model, FluidModel::KarmanVortex));
        assert!((back_params.visc - 0.999).abs() < 1e-10);
    }

    #[test]
    fn test_compute_sim_nx_rb() {
        let nx = compute_sim_nx(800, 600, FluidModel::RayleighBenard);
        assert_eq!(nx, N, "RB always uses N");
    }

    #[test]
    fn test_compute_sim_nx_karman() {
        let nx = compute_sim_nx(800, 400, FluidModel::KarmanVortex);
        // aspect = 800/400 = 2.0, so nx = N * 2
        assert_eq!(nx, N * 2);
    }

    #[test]
    fn test_compute_sim_nx_karman_min() {
        // Very tall window: aspect < 1
        let nx = compute_sim_nx(200, 800, FluidModel::KarmanVortex);
        assert_eq!(nx, N, "Karman NX should not go below N");
    }

    #[test]
    fn test_qa_compute_sim_nx_kh() {
        let nx = compute_sim_nx(800, 600, FluidModel::KelvinHelmholtz);
        assert_eq!(nx, N, "KH model should use N for nx");
    }
}
