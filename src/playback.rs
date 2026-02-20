// Playback controller for spmodel-rs .spg and .nc output.

use crate::spherical::{SphericalParticleSystem, SphericalSnapshot, PARTICLE_COUNT};
use crate::spgrid::{SpgFrame, SpgReader};
use gtool_rs::reader::GtoolReader;

// --- Channel particle system ---

const CHANNEL_PARTICLE_COUNT: usize = 500;
const CHANNEL_TRAIL_LEN: usize = 32;
const CHANNEL_SUBSTEPS: usize = 4;

pub struct ChannelParticles {
    pub x: Vec<f64>,
    pub z: Vec<f64>,
    pub enabled: bool,
    trail_x: Vec<Vec<f64>>,
    trail_z: Vec<Vec<f64>>,
    trail_cursor: usize,
    trail_count: usize,
}

impl ChannelParticles {
    pub fn new(count: usize, lx: f64, lz: f64) -> Self {
        let mut x = Vec::with_capacity(count);
        let mut z = Vec::with_capacity(count);
        let mut rng: u64 = 0xDEAD_BEEF_CAFE;
        for _ in 0..count {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let rx = (rng >> 33) as f64 / (1u64 << 31) as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let rz = (rng >> 33) as f64 / (1u64 << 31) as f64;
            x.push(rx * lx);
            z.push(rz * lz * 0.96 + lz * 0.02);
        }
        let trail_x = vec![vec![0.0; count]; CHANNEL_TRAIL_LEN];
        let trail_z = vec![vec![0.0; count]; CHANNEL_TRAIL_LEN];
        Self { x, z, enabled: true, trail_x, trail_z, trail_cursor: 0, trail_count: 0 }
    }

    fn push_trail(&mut self) {
        let slot = self.trail_cursor % CHANNEL_TRAIL_LEN;
        for i in 0..self.x.len() {
            self.trail_x[slot][i] = self.x[i];
            self.trail_z[slot][i] = self.z[i];
        }
        self.trail_cursor += 1;
        if self.trail_count < CHANNEL_TRAIL_LEN {
            self.trail_count += 1;
        }
    }

    pub fn ordered_trails(&self) -> Vec<(&[f64], &[f64])> {
        let n = self.trail_count;
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let idx = (self.trail_cursor + CHANNEL_TRAIL_LEN - n + k) % CHANNEL_TRAIL_LEN;
            out.push((self.trail_x[idx].as_slice(), self.trail_z[idx].as_slice()));
        }
        out
    }

    /// Advect particles using stream function psi with sub-stepping.
    /// u = dψ/dz (horizontal), w = -dψ/dx (vertical).
    /// Pushes a trail entry at each sub-step for smooth trails.
    pub fn advect(&mut self, psi: &[f64], nx: usize, nz: usize, lx: f64, lz: f64, dt: f64) {
        let dx = lx / nx as f64;
        let dz = lz / (nz + 1) as f64;
        let eps_x = dx * 0.5;
        let eps_z = dz * 0.5;
        let sub_dt = dt / CHANNEL_SUBSTEPS as f64;
        let wall_eps = lz * 0.001;

        for _ in 0..CHANNEL_SUBSTEPS {
            self.push_trail();

            for i in 0..self.x.len() {
                let px = self.x[i];
                let pz = self.z[i];

                // u = ∂ψ/∂z
                let psi_up = interp_psi(psi, nx, nz, lx, lz, px, pz + eps_z);
                let psi_dn = interp_psi(psi, nx, nz, lx, lz, px, pz - eps_z);
                let u = (psi_up - psi_dn) / (2.0 * eps_z);

                // w = -∂ψ/∂x
                let psi_rt = interp_psi(psi, nx, nz, lx, lz, (px + eps_x).rem_euclid(lx), pz);
                let psi_lt = interp_psi(psi, nx, nz, lx, lz, (px - eps_x).rem_euclid(lx), pz);
                let w = -(psi_rt - psi_lt) / (2.0 * eps_x);

                self.x[i] = (px + u * sub_dt).rem_euclid(lx);
                self.z[i] = (pz + w * sub_dt).clamp(wall_eps, lz - wall_eps);
            }
        }
    }
}

/// Bilinear interpolation of psi on the channel grid.
/// Grid: psi[j * nx + i], z_j = (j+1)*lz/(nz+1), x_i = i*lx/nx (periodic).
/// Boundary: psi = 0 at z=0 and z=lz.
fn interp_psi(psi: &[f64], nx: usize, nz: usize, lx: f64, lz: f64, px: f64, pz: f64) -> f64 {
    let dx = lx / nx as f64;
    let dz = lz / (nz + 1) as f64;

    // x: periodic
    let fx = px / dx;
    let ix0f = fx.floor();
    let ix0 = ((ix0f as isize) % nx as isize + nx as isize) as usize % nx;
    let ix1 = (ix0 + 1) % nx;
    let sx = fx - ix0f;

    // z: interior points at z_j = (j+1)*dz, so j = pz/dz - 1
    let fz = pz / dz - 1.0;
    let jz0 = fz.floor() as isize;
    let jz1 = jz0 + 1;
    let sz = fz - jz0 as f64;

    let val = |j: isize, i: usize| -> f64 {
        if j < 0 || j >= nz as isize { 0.0 } else { psi[j as usize * nx + i] }
    };

    val(jz0, ix0) * (1.0 - sx) * (1.0 - sz)
        + val(jz0, ix1) * sx * (1.0 - sz)
        + val(jz1, ix0) * (1.0 - sx) * sz
        + val(jz1, ix1) * sx * sz
}

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
    /// Domain length for 1D periodic data.
    domain_length: f64,
    /// True if this is a channel grid (2D rectangular, not spherical).
    is_channel: bool,
    /// Channel domain (lx, lz) if this is a channel grid.
    channel_domain: (f64, f64),
    /// Index of "psi" field (for channel particle advection).
    psi_index: Option<usize>,
    /// Channel particle system (only for channel grids with psi field).
    pub channel_particles: Option<ChannelParticles>,
    /// Index of "vort" field (for gyre center tracking).
    vort_index: Option<usize>,
    /// Pre-computed gyre center (lon, lat) in radians per frame.
    gyre_centers: Vec<(f64, f64)>,
    /// Whether to display the gyre trajectory track.
    pub show_gyre_track: bool,
}

impl PlaybackState {
    pub fn from_reader(reader: &SpgReader) -> std::io::Result<Self> {
        let frames = reader.read_all_frames()?;
        let im = reader.manifest.grid.im;
        let jm = reader.manifest.grid.jm;
        let field_names = reader.manifest.fields.clone();
        let is_1d = jm == 1;

        // Compute gauss nodes from nm (skip for 1D data)
        let gauss_nodes = if is_1d {
            Vec::new()
        } else {
            let nm = reader.manifest.grid.nm;
            compute_gauss_nodes(nm, jm)
        };

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

        // Detect velocity fields (skip for 1D)
        let u_cos_index = if is_1d { None } else { field_names.iter().position(|n| n == "u_cos") };
        let v_cos_index = if is_1d { None } else { field_names.iter().position(|n| n == "v_cos") };
        let particles = if u_cos_index.is_some() && v_cos_index.is_some() {
            Some(SphericalParticleSystem::new(PARTICLE_COUNT))
        } else {
            None
        };

        let model_dt = reader.manifest.time.dt;
        let output_interval = reader.manifest.time.output_interval;
        let domain_length = reader.manifest.domain_length();

        // Detect vort field and precompute gyre centers
        let vort_index = field_names.iter().position(|n| n == "vort");
        let gyre_centers = precompute_gyre_centers(&frames, vort_index, im, jm, &gauss_nodes);
        let show_gyre_track = vort_index.is_some();

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
            domain_length,
            is_channel: false,
            channel_domain: (0.0, 0.0),
            psi_index: None,
            channel_particles: None,
            vort_index,
            gyre_centers,
            show_gyre_track,
        })
    }

    pub fn from_netcdf(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let reader = GtoolReader::open(path)?;
        let info = reader.file_info();
        let frames = reader.read_all_frames()?;
        let im = info.grid.im;
        let jm = info.grid.jm;
        let is_1d = jm == 1;
        let is_channel = info.grid.grid_type == gtool_rs::GridType::Channel;
        let channel_domain = reader.channel_domain().unwrap_or((0.0, 0.0));

        // Read gauss nodes directly from the NetCDF file (skip for 1D and channel)
        let gauss_nodes = if is_1d || is_channel {
            Vec::new()
        } else {
            reader
                .gauss_nodes()
                .unwrap_or_else(|| gtool_rs::coord::gauss_nodes(jm))
        };

        let field_names = info.field_names.clone();
        let spg_frames: Vec<SpgFrame> = frames
            .into_iter()
            .map(|f| SpgFrame {
                step: f.step,
                time: f.time,
                fields: f.fields,
            })
            .collect();

        // Pre-compute global min/max per field across all frames
        let nfields = field_names.len();
        let mut global_ranges = vec![(f64::INFINITY, f64::NEG_INFINITY); nfields];
        for frame in &spg_frames {
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
        for r in &mut global_ranges {
            if r.0 < 0.0 && r.1 > 0.0 {
                let abs_max = r.0.abs().max(r.1.abs());
                r.0 = -abs_max;
                r.1 = abs_max;
            }
        }
        for r in &mut global_ranges {
            let mean = (r.0 + r.1) * 0.5;
            let current_range = r.1 - r.0;
            let min_range = mean.abs() * 0.01;
            if current_range < min_range && mean.abs() > 1e-10 {
                r.0 = mean - min_range * 0.5;
                r.1 = mean + min_range * 0.5;
            }
        }

        // Detect velocity fields (skip for 1D)
        let u_cos_index = if is_1d { None } else { field_names.iter().position(|n| n == "u_cos") };
        let v_cos_index = if is_1d { None } else { field_names.iter().position(|n| n == "v_cos") };
        let particles = if u_cos_index.is_some() && v_cos_index.is_some() {
            Some(SphericalParticleSystem::new(PARTICLE_COUNT))
        } else {
            None
        };

        let model_dt = info.time.dt;
        let output_interval = info.time.output_interval;
        let domain_length = reader.domain_length().unwrap_or(0.0);

        let psi_index = if is_channel {
            field_names.iter().position(|n| n == "psi")
        } else {
            None
        };
        let channel_particles = if is_channel && psi_index.is_some() {
            Some(ChannelParticles::new(
                CHANNEL_PARTICLE_COUNT,
                channel_domain.0,
                channel_domain.1,
            ))
        } else {
            None
        };

        // Try to read pre-computed gyre centers from NC file
        let gyre_from_nc = if !is_1d && !is_channel {
            match (reader.read_scalar("gyre_lon"), reader.read_scalar("gyre_lat")) {
                (Some(lons), Some(lats)) if lons.len() == spg_frames.len() => {
                    Some(lons.into_iter().zip(lats).collect::<Vec<(f64, f64)>>())
                }
                _ => None,
            }
        } else {
            None
        };

        let has_nc_gyre = gyre_from_nc.is_some();
        let vort_index = if is_1d || is_channel {
            None
        } else {
            field_names.iter().position(|n| n == "vort")
        };
        let gyre_centers = gyre_from_nc.unwrap_or_else(|| {
            precompute_gyre_centers(&spg_frames, vort_index, im, jm, &gauss_nodes)
        });
        // Only auto-show when NC contains explicit gyre_lon/gyre_lat;
        // for vort-only data, the track is available via G key but hidden by default.
        let show_gyre_track = has_nc_gyre;

        Ok(Self {
            frames: spg_frames,
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
            model_name: info.model.clone(),
            u_cos_index,
            v_cos_index,
            particles,
            model_dt,
            output_interval,
            domain_length,
            is_channel,
            channel_domain,
            psi_index,
            channel_particles,
            vort_index,
            gyre_centers,
            show_gyre_track,
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
                self.advect_channel_particles();
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

    fn advect_channel_particles(&mut self) {
        let Some(pi) = self.psi_index else { return };
        let Some(ref mut cp) = self.channel_particles else { return };

        let frame = &self.frames[self.current_frame];
        if pi >= frame.fields.len() {
            return;
        }

        let psi_data = &frame.fields[pi].1;
        let (lx, lz) = self.channel_domain;
        let sim_dt = self.model_dt * self.output_interval as f64;

        cp.advect(psi_data, self.im, self.jm, lx, lz, sim_dt);
    }

    pub fn toggle_play(&mut self) {
        self.playing = !self.playing;
    }

    pub fn toggle_particles(&mut self) {
        if let Some(ref mut ps) = self.particles {
            ps.enabled = !ps.enabled;
        }
        if let Some(ref mut cp) = self.channel_particles {
            cp.enabled = !cp.enabled;
        }
    }

    pub fn toggle_gyre_track(&mut self) {
        self.show_gyre_track = !self.show_gyre_track;
    }

    pub fn has_gyre_track(&self) -> bool {
        self.vort_index.is_some()
    }

    pub fn gyre_track_up_to_current(&self) -> &[(f64, f64)] {
        if self.gyre_centers.is_empty() {
            return &[];
        }
        let end = (self.current_frame + 1).min(self.gyre_centers.len());
        &self.gyre_centers[..end]
    }

    pub fn has_particles(&self) -> bool {
        self.particles.is_some() || self.channel_particles.is_some()
    }

    pub fn particles_enabled(&self) -> bool {
        self.particles.as_ref().map_or(false, |ps| ps.enabled)
            || self.channel_particles.as_ref().map_or(false, |cp| cp.enabled)
    }

    pub fn step_forward(&mut self) {
        if self.current_frame + 1 < self.frames.len() {
            self.current_frame += 1;
            self.advect_particles();
            self.advect_channel_particles();
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

    /// True if this is 1D data (jm == 1).
    pub fn is_1d(&self) -> bool {
        self.jm == 1
    }

    /// True if this is a 2D channel grid (not spherical).
    pub fn is_channel(&self) -> bool {
        self.is_channel
    }

    /// Channel domain (lx, lz).
    pub fn channel_domain(&self) -> (f64, f64) {
        self.channel_domain
    }

    /// Domain length for 1D periodic data.
    pub fn domain_length(&self) -> f64 {
        self.domain_length
    }

    /// Current field data slice for 1D rendering.
    pub fn field_data(&self) -> &[f64] {
        let frame = &self.frames[self.current_frame];
        if self.field_index < frame.fields.len() {
            &frame.fields[self.field_index].1
        } else {
            &[]
        }
    }

    /// Global (vmin, vmax) for the currently selected field.
    pub fn current_global_range(&self) -> (f64, f64) {
        self.global_ranges
            .get(self.field_index)
            .copied()
            .unwrap_or((0.0, 1.0))
    }

    /// Step number of the current frame.
    pub fn current_step(&self) -> u64 {
        self.frames[self.current_frame].step
    }

    /// Simulation time of the current frame.
    pub fn current_time(&self) -> f64 {
        self.frames[self.current_frame].time
    }

    /// Name of the currently selected field.
    pub fn current_field_name(&self) -> &str {
        if self.field_index < self.field_names.len() {
            &self.field_names[self.field_index]
        } else {
            "?"
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

/// Find the center of the vorticity maximum (signed) in a single frame.
/// Tracks the most positive vorticity value (counterclockwise vortex center).
/// Returns (lon, lat) in radians with sub-grid parabolic interpolation.
fn find_vort_center(vort: &[f64], im: usize, jm: usize, gauss_nodes: &[f64]) -> (f64, f64) {
    use std::f64::consts::PI;

    // Find grid cell with max vort (signed, not absolute)
    let mut max_val = f64::NEG_INFINITY;
    let mut max_i = 0;
    let mut max_j = 0;
    for j in 0..jm {
        for i in 0..im {
            let v = vort[j * im + i];
            if v > max_val {
                max_val = v;
                max_i = i;
                max_j = j;
            }
        }
    }

    // Parabolic sub-grid interpolation in i (periodic)
    // Use signed values — parabolic fit finds the peak of the signed field
    let i_left = (max_i + im - 1) % im;
    let i_right = (max_i + 1) % im;
    let f_l = vort[max_j * im + i_left];
    let f_c = vort[max_j * im + max_i];
    let f_r = vort[max_j * im + i_right];
    let denom_i = f_l - 2.0 * f_c + f_r;
    let delta_i = if denom_i.abs() > 1e-30 {
        0.5 * (f_l - f_r) / denom_i
    } else {
        0.0
    };
    let i_sub = max_i as f64 + delta_i.clamp(-0.5, 0.5);
    let lon = (i_sub / im as f64 * 2.0 * PI).rem_euclid(2.0 * PI);

    // Parabolic sub-grid interpolation in j (gauss_nodes)
    let mu = if max_j >= 1 && max_j < jm - 1 && !gauss_nodes.is_empty() {
        let f_b = vort[(max_j - 1) * im + max_i];
        let f_a = vort[(max_j + 1) * im + max_i];
        let denom_j = f_b - 2.0 * f_c + f_a;
        let delta_j = if denom_j.abs() > 1e-30 {
            0.5 * (f_b - f_a) / denom_j
        } else {
            0.0
        };
        let j_sub = max_j as f64 + delta_j.clamp(-0.5, 0.5);
        // Linearly interpolate gauss_nodes at fractional index
        let j_lo = (j_sub.floor() as usize).min(jm - 1);
        let j_hi = (j_lo + 1).min(jm - 1);
        let frac = j_sub - j_lo as f64;
        gauss_nodes[j_lo] * (1.0 - frac) + gauss_nodes[j_hi] * frac
    } else if !gauss_nodes.is_empty() {
        gauss_nodes[max_j.min(gauss_nodes.len() - 1)]
    } else {
        0.0
    };

    let lat = mu.clamp(-1.0, 1.0).asin();
    (lon, lat)
}

/// Precompute gyre centers for all frames that have a vort field.
fn precompute_gyre_centers(
    frames: &[SpgFrame],
    vort_index: Option<usize>,
    im: usize,
    jm: usize,
    gauss_nodes: &[f64],
) -> Vec<(f64, f64)> {
    let Some(vi) = vort_index else {
        return Vec::new();
    };
    frames
        .iter()
        .map(|frame| {
            if vi < frame.fields.len() {
                find_vort_center(&frame.fields[vi].1, im, jm, gauss_nodes)
            } else {
                (0.0, 0.0)
            }
        })
        .collect()
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
