mod color;
mod font;
pub mod spherical;

// Re-export public API
pub use color::ColorMap;
pub use font::render_status;
pub(crate) use font::{draw_text, draw_text_sized, FONT_HEIGHT, FONT_WIDTH, STATUS_BAR_HEIGHT};

use color::{BAR_GAP, BAR_TOTAL, BAR_WIDTH, LABEL_GAP, TICK_LEN};
use font::FONT_HEIGHT as FH;
use crate::state::{idx, FrameSnapshot, N};

/// Visualization mode for Karman vortex display.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VizMode {
    /// Temperature (RB) / dye concentration (Karman).
    Field,
    /// Vorticity magnitude.
    Vorticity,
    /// Stream function contour lines.
    Streamline,
    /// Black background -- particles only.
    None,
}

impl VizMode {
    /// Cycle to the next visualization mode.
    pub fn next(self) -> Self {
        match self {
            VizMode::Field => VizMode::Vorticity,
            VizMode::Vorticity => VizMode::Streamline,
            VizMode::Streamline => VizMode::None,
            VizMode::None => VizMode::Field,
        }
    }

    /// Short label for the color bar.
    pub fn label(self) -> &'static str {
        match self {
            VizMode::Field => "temp",
            VizMode::Vorticity => "vort",
            VizMode::Streamline => "vel",
            VizMode::None => "none",
        }
    }
}

/// Dynamic render layout computed from window pixel size.
pub struct RenderConfig {
    pub display_width: usize,
    pub display_height: usize,
    pub frame_width: usize,
    pub frame_height: usize,
    pub tiles: usize,
    pub sim_nx: usize,
    /// Particle dot radius: 0 = single pixel, 1 = 3x3 diamond (default).
    pub particle_radius: u8,
    /// Horizontal pixel offset for centering (pillarboxing). 0 for stretch-to-fit.
    pub display_x_offset: usize,
}

impl RenderConfig {
    /// Compute layout to fit the given pixel dimensions.
    /// `sim_nx` is the simulation grid width (N for RB, wider for Karman).
    /// The simulation always stretches to fill the display area since NX
    /// is calculated to match the window aspect ratio.
    pub fn fit(pixel_width: usize, pixel_height: usize, tiles: usize, sim_nx: usize) -> Self {
        let display_width = pixel_width.saturating_sub(BAR_TOTAL).max(sim_nx);
        let display_height = pixel_height.max(N);

        Self {
            display_width,
            display_height,
            frame_width: display_width + BAR_TOTAL,
            frame_height: display_height + STATUS_BAR_HEIGHT,
            tiles,
            sim_nx,
            particle_radius: 1,
            display_x_offset: 0,
        }
    }

    /// Compute layout for a square bounded domain (e.g. Lid-Driven Cavity).
    /// Preserves aspect ratio with pillarboxing (black bars on sides).
    pub fn fit_square(pixel_width: usize, pixel_height: usize, tiles: usize, sim_nx: usize) -> Self {
        let available_width = pixel_width.saturating_sub(BAR_TOTAL).max(sim_nx);
        let display_height = pixel_height.max(N);

        // Aspect-ratio-preserving: display_width / display_height = (tiles * sim_nx) / N
        let max_width = display_height * tiles * sim_nx / N;
        let display_width = available_width.min(max_width);
        let display_x_offset = (available_width - display_width) / 2;

        Self {
            display_width,
            display_height,
            frame_width: available_width + BAR_TOTAL,
            frame_height: display_height + STATUS_BAR_HEIGHT,
            tiles,
            sim_nx,
            particle_radius: 1,
            display_x_offset,
        }
    }

    /// Fallback config when terminal size cannot be determined.
    #[cfg(test)]
    pub fn default_config() -> Self {
        Self::fit(542, 512, 3, N)
    }

    /// Horizontal scale: display pixels per simulation cell.
    pub fn scale_x(&self) -> f64 {
        self.display_width as f64 / (self.tiles as f64 * self.sim_nx as f64)
    }

    /// Vertical scale: display pixels per simulation cell.
    pub fn scale_y(&self) -> f64 {
        self.display_height as f64 / N as f64
    }
}

/// Compute vorticity field from velocity.
/// Returns (field, positive_max, negative_max) where both max values are >= 0.
fn compute_vorticity(vx: &[f64], vy: &[f64], nx: usize) -> (Vec<f64>, f64, f64) {
    let mut omega = vec![0.0; nx * N];
    let mut pos_max = 0.0_f64;
    let mut neg_max = 0.0_f64;
    for j in 1..(N - 1) as i32 {
        for i in 1..(nx - 1) as i32 {
            // Central difference: dvy/dx - dvx/dy
            let dvydx = (vy[idx(i + 1, j, nx)] - vy[idx(i - 1, j, nx)]) * 0.5;
            let dvxdy = (vx[idx(i, j + 1, nx)] - vx[idx(i, j - 1, nx)]) * 0.5;
            let w = dvydx - dvxdy;
            omega[idx(i, j, nx)] = w;
            if w > 0.0 { pos_max = pos_max.max(w); }
            else { neg_max = neg_max.max(-w); }
        }
    }
    (omega, pos_max, neg_max)
}

/// Compute stream function psi from vx by integrating in y-direction.
/// psi(x, 0) = 0, psi(x, y) = psi(x, y-1) + vx(x, y).
/// Returns (psi_field, psi_min, psi_max).
fn compute_stream_function(vx: &[f64], nx: usize) -> (Vec<f64>, f64, f64) {
    let mut psi = vec![0.0; nx * N];
    let mut psi_min = 0.0_f64;
    let mut psi_max = 0.0_f64;
    for x in 0..nx {
        // psi(x, 0) = 0 (already initialized)
        for y in 1..N {
            let prev = psi[idx(x as i32, (y - 1) as i32, nx)];
            let val = prev + vx[idx(x as i32, y as i32, nx)];
            psi[idx(x as i32, y as i32, nx)] = val;
            psi_min = psi_min.min(val);
            psi_max = psi_max.max(val);
        }
    }
    (psi, psi_min, psi_max)
}

// Meteor trail color palette: warm white -> cyan -> blue -> deep blue
const METEOR_COLORS: [[f64; 3]; 4] = [
    [255.0, 240.0, 200.0], // warm white (newest)
    [180.0, 220.0, 255.0], // light cyan
    [80.0, 140.0, 255.0],  // blue
    [30.0, 50.0, 120.0],   // deep blue (oldest)
];

// Interpolate meteor color from palette. t: 0=oldest, 1=newest.
#[inline]
fn meteor_color(t: f64) -> [f64; 3] {
    let t = t.clamp(0.0, 1.0);
    let seg = t * (METEOR_COLORS.len() - 1) as f64;
    let i = (seg as usize).min(METEOR_COLORS.len() - 2);
    let f = seg - i as f64;
    let c0 = &METEOR_COLORS[METEOR_COLORS.len() - 1 - i]; // reversed: high t = warm
    let c1 = &METEOR_COLORS[(METEOR_COLORS.len() - 2).saturating_sub(i)];
    [
        c0[0] + f * (c1[0] - c0[0]),
        c0[1] + f * (c1[1] - c0[1]),
        c0[2] + f * (c1[2] - c0[2]),
    ]
}

// Additive blend: dst = min(255, dst + src * alpha) -- best for black backgrounds
#[inline]
fn additive_blend(buf: &mut [u8], off: usize, r: f64, g: f64, b: f64, alpha: f64) {
    buf[off] = (buf[off] as f64 + r * alpha).min(255.0) as u8;
    buf[off + 1] = (buf[off + 1] as f64 + g * alpha).min(255.0) as u8;
    buf[off + 2] = (buf[off + 2] as f64 + b * alpha).min(255.0) as u8;
}

// Screen blend: result = 1 - (1-bg)(1-fg*alpha) -- always brightens, good for colored backgrounds
#[inline]
fn screen_blend(buf: &mut [u8], off: usize, r: f64, g: f64, b: f64, alpha: f64) {
    let br = buf[off] as f64;
    let bg = buf[off + 1] as f64;
    let bb = buf[off + 2] as f64;
    let fr = (r * alpha).min(255.0);
    let fg = (g * alpha).min(255.0);
    let fb = (b * alpha).min(255.0);
    buf[off] = (br + fr - br * fr / 255.0).min(255.0) as u8;
    buf[off + 1] = (bg + fg - bg * fg / 255.0).min(255.0) as u8;
    buf[off + 2] = (bb + fb - bb * fb / 255.0).min(255.0) as u8;
}

/// Bresenham line drawing with alpha-blended color.
/// Draws from (x0,y0) to (x1,y1) in screen coordinates, applying x_off to buffer offset.
fn draw_line_blended(
    buf: &mut [u8], frame_width: usize,
    x0: isize, y0: isize, x1: isize, y1: isize,
    dw: usize, dh: usize, x_off: usize,
    color: [f64; 3], alpha: f64,
    blend: fn(&mut [u8], usize, f64, f64, f64, f64),
) {
    let mut cx = x0;
    let mut cy = y0;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: isize = if x0 < x1 { 1 } else { -1 };
    let sy: isize = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if cx >= 0 && (cx as usize) < dw && cy >= 0 && (cy as usize) < dh {
            let off = (cy as usize * frame_width + cx as usize + x_off) * 4;
            blend(buf, off, color[0], color[1], color[2], alpha);
        }
        if cx == x1 && cy == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; cx += sx; }
        if e2 <= dx { err += dx; cy += sy; }
    }
}

/// Render field + color bar into a pre-allocated RGBA buffer.
/// The buffer is resized and zeroed as needed.
pub fn render_into(buf: &mut Vec<u8>, snap: &FrameSnapshot, cfg: &RenderConfig, viz_mode: VizMode, colormap: ColorMap, show_arrows: bool) {
    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let frame_width = cfg.frame_width;
    let frame_height = cfg.frame_height;
    let x_off = cfg.display_x_offset;
    let nx = snap.nx;

    let total = frame_width * frame_height * 4;
    buf.resize(total, 0);
    buf.fill(0);

    // Precompute derived fields if needed
    let (vort_field, vort_pos_max, vort_neg_max) = if viz_mode == VizMode::Vorticity {
        compute_vorticity(&snap.vx, &snap.vy, nx)
    } else {
        (vec![], 0.0, 0.0)
    };
    let (psi_field, psi_min, psi_max) = if viz_mode == VizMode::Streamline {
        compute_stream_function(&snap.vx, nx)
    } else {
        (vec![], 0.0, 0.0)
    };

    // Draw simulation tiled horizontally.
    // NX is calculated to match the window aspect ratio, so always stretch.
    let tiles = cfg.tiles;
    let cylinder = snap.cylinder;
    let sim_total_x = tiles as f64 * nx as f64;
    for screen_y in 0..dh {
        for screen_x in 0..dw {
            let offset = (screen_y * frame_width + screen_x + x_off) * 4;

            // Map screen pixel to simulation coordinates
            let sim_y_raw = screen_y as f64 / dh as f64 * N as f64;
            let sim_y_f = ((N - 1) as f64 - sim_y_raw).clamp(0.0, (N - 1) as f64);
            let sim_x_f = (screen_x as f64 / dw as f64 * sim_total_x).clamp(0.0, sim_total_x - 0.001);

            // Bilinear interpolation
            let sy = sim_y_f.clamp(0.0, (N - 1) as f64);
            let sx = sim_x_f;
            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;
            let x1 = x0 + 1;
            let y1 = (y0 + 1).min((N - 1) as i32);
            let fx = sx - sx.floor();
            let fy = sy - sy.floor();

            let rgba = match viz_mode {
                VizMode::Vorticity => {
                    let w00 = vort_field[idx(x0, y0, nx)];
                    let w10 = vort_field[idx(x1, y0, nx)];
                    let w01 = vort_field[idx(x0, y1, nx)];
                    let w11 = vort_field[idx(x1, y1, nx)];
                    let w = w00 * (1.0 - fx) * (1.0 - fy)
                          + w10 * fx * (1.0 - fy)
                          + w01 * (1.0 - fx) * fy
                          + w11 * fx * fy;
                    // Signed diverging with independent normalization per sign:
                    // blue (negative) ← dark (zero) → red (positive)
                    if w >= 0.0 {
                        let norm = if vort_pos_max > 0.0 { (w / vort_pos_max).min(1.0) } else { 0.0 };
                        let i = norm.powf(0.4);
                        [(i * 220.0) as u8, (i * 40.0) as u8, (i * 30.0) as u8, 255]
                    } else {
                        let norm = if vort_neg_max > 0.0 { (-w / vort_neg_max).min(1.0) } else { 0.0 };
                        let i = norm.powf(0.4);
                        [(i * 30.0) as u8, (i * 60.0) as u8, (i * 220.0) as u8, 255]
                    }
                }
                VizMode::Streamline => {
                    // Bilinear interpolate psi
                    let p00 = psi_field[idx(x0, y0, nx)];
                    let p10 = psi_field[idx(x1, y0, nx)];
                    let p01 = psi_field[idx(x0, y1, nx)];
                    let p11 = psi_field[idx(x1, y1, nx)];
                    let psi = p00 * (1.0 - fx) * (1.0 - fy)
                            + p10 * fx * (1.0 - fy)
                            + p01 * (1.0 - fx) * fy
                            + p11 * fx * fy;

                    // Velocity magnitude for background shading
                    let vx00 = snap.vx[idx(x0, y0, nx)];
                    let vx10 = snap.vx[idx(x1, y0, nx)];
                    let vx01 = snap.vx[idx(x0, y1, nx)];
                    let vx11 = snap.vx[idx(x1, y1, nx)];
                    let vx_i = vx00 * (1.0 - fx) * (1.0 - fy)
                             + vx10 * fx * (1.0 - fy)
                             + vx01 * (1.0 - fx) * fy
                             + vx11 * fx * fy;
                    let vy00 = snap.vy[idx(x0, y0, nx)];
                    let vy10 = snap.vy[idx(x1, y0, nx)];
                    let vy01 = snap.vy[idx(x0, y1, nx)];
                    let vy11 = snap.vy[idx(x1, y1, nx)];
                    let vy_i = vy00 * (1.0 - fx) * (1.0 - fy)
                             + vy10 * fx * (1.0 - fy)
                             + vy01 * (1.0 - fx) * fy
                             + vy11 * fx * fy;
                    let vel = (vx_i * vx_i + vy_i * vy_i).sqrt();

                    // Contour lines from psi
                    let psi_range = psi_max - psi_min;
                    if psi_range > 1e-12 {
                        let num_lines = 24.0;
                        let interval = psi_range / num_lines;
                        let frac = ((psi - psi_min) / interval) % 1.0;
                        let line_dist = (frac.min(1.0 - frac)) * 2.0; // 0=on line, 1=between
                        let threshold = 0.15;
                        if line_dist < threshold {
                            // Contour line: bright cyan, brighter when closer to line
                            let brightness = 1.0 - (line_dist / threshold);
                            let r = (100.0 + brightness * 80.0) as u8;
                            let g = (180.0 + brightness * 60.0) as u8;
                            let b = (210.0 + brightness * 45.0) as u8;
                            [r, g, b, 255]
                        } else {
                            // Background: dark navy shaded by velocity magnitude
                            let v = (vel * 5.0).min(1.0);
                            let r = (10.0 + v * 20.0) as u8;
                            let g = (12.0 + v * 40.0) as u8;
                            let b = (30.0 + v * 60.0) as u8;
                            [r, g, b, 255]
                        }
                    } else {
                        // No flow: uniform dark
                        [10, 12, 30, 255]
                    }
                }
                VizMode::Field => {
                    let t00 = snap.temperature[idx(x0, y0, nx)];
                    let t10 = snap.temperature[idx(x1, y0, nx)];
                    let t01 = snap.temperature[idx(x0, y1, nx)];
                    let t11 = snap.temperature[idx(x1, y1, nx)];
                    let t = t00 * (1.0 - fx) * (1.0 - fy)
                        + t10 * fx * (1.0 - fy)
                        + t01 * (1.0 - fx) * fy
                        + t11 * fx * fy;
                    color::map_to_rgba(t, colormap)
                }
                VizMode::None => [0, 0, 0, 255],
            };

            // Smooth cylinder rendering using distance-based anti-aliasing
            if let Some((cx, cy, radius)) = cylinder {
                let tile_sim_x = sim_x_f % nx as f64;
                let dx = tile_sim_x - cx;
                let dy = sim_y_f - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < radius - 0.5 {
                    // Fully inside cylinder
                    buf[offset] = 0x33;
                    buf[offset + 1] = 0x33;
                    buf[offset + 2] = 0x33;
                    buf[offset + 3] = 255;
                    continue;
                } else if dist < radius + 0.5 {
                    // Anti-aliased edge: blend between cylinder gray and field color
                    let alpha = (dist - (radius - 0.5)).clamp(0.0, 1.0);
                    buf[offset] = (0x33 as f64 * (1.0 - alpha) + rgba[0] as f64 * alpha) as u8;
                    buf[offset + 1] = (0x33 as f64 * (1.0 - alpha) + rgba[1] as f64 * alpha) as u8;
                    buf[offset + 2] = (0x33 as f64 * (1.0 - alpha) + rgba[2] as f64 * alpha) as u8;
                    buf[offset + 3] = 255;
                    continue;
                }
            }

            buf[offset] = rgba[0];
            buf[offset + 1] = rgba[1];
            buf[offset + 2] = rgba[2];
            buf[offset + 3] = rgba[3];
        }
    }

    // Draw color bar (1.0 at top, 0.0 at bottom -- matching tick labels)
    let bar_x = x_off + dw + BAR_GAP;
    for y in 0..dh {
        let t = 1.0 - y as f64 / (dh - 1) as f64;
        let rgba = match viz_mode {
            VizMode::Vorticity => {
                // Diverging bar: blue (-1, bottom) → dark (0, middle) → red (+1, top)
                let signed = t * 2.0 - 1.0;
                let intensity = signed.abs().powf(0.4);
                if signed >= 0.0 {
                    [(intensity * 220.0) as u8, (intensity * 40.0) as u8, (intensity * 30.0) as u8, 255]
                } else {
                    [(intensity * 30.0) as u8, (intensity * 60.0) as u8, (intensity * 220.0) as u8, 255]
                }
            }
            VizMode::Streamline => {
                // Velocity magnitude gradient: dark navy -> bright teal
                let r = (10.0 + t * 20.0) as u8;
                let g = (12.0 + t * 40.0) as u8;
                let b = (30.0 + t * 60.0) as u8;
                [r, g, b, 255]
            }
            VizMode::Field => color::map_to_rgba(t, colormap),
            VizMode::None => [0, 0, 0, 255],
        };
        for bx in 0..BAR_WIDTH {
            let offset = (y * frame_width + bar_x + bx) * 4;
            buf[offset] = rgba[0];
            buf[offset + 1] = rgba[1];
            buf[offset + 2] = rgba[2];
            buf[offset + 3] = rgba[3];
        }
    }

    // Draw tick marks and value labels at 0%, 25%, 50%, 75%, 100%
    let tick_x = bar_x + BAR_WIDTH;
    let label_x = tick_x + TICK_LEN + LABEL_GAP;
    let label_color: [u8; 3] = [0x88, 0x88, 0x88];
    let tick_labels = ["1.0", "0.7", "0.5", "0.2", "0.0"];
    for (tick, label) in tick_labels.iter().enumerate() {
        let y = (tick as usize) * (dh - 1) / 4;
        // Tick marks
        for dy in 0..2usize {
            let yy = (y + dy).min(dh - 1);
            for tx in 0..TICK_LEN {
                let offset = (yy * frame_width + tick_x + tx) * 4;
                buf[offset] = 255;
                buf[offset + 1] = 255;
                buf[offset + 2] = 255;
                buf[offset + 3] = 255;
            }
        }
        // Value label (vertically centered on tick)
        let label_y = if y >= FH / 2 { y - FH / 2 } else { 0 };
        let label_y = label_y.min(dh.saturating_sub(FH));
        font::draw_text(buf, frame_width, label_x, label_y, label, label_color);
    }

    // Draw type label above the bar
    let type_label = viz_mode.label();
    let type_label_y = if dh > FH + 4 { 2 } else { 0 };
    // Right-align within bar area
    let type_label_w = type_label.len() * (FONT_WIDTH + 1);
    let type_label_x = if bar_x + BAR_WIDTH / 2 >= type_label_w / 2 {
        bar_x + BAR_WIDTH / 2 - type_label_w / 2
    } else {
        bar_x
    };
    font::draw_text(buf, frame_width, type_label_x, type_label_y, type_label, [0xAA, 0xAA, 0xAA]);

    // --- Arrow overlay (quiver plot) ---
    if show_arrows {
        let sx = cfg.scale_x();
        let sy = cfg.scale_y();
        let stride = 6usize; // sample every 6 sim cells
        let arrow_color = [160.0, 220.0, 255.0]; // light cyan
        let arrow_alpha = 0.7;

        // Find max velocity magnitude for normalization
        let mut vmax_sq = 0.0_f64;
        for j in (stride / 2..N).step_by(stride) {
            for i in (stride / 2..nx).step_by(stride) {
                let ii = idx(i as i32, j as i32, nx);
                let vx_val = snap.vx[ii];
                let vy_val = snap.vy[ii];
                vmax_sq = vmax_sq.max(vx_val * vx_val + vy_val * vy_val);
            }
        }
        let vmax = vmax_sq.sqrt();

        if vmax > 1e-12 {
            let max_arrow_px = stride as f64 * sx.min(sy) * 0.8;

            for j in (stride / 2..N).step_by(stride) {
                for i in (stride / 2..nx).step_by(stride) {
                    let ii = idx(i as i32, j as i32, nx);
                    let vx_val = snap.vx[ii];
                    let vy_val = snap.vy[ii];
                    let mag = (vx_val * vx_val + vy_val * vy_val).sqrt();

                    // Skip near-zero velocity
                    if mag < vmax * 0.01 { continue; }

                    let scale = (mag / vmax) * max_arrow_px;
                    let dx = vx_val / mag * scale;
                    // Y is flipped for screen coords
                    let dy = -vy_val / mag * scale;

                    for tile in 0..tiles {
                        let base_x = (i as f64 + tile as f64 * nx as f64) * sx;
                        let base_y = ((N - 1) as f64 - j as f64) * sy;

                        let x0 = (base_x - dx * 0.5) as isize;
                        let y0 = (base_y - dy * 0.5) as isize;
                        let x1 = (base_x + dx * 0.5) as isize;
                        let y1 = (base_y + dy * 0.5) as isize;

                        // Draw shaft
                        draw_line_blended(buf, frame_width, x0, y0, x1, y1, dw, dh, x_off, arrow_color, arrow_alpha, screen_blend);

                        // Draw arrowhead (two lines from tip at ±150° from direction)
                        if scale > 2.0 {
                            let head_len = 3.0_f64.min(scale * 0.35);
                            let dir_x = dx / scale;
                            let dir_y = dy / scale;
                            // ±150° rotation (cos150 ≈ -0.866, sin150 ≈ 0.5)
                            let cos_a = -0.866_f64;
                            let sin_a = 0.5_f64;
                            for &sign in &[1.0_f64, -1.0] {
                                let hx = (dir_x * cos_a - dir_y * sin_a * sign) * head_len;
                                let hy = (dir_x * sin_a * sign + dir_y * cos_a) * head_len;
                                let hx1 = (x1 as f64 + hx) as isize;
                                let hy1 = (y1 as f64 + hy) as isize;
                                draw_line_blended(buf, frame_width, x1, y1, hx1, hy1, dw, dh, x_off, arrow_color, arrow_alpha, screen_blend);
                            }
                        }
                    }
                }
            }
        }
    }

    // Draw particles as glowing trails with meteor color palette.
    let sx = cfg.scale_x();
    let sy = cfg.scale_y();
    let tile_width_px = (dw as isize) / tiles as isize;

    let use_small_particles = cfg.particle_radius == 0;
    let is_meteor = viz_mode == VizMode::None;

    // Choose blend function based on mode
    let blend: fn(&mut [u8], usize, f64, f64, f64, f64) = if is_meteor {
        additive_blend
    } else {
        screen_blend
    };

    let trail_count = snap.trail_xs.len();
    if trail_count > 1 {
        for trail_idx in 0..(trail_count - 1) {
            let frac = (trail_idx + 1) as f64 / trail_count as f64;
            let alpha = frac.powf(2.5);
            let color = meteor_color(frac);

            // Sparkle: pseudo-random brightness boost per-particle
            let txs = &snap.trail_xs[trail_idx];
            let tys = &snap.trail_ys[trail_idx];

            for p in 0..txs.len() {
                let base_cx = (txs[p] * sx) as isize;
                let cy = (((N - 1) as f64 - tys[p]) * sy) as isize;

                let sparkle_hash = (p * 7 + trail_idx * 13) % 17;
                let sparkle_mult = if sparkle_hash < 3 { 1.5 } else { 1.0 };

                // Tapering: newest=diamond, oldest=dot
                let use_dot = frac < 0.4;

                for tile in 0..tiles {
                    let cx = base_cx + tile_width_px * tile as isize;

                    if use_dot || use_small_particles {
                        let px = cx as usize;
                        let py = cy as usize;
                        if px < dw && py < dh {
                            let off = (py * frame_width + px + x_off) * 4;
                            let a = alpha * sparkle_mult;
                            blend(buf, off, color[0], color[1], color[2], a);
                        }
                    } else {
                        let diamond: &[(isize, isize, f64)] = &[
                            (0, -1, 0.5), (-1, 0, 0.5),
                            (0, 0, 1.0),
                            (1, 0, 0.5), (0, 1, 0.5),
                        ];
                        for &(dx, dy, weight) in diamond {
                            let px = (cx + dx) as usize;
                            let py = (cy + dy) as usize;
                            if px < dw && py < dh {
                                let off = (py * frame_width + px + x_off) * 4;
                                let a = alpha * weight * sparkle_mult;
                                blend(buf, off, color[0], color[1], color[2], a);
                            }
                        }
                    }
                }
            }
        }
    }

    // Draw particle heads with 5x5 soft glow
    let head_color = METEOR_COLORS[0]; // warm white
    const GLOW: &[(isize, isize, f64)] = &[
                                    (0, -2, 0.15),
                      (-1, -1, 0.25), (0, -1, 0.6), (1, -1, 0.25),
            (-2, 0, 0.15), (-1, 0, 0.6), (0, 0, 1.0), (1, 0, 0.6), (2, 0, 0.15),
                      (-1, 1, 0.25), (0, 1, 0.6), (1, 1, 0.25),
                                    (0, 2, 0.15),
    ];

    for i in 0..snap.particles_x.len() {
        let sim_x = snap.particles_x[i];
        let sim_y = snap.particles_y[i];
        let base_cx = (sim_x * sx) as isize;
        let cy = (((N - 1) as f64 - sim_y) * sy) as isize;

        for t in 0..tiles {
            let copy_offset = tile_width_px * t as isize;
            let cx = base_cx + copy_offset;

            if use_small_particles {
                let px = cx as usize;
                let py = cy as usize;
                if px < dw && py < dh {
                    let off = (py * frame_width + px + x_off) * 4;
                    blend(buf, off, head_color[0], head_color[1], head_color[2], 1.0);
                }
            } else {
                for &(dx, dy, weight) in GLOW {
                    let px = (cx + dx) as usize;
                    let py = (cy + dy) as usize;
                    if px < dw && py < dh {
                        let off = (py * frame_width + px + x_off) * 4;
                        blend(buf, off, head_color[0], head_color[1], head_color[2], weight);
                    }
                }
            }
        }
    }

}

/// Render field + color bar to a new RGBA buffer (test convenience wrapper).
#[cfg(any(test, debug_assertions))]
pub fn render(snap: &FrameSnapshot, cfg: &RenderConfig, viz_mode: VizMode, colormap: ColorMap) -> Vec<u8> {
    let mut buf = Vec::new();
    render_into(&mut buf, snap, cfg, viz_mode, colormap, false);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::color::LABEL_WIDTH;
    use crate::state::SimState;

    fn test_config() -> RenderConfig {
        RenderConfig::fit(542, 512, 3, N)
    }

    #[test]
    fn test_render_config_fit() {
        let cfg = RenderConfig::fit(1200, 800, 3, N);
        assert_eq!(cfg.display_width, 1200 - BAR_TOTAL);
        assert_eq!(cfg.display_height, 800);
        assert_eq!(cfg.frame_width, cfg.display_width + BAR_TOTAL);
        assert_eq!(cfg.frame_height, 800 + STATUS_BAR_HEIGHT);
        assert_eq!(cfg.tiles, 3);
        assert_eq!(cfg.sim_nx, N);
    }

    #[test]
    fn test_render_config_small_terminal() {
        let cfg = RenderConfig::fit(200, 50, 3, N);
        assert_eq!(cfg.display_width, 200 - BAR_TOTAL); // 200 - 56 = 144
        assert_eq!(cfg.display_height, N); // clamped to minimum N (50 < 80)
    }

    #[test]
    fn test_render_buffer_size() {
        let snap = SimState::new(400, 0.15, N).snapshot();
        let cfg = test_config();
        let buf = render(&snap, &cfg, VizMode::Field, ColorMap::TokyoNight);
        assert_eq!(buf.len(), cfg.frame_width * cfg.frame_height * 4);
    }

    #[test]
    fn test_render_y_flip() {
        let snap = SimState::new(400, 0.15, N).snapshot();
        let cfg = test_config();
        let buf = render(&snap, &cfg, VizMode::Field, ColorMap::TokyoNight);

        // Sample top and bottom of display area
        let x = cfg.display_width / 2;
        let top_off = (0 * cfg.frame_width + x) * 4;
        let bot_off = ((cfg.display_height - 1) * cfg.frame_width + x) * 4;

        let top_brightness = buf[top_off] as u32 + buf[top_off + 1] as u32 + buf[top_off + 2] as u32;
        let bot_brightness = buf[bot_off] as u32 + buf[bot_off + 1] as u32 + buf[bot_off + 2] as u32;
        assert!(bot_brightness > top_brightness, "Bottom (hot) should be brighter than top (cold)");
    }

    #[test]
    fn test_particles_rendered_diamond_adaptive() {
        let mut snap = SimState::new(400, 0.15, N).snapshot();
        snap.particles_x.clear();
        snap.particles_y.clear();
        let mid = (N / 2) as f64;
        snap.particles_x.push(mid);
        snap.particles_y.push(mid);

        let cfg = test_config();
        let buf = render(&snap, &cfg, VizMode::Field, ColorMap::TokyoNight);

        let cx = (mid * cfg.scale_x()) as usize;
        let cy = (((N - 1) as f64 - mid) * cfg.scale_y()) as usize;
        let center = (cy * cfg.frame_width + cx) * 4;
        let bg_offset = ((cy - 2) * cfg.frame_width + cx) * 4;
        let core_lum = buf[center] as i32 + buf[center + 1] as i32 + buf[center + 2] as i32;
        let bg_lum = buf[bg_offset] as i32 + buf[bg_offset + 1] as i32 + buf[bg_offset + 2] as i32;
        let contrast = (core_lum - bg_lum).abs();
        assert!(contrast > 50, "Particle core should contrast with background: core_lum={}, bg_lum={}", core_lum, bg_lum);
    }

    #[test]
    fn test_particles_glow_brighter_than_background() {
        // Screen blend should make particles BRIGHTER than background, never darker
        let mut snap = SimState::new(400, 0.15, N).snapshot();
        snap.particles_x.clear();
        snap.particles_y.clear();
        let mid = (N / 2) as f64;
        snap.particles_x.push(mid);
        snap.particles_y.push(mid);

        let cfg = test_config();

        // Render with particle
        let buf_with = render(&snap, &cfg, VizMode::Field, ColorMap::TokyoNight);
        // Render without particle
        snap.particles_x.clear();
        snap.particles_y.clear();
        let buf_without = render(&snap, &cfg, VizMode::Field, ColorMap::TokyoNight);

        let cx = (mid * cfg.scale_x()) as usize;
        let cy = (((N - 1) as f64 - mid) * cfg.scale_y()) as usize;
        let center = (cy * cfg.frame_width + cx) * 4;
        let with_lum = buf_with[center] as i32 + buf_with[center + 1] as i32 + buf_with[center + 2] as i32;
        let without_lum = buf_without[center] as i32 + buf_without[center + 1] as i32 + buf_without[center + 2] as i32;
        assert!(with_lum >= without_lum,
            "Particle should brighten (screen blend), not darken: with={with_lum}, without={without_lum}");
    }

    #[test]
    fn test_color_bar_gradient() {
        let snap = SimState::new(400, 0.15, N).snapshot();
        let cfg = test_config();
        let buf = render(&snap, &cfg, VizMode::Field, ColorMap::TokyoNight);

        let bar_x = cfg.display_width + BAR_GAP + BAR_WIDTH / 2;
        let top_offset = bar_x * 4;
        let bot_offset = ((cfg.display_height - 1) * cfg.frame_width + bar_x) * 4;
        let top_bright = buf[top_offset] as u32 + buf[top_offset + 1] as u32 + buf[top_offset + 2] as u32;
        let bot_bright = buf[bot_offset] as u32 + buf[bot_offset + 1] as u32 + buf[bot_offset + 2] as u32;
        assert!(top_bright > bot_bright, "Bar top (1.0/hot) should be brighter than bottom (0.0/cold)");
    }

    #[test]
    fn test_render_cylinder_gray() {
        let mut snap = SimState::new(0, 0.15, N).snapshot();
        // Add a cylinder at center
        snap.cylinder = Some((64.0, 64.0, 20.0));
        snap.temperature.fill(0.5); // uniform temp

        let cfg = test_config();
        let buf = render(&snap, &cfg, VizMode::Field, ColorMap::TokyoNight);

        // Cylinder center in sim: (64, 64)
        let cx_screen = (64.0 * cfg.scale_x()) as usize;
        let cy_screen = (((N - 1) as f64 - 64.0) * cfg.scale_y()) as usize;
        let offset = (cy_screen * cfg.frame_width + cx_screen) * 4;
        assert_eq!(buf[offset], 0x33, "Cylinder pixel R should be 0x33");
        assert_eq!(buf[offset + 1], 0x33, "Cylinder pixel G should be 0x33");
        assert_eq!(buf[offset + 2], 0x33, "Cylinder pixel B should be 0x33");
    }

    #[test]
    fn test_render_no_cylinder_unchanged() {
        let snap = SimState::new(0, 0.15, N).snapshot();
        assert!(snap.cylinder.is_none());
        let cfg = test_config();
        let buf = render(&snap, &cfg, VizMode::Field, ColorMap::TokyoNight);
        // Just check it doesn't crash and has correct size
        assert_eq!(buf.len(), cfg.frame_width * cfg.frame_height * 4);
    }

    #[test]
    fn test_render_nonsquare() {
        // Test with nx=256 (wider than N=128)
        let nx = 256;
        let snap = SimState::new(0, 0.15, nx).snapshot();
        assert_eq!(snap.nx, nx);
        let cfg = RenderConfig::fit(800, 512, 1, nx);
        let buf = render(&snap, &cfg, VizMode::Field, ColorMap::TokyoNight);
        assert_eq!(buf.len(), cfg.frame_width * cfg.frame_height * 4);
    }

    #[test]
    fn test_color_bar_labels() {
        let snap = SimState::new(0, 0.15, N).snapshot();
        let cfg = test_config();
        let buf = render(&snap, &cfg, VizMode::Field, ColorMap::TokyoNight);

        // Check that the label area (right of ticks) has some non-zero pixels
        let label_x = cfg.display_width + BAR_GAP + BAR_WIDTH + TICK_LEN + LABEL_GAP;
        let mut found_label = false;
        for y in 0..cfg.display_height {
            for x in label_x..label_x + LABEL_WIDTH {
                let off = (y * cfg.frame_width + x) * 4;
                if off + 3 < buf.len() && buf[off] != 0 {
                    found_label = true;
                    break;
                }
            }
            if found_label { break; }
        }
        assert!(found_label, "Color bar should have value labels");
    }

    #[test]
    fn test_trail_renders_faded_dots() {
        let mut state = SimState::new(1, 0.15, N);
        // Place particle at center and take several snapshots to build trail
        let mid = (N / 2) as f64;
        for i in 0..4 {
            state.particles_x[0] = mid + i as f64 * 2.0;
            state.particles_y[0] = mid;
            let _ = state.snapshot();
        }
        // Final position
        state.particles_x[0] = mid + 8.0;
        state.particles_y[0] = mid;
        let snap = state.snapshot();
        assert!(snap.trail_xs.len() >= 4, "Should have trail history");

        let cfg = test_config();
        let buf_with_trail = render(&snap, &cfg, VizMode::Field, ColorMap::TokyoNight);

        // Compare against render without trail
        let mut no_trail_snap = snap;
        no_trail_snap.trail_xs.clear();
        no_trail_snap.trail_ys.clear();
        let buf_no_trail = render(&no_trail_snap, &cfg, VizMode::Field, ColorMap::TokyoNight);

        // The two buffers should differ (trail pixels changed some values)
        let diffs: usize = buf_with_trail.iter().zip(buf_no_trail.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(diffs > 0, "Trail rendering should modify some pixels");
    }

    #[test]
    fn test_viz_mode_cycle() {
        assert_eq!(VizMode::Field.next(), VizMode::Vorticity);
        assert_eq!(VizMode::Vorticity.next(), VizMode::Streamline);
        assert_eq!(VizMode::Streamline.next(), VizMode::None);
        assert_eq!(VizMode::None.next(), VizMode::Field);
    }

    #[test]
    fn test_viz_mode_label() {
        assert_eq!(VizMode::Field.label(), "temp");
        assert_eq!(VizMode::Vorticity.label(), "vort");
        assert_eq!(VizMode::Streamline.label(), "vel");
        assert_eq!(VizMode::None.label(), "none");
    }

    #[test]
    fn test_viz_mode_none_renders_black_background() {
        let snap = SimState::new(400, 0.15, N).snapshot();
        let cfg = test_config();
        let buf = render(&snap, &cfg, VizMode::None, ColorMap::TokyoNight);
        // Sample a pixel in the middle of the field area (not on a particle)
        // Background should be black [0, 0, 0, 255]
        // Check multiple pixels; at least most should be black
        let dw = cfg.display_width;
        let dh = cfg.display_height;
        let mid_y = dh / 2;
        let mid_x = dw / 4; // avoid particles in center
        let offset = (mid_y * cfg.frame_width + mid_x) * 4;
        // The pixel should be very dark (black or near-black background)
        let r = buf[offset];
        let g = buf[offset + 1];
        let b = buf[offset + 2];
        assert!(r <= 10 && g <= 10 && b <= 10,
            "VizMode::None background should be black, got ({r}, {g}, {b})");
    }

    #[test]
    fn test_compute_stream_function_uniform() {
        // Uniform vx=1.0 -> psi increases linearly in y
        let nx = 10;
        let vx = vec![1.0; nx * N];
        let (psi, psi_min, psi_max) = compute_stream_function(&vx, nx);
        assert_eq!(psi.len(), nx * N);
        // psi(x, 0) = 0, psi(x, 1) = 1, psi(x, 2) = 2, ...
        for x in 1..(nx - 1) {
            let p0 = psi[idx(x as i32, 0, nx)];
            let p1 = psi[idx(x as i32, 1, nx)];
            let p2 = psi[idx(x as i32, 2, nx)];
            assert!((p0 - 0.0).abs() < 1e-10, "psi(x, 0) should be 0");
            assert!(p1 > p0, "psi should increase with y");
            assert!(p2 > p1, "psi should increase with y");
        }
        assert!(psi_max > psi_min);
    }

    #[test]
    fn test_compute_stream_function_zero() {
        let nx = 10;
        let vx = vec![0.0; nx * N];
        let (psi, psi_min, psi_max) = compute_stream_function(&vx, nx);
        for &v in &psi {
            assert!((v - 0.0).abs() < 1e-10, "Zero vx -> psi=0 everywhere");
        }
        assert!((psi_min - 0.0).abs() < 1e-10);
        assert!((psi_max - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_render_streamline_buffer_size() {
        let nx = 256;
        let mut snap = SimState::new(0, 0.15, nx).snapshot();
        // Give it some velocity for stream function
        for v in snap.vx.iter_mut() { *v = 0.5; }
        let cfg = RenderConfig::fit(800, 512, 1, nx);
        let buf = render(&snap, &cfg, VizMode::Streamline, ColorMap::TokyoNight);
        assert_eq!(buf.len(), cfg.frame_width * cfg.frame_height * 4);
    }

    #[test]
    fn test_viz_mode_none_meteor_trail_brighter_near_head() {
        // Newest trail points should be brighter than oldest ones,
        // AND oldest trail should have blue tint (meteor color gradient)
        let mut state = SimState::new(1, 0.15, N);
        let mid = (N / 2) as f64;
        // Build trail by moving particle across several snapshots
        for i in 0..6 {
            state.particles_x[0] = mid + i as f64 * 3.0;
            state.particles_y[0] = mid;
            let _ = state.snapshot();
        }
        state.particles_x[0] = mid + 18.0;
        state.particles_y[0] = mid;
        let snap = state.snapshot();
        assert!(snap.trail_xs.len() >= 5, "Should have enough trail history");

        let cfg = test_config();
        let buf = render(&snap, &cfg, VizMode::None, ColorMap::TokyoNight);

        let sx = cfg.scale_x();
        let sy = cfg.scale_y();

        // Oldest trail (first entry) -- should have blue tint (B > R)
        let oldest_x = (snap.trail_xs[0][0] * sx) as usize;
        let oldest_y = (((N - 1) as f64 - snap.trail_ys[0][0]) * sy) as usize;
        if oldest_x < cfg.display_width && oldest_y < cfg.display_height {
            let off = (oldest_y * cfg.frame_width + oldest_x) * 4;
            let r = buf[off] as f64;
            let b = buf[off + 2] as f64;
            // Meteor gradient: oldest should trend bluer (B >= R)
            assert!(b >= r,
                "Oldest trail should have blue tint (meteor gradient), got R={r}, B={b}");
        }

        // Newest trail -- should be warm (R >= B)
        let newest_idx = snap.trail_xs.len() - 1;
        let newest_x = (snap.trail_xs[newest_idx][0] * sx) as usize;
        let newest_y = (((N - 1) as f64 - snap.trail_ys[newest_idx][0]) * sy) as usize;

        // Sample 3x3 area luminance around each point
        let sample_lum = |cx: usize, cy: usize| -> f64 {
            let mut total = 0.0;
            let mut count = 0;
            for dy in 0..3_usize {
                for dx in 0..3_usize {
                    let px = cx.wrapping_add(dx).wrapping_sub(1);
                    let py = cy.wrapping_add(dy).wrapping_sub(1);
                    if px < cfg.display_width && py < cfg.display_height {
                        let off = (py * cfg.frame_width + px) * 4;
                        total += buf[off] as f64 * 0.3
                            + buf[off + 1] as f64 * 0.59
                            + buf[off + 2] as f64 * 0.11;
                        count += 1;
                    }
                }
            }
            if count > 0 { total / count as f64 } else { 0.0 }
        };

        let newest_lum = sample_lum(newest_x, newest_y);
        let oldest_lum = sample_lum(oldest_x, oldest_y);
        assert!(newest_lum > oldest_lum + 5.0,
            "Newest trail should be brighter than oldest: newest={newest_lum:.1}, oldest={oldest_lum:.1}");
    }

    #[test]
    fn test_draw_arrow_pixels() {
        // Arrow overlay should draw non-zero pixels on a black buffer
        let mut state = SimState::new(0, 0.15, N);
        // Set uniform velocity so arrows appear
        for v in state.vx.iter_mut() { *v = 1.0; }
        for v in state.vy.iter_mut() { *v = 0.5; }
        let snap = state.snapshot();
        let cfg = test_config();
        let mut buf = Vec::new();
        render_into(&mut buf, &snap, &cfg, VizMode::None, ColorMap::TokyoNight, true);

        // Count non-black pixels in display area (excluding status bar / color bar)
        let dw = cfg.display_width;
        let dh = cfg.display_height;
        let mut lit = 0;
        for y in 0..dh {
            for x in 0..dw {
                let off = (y * cfg.frame_width + x) * 4;
                if buf[off] > 0 || buf[off + 1] > 0 || buf[off + 2] > 0 {
                    lit += 1;
                }
            }
        }
        assert!(lit > 10, "Arrow overlay should draw visible pixels, got {lit}");
    }

    #[test]
    fn test_arrows_overlay_modifies_buffer() {
        // Rendering with show_arrows=true should differ from show_arrows=false
        let mut state = SimState::new(0, 0.15, N);
        for v in state.vx.iter_mut() { *v = 1.0; }
        for v in state.vy.iter_mut() { *v = 0.5; }
        let snap = state.snapshot();
        let cfg = test_config();

        let mut buf_off = Vec::new();
        render_into(&mut buf_off, &snap, &cfg, VizMode::Field, ColorMap::TokyoNight, false);
        let mut buf_on = Vec::new();
        render_into(&mut buf_on, &snap, &cfg, VizMode::Field, ColorMap::TokyoNight, true);

        let diffs: usize = buf_off.iter().zip(buf_on.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(diffs > 0, "Arrow overlay should modify buffer pixels");
    }

    #[test]
    fn test_viz_mode_none_additive_glow() {
        // Head particle should have glow extending beyond the 3x3 diamond
        // (5x5 soft glow in meteor mode)
        let mut state = SimState::new(1, 0.15, N);
        let mid = (N / 2) as f64;
        // Build a short trail
        for i in 0..3 {
            state.particles_x[0] = mid + i as f64 * 2.0;
            state.particles_y[0] = mid;
            let _ = state.snapshot();
        }
        state.particles_x[0] = mid + 6.0;
        state.particles_y[0] = mid;
        let snap = state.snapshot();

        let cfg = test_config();
        let buf = render(&snap, &cfg, VizMode::None, ColorMap::TokyoNight);

        // Find the head position on screen
        let sx = cfg.scale_x();
        let sy = cfg.scale_y();
        let head_x = (snap.particles_x[0] * sx) as isize;
        let head_y = (((N - 1) as f64 - snap.particles_y[0]) * sy) as isize;

        // Check pixels at distance=2 from head (beyond the 3x3 diamond)
        // These should be lit by the 5x5 soft glow
        let outer_offsets: [(isize, isize); 4] = [(-2, 0), (2, 0), (0, -2), (0, 2)];
        let mut outer_lit = 0;
        for (dx, dy) in outer_offsets {
            let px = (head_x + dx) as usize;
            let py = (head_y + dy) as usize;
            if px < cfg.display_width && py < cfg.display_height {
                let off = (py * cfg.frame_width + px) * 4;
                let lum = buf[off] as f64 * 0.3
                    + buf[off + 1] as f64 * 0.59
                    + buf[off + 2] as f64 * 0.11;
                if lum > 5.0 {
                    outer_lit += 1;
                }
            }
        }
        assert!(outer_lit >= 2,
            "Head glow should extend beyond 3x3 diamond (5x5 soft glow), got {outer_lit}/4 outer pixels lit");
    }
}
