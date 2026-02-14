use crate::state::{idx, FrameSnapshot, N};

/// Tokyo Night–inspired color stops for field mapping.
/// Deep navy -> blue -> purple -> pink -> orange
const COLOR_STOPS: [(f64, f64, f64); 5] = [
    (26.0, 27.0, 38.0),    // #1a1b26 navy         (0.00)
    (122.0, 162.0, 247.0), // #7aa2f7 blue         (0.25)
    (187.0, 154.0, 247.0), // #bb9af7 purple       (0.50)
    (247.0, 118.0, 142.0), // #f7768e pink         (0.75)
    (255.0, 158.0, 100.0), // #ff9e64 orange       (1.00)
];

/// Convert temperature [0.0, 1.0] to RGBA color (Tokyo Night palette).
pub fn temperature_to_rgba(t: f64) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let seg = t * 4.0;
    let i = (seg as usize).min(3);
    let s = seg - i as f64;

    let (r0, g0, b0) = COLOR_STOPS[i];
    let (r1, g1, b1) = COLOR_STOPS[i + 1];

    [
        (r0 + s * (r1 - r0)) as u8,
        (g0 + s * (g1 - g0)) as u8,
        (b0 + s * (b1 - b0)) as u8,
        255,
    ]
}

/// Color bar layout constants.
const BAR_GAP: usize = 6;
const BAR_WIDTH: usize = 20;
const TICK_LEN: usize = 4;
const LABEL_GAP: usize = 2;
const LABEL_WIDTH: usize = 24;
const BAR_TOTAL: usize = BAR_GAP + BAR_WIDTH + TICK_LEN + LABEL_GAP + LABEL_WIDTH;

/// Status bar layout constants.
pub(crate) const FONT_WIDTH: usize = 5;
pub(crate) const FONT_HEIGHT: usize = 7;
const STATUS_PAD_TOP: usize = 3;
const STATUS_PAD_BOTTOM: usize = 2;
pub(crate) const STATUS_BAR_HEIGHT: usize = STATUS_PAD_TOP + FONT_HEIGHT + STATUS_PAD_BOTTOM;

/// 5x7 bitmap font glyph lookup. Each row is a u8 with lower 5 bits = pixels (bit4=left).
const fn glyph(ch: u8) -> [u8; FONT_HEIGHT] {
    match ch {
        b' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        b'.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00],
        b'-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        b'>' => [0x10, 0x08, 0x04, 0x02, 0x04, 0x08, 0x10],
        b'=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        b'|' => [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        b'0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        b'1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        b'2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        b'3' => [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
        b'4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        b'5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        b'6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        b'7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        b'8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        b'9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        b'a' => [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F],
        b'b' => [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x1E],
        b'c' => [0x00, 0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E],
        b'd' => [0x01, 0x01, 0x0D, 0x13, 0x11, 0x11, 0x0F],
        b'e' => [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E],
        b'f' => [0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08],
        b'g' => [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E],
        b'h' => [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11],
        b'i' => [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E],
        b'j' => [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C],
        b'k' => [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
        b'l' => [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        b'm' => [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11],
        b'n' => [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],
        b'o' => [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],
        b'p' => [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10],
        b'q' => [0x00, 0x00, 0x0D, 0x13, 0x0F, 0x01, 0x01],
        b'r' => [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
        b's' => [0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E],
        b't' => [0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06],
        b'u' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D],
        b'v' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04],
        b'w' => [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A],
        b'x' => [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],
        b'y' => [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E],
        b'z' => [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F],
        _ => [0x00; FONT_HEIGHT],
    }
}

fn draw_char(buf: &mut [u8], frame_width: usize, x: usize, y: usize, ch: u8, color: [u8; 3]) {
    let g = glyph(ch);
    for row in 0..FONT_HEIGHT {
        let bits = g[row];
        for col in 0..FONT_WIDTH {
            if bits & (1 << (FONT_WIDTH - 1 - col)) != 0 {
                let px = x + col;
                let py = y + row;
                let offset = (py * frame_width + px) * 4;
                if offset + 3 < buf.len() {
                    buf[offset] = color[0];
                    buf[offset + 1] = color[1];
                    buf[offset + 2] = color[2];
                    buf[offset + 3] = 255;
                }
            }
        }
    }
}

/// Draw a string of text at (x, y) in the given color. Returns the x position after the last character.
pub(crate) fn draw_text(buf: &mut [u8], frame_width: usize, x: usize, y: usize, text: &str, color: [u8; 3]) -> usize {
    let char_step = FONT_WIDTH + 1;
    let mut cx = x;
    for &ch in text.as_bytes() {
        draw_char(buf, frame_width, cx, y, ch, color);
        cx += char_step;
    }
    cx
}

/// Draw a character at (x, y) resized to target (cw × ch) pixels via nearest-neighbor.
fn draw_char_sized(buf: &mut [u8], frame_width: usize, x: usize, y: usize, ch_code: u8, color: [u8; 3], cw: usize, ch: usize) {
    let g = glyph(ch_code);
    for py in 0..ch {
        let src_row = py * FONT_HEIGHT / ch;
        let bits = g[src_row];
        for px in 0..cw {
            let src_col = px * FONT_WIDTH / cw;
            if bits & (1 << (FONT_WIDTH - 1 - src_col)) != 0 {
                let offset = ((y + py) * frame_width + x + px) * 4;
                if offset + 3 < buf.len() {
                    buf[offset] = color[0];
                    buf[offset + 1] = color[1];
                    buf[offset + 2] = color[2];
                    buf[offset + 3] = 255;
                }
            }
        }
    }
}

/// Draw a string of text at (x, y) with each character sized to (cw × ch) pixels.
/// Returns the x position after the last character.
pub(crate) fn draw_text_sized(buf: &mut [u8], frame_width: usize, x: usize, y: usize, text: &str, color: [u8; 3], cw: usize, ch: usize) -> usize {
    let char_step = cw + cw / 5 + 1; // proportional spacing (~20% of char width)
    let mut cx = x;
    for &byte in text.as_bytes() {
        draw_char_sized(buf, frame_width, cx, y, byte, color, cw, ch);
        cx += char_step;
    }
    cx
}

/// Draw status text at the bottom of the frame buffer.
pub fn render_status(buf: &mut [u8], cfg: &RenderConfig, text: &str) {
    let fw = cfg.frame_width;
    let y_start = cfg.display_height;

    // Fill status bar background (#0D0D0D)
    for y in y_start..cfg.frame_height {
        for x in 0..fw {
            let offset = (y * fw + x) * 4;
            if offset + 3 < buf.len() {
                buf[offset] = 0x0D;
                buf[offset + 1] = 0x0D;
                buf[offset + 2] = 0x0D;
                buf[offset + 3] = 255;
            }
        }
    }

    // Separator line (#333333)
    for x in 0..fw {
        let offset = (y_start * fw + x) * 4;
        if offset + 3 < buf.len() {
            buf[offset] = 0x33;
            buf[offset + 1] = 0x33;
            buf[offset + 2] = 0x33;
            buf[offset + 3] = 255;
        }
    }

    // Draw text
    let text_y = y_start + STATUS_PAD_TOP;
    let text_color: [u8; 3] = [0x88, 0x88, 0x88];
    let char_step = FONT_WIDTH + 1;
    let mut cx = 4; // left padding
    for &ch in text.as_bytes() {
        if cx + FONT_WIDTH > fw {
            break;
        }
        draw_char(buf, fw, cx, text_y, ch, text_color);
        cx += char_step;
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

/// Compute vorticity field from velocity and find max absolute value.
fn compute_vorticity(vx: &[f64], vy: &[f64], nx: usize) -> (Vec<f64>, f64) {
    let mut omega = vec![0.0; nx * N];
    let mut max_abs = 0.0_f64;
    for j in 1..(N - 1) as i32 {
        for i in 1..(nx - 1) as i32 {
            // Central difference: ∂vy/∂x - ∂vx/∂y
            let dvydx = (vy[idx(i + 1, j, nx)] - vy[idx(i - 1, j, nx)]) * 0.5;
            let dvxdy = (vx[idx(i, j + 1, nx)] - vx[idx(i, j - 1, nx)]) * 0.5;
            let w = dvydx - dvxdy;
            omega[idx(i, j, nx)] = w;
            max_abs = max_abs.max(w.abs());
        }
    }
    (omega, max_abs)
}

/// Render field + color bar into a pre-allocated RGBA buffer.
/// The buffer is resized and zeroed as needed.
/// When `show_vorticity` is true, renders vorticity instead of temperature/dye.
pub fn render_into(buf: &mut Vec<u8>, snap: &FrameSnapshot, cfg: &RenderConfig, show_vorticity: bool) {
    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let frame_width = cfg.frame_width;
    let frame_height = cfg.frame_height;
    let nx = snap.nx;

    let total = frame_width * frame_height * 4;
    buf.resize(total, 0);
    buf.fill(0);

    // Precompute vorticity if needed
    let (vort_field, vort_max) = if show_vorticity {
        compute_vorticity(&snap.vx, &snap.vy, nx)
    } else {
        (vec![], 0.0)
    };

    // Draw simulation tiled horizontally.
    // NX is calculated to match the window aspect ratio, so always stretch.
    let tiles = cfg.tiles;
    let cylinder = snap.cylinder;
    let sim_total_x = tiles as f64 * nx as f64;
    for screen_y in 0..dh {
        for screen_x in 0..dw {
            let offset = (screen_y * frame_width + screen_x) * 4;

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

            let rgba = if show_vorticity {
                let w00 = vort_field[idx(x0, y0, nx)];
                let w10 = vort_field[idx(x1, y0, nx)];
                let w01 = vort_field[idx(x0, y1, nx)];
                let w11 = vort_field[idx(x1, y1, nx)];
                let w = w00 * (1.0 - fx) * (1.0 - fy)
                      + w10 * fx * (1.0 - fy)
                      + w01 * (1.0 - fx) * fy
                      + w11 * fx * fy;
                // |ω| with gamma compression → dark red(quiet) to black(strong vortex)
                let raw = if vort_max > 0.0 { (w.abs() / vort_max).min(1.0) } else { 0.0 };
                let s = 1.0 - raw.powf(0.25); // gamma < 1 enhances weak vorticity
                [(s * 140.0) as u8, (s * 12.0) as u8, (s * 12.0) as u8, 255]
            } else {
                let t00 = snap.temperature[idx(x0, y0, nx)];
                let t10 = snap.temperature[idx(x1, y0, nx)];
                let t01 = snap.temperature[idx(x0, y1, nx)];
                let t11 = snap.temperature[idx(x1, y1, nx)];
                let t = t00 * (1.0 - fx) * (1.0 - fy)
                    + t10 * fx * (1.0 - fy)
                    + t01 * (1.0 - fx) * fy
                    + t11 * fx * fy;
                temperature_to_rgba(t)
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

    // Draw color bar (1.0 at top, 0.0 at bottom — matching tick labels)
    let bar_x = dw + BAR_GAP;
    for y in 0..dh {
        let t = 1.0 - y as f64 / (dh - 1) as f64;
        let rgba = if show_vorticity {
            // Dark red-black bar
            let s = t;
            [(s * 140.0) as u8, (s * 12.0) as u8, (s * 12.0) as u8, 255]
        } else {
            temperature_to_rgba(t)
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
        let label_y = if y >= FONT_HEIGHT / 2 { y - FONT_HEIGHT / 2 } else { 0 };
        let label_y = label_y.min(dh.saturating_sub(FONT_HEIGHT));
        draw_text(buf, frame_width, label_x, label_y, label, label_color);
    }

    // Draw type label above the bar
    let type_label = if show_vorticity { "vort" } else { "temp" };
    let type_label_y = if dh > FONT_HEIGHT + 4 { 2 } else { 0 };
    // Right-align within bar area
    let type_label_w = type_label.len() * (FONT_WIDTH + 1);
    let type_label_x = if bar_x + BAR_WIDTH / 2 >= type_label_w / 2 {
        bar_x + BAR_WIDTH / 2 - type_label_w / 2
    } else {
        bar_x
    };
    draw_text(buf, frame_width, type_label_x, type_label_y, type_label, [0xAA, 0xAA, 0xAA]);

    // Draw particles as 3x3 diamond with adaptive contrast.
    const CORE_BRIGHT: [f64; 3] = [240.0, 240.0, 220.0];
    const CORE_DARK: [f64; 3] = [8.0, 8.0, 8.0];
    const DIAMOND: [(isize, isize, bool); 5] = [
        (0, -1, false),
        (-1, 0, false),
        (0, 0, true),
        (1, 0, false),
        (0, 1, false),
    ];

    let sx = cfg.scale_x();
    let sy = cfg.scale_y();
    let tile_width_px = (dw as isize) / tiles as isize;

    // Draw particle trails (oldest first, fading in + tapering size).
    const TRAIL_SHAPES: [&[(isize, isize, f64)]; 3] = [
        // Small: 3x3 diamond
        &[(0, -1, 0.6), (-1, 0, 0.6), (0, 0, 1.0), (1, 0, 0.6), (0, 1, 0.6)],
        // Medium: 3x3 filled block
        &[
            (-1, -1, 0.4), (0, -1, 0.7), (1, -1, 0.4),
            (-1,  0, 0.7), (0,  0, 1.0), (1,  0, 0.7),
            (-1,  1, 0.4), (0,  1, 0.7), (1,  1, 0.4),
        ],
        // Large: 5x5 diamond
        &[
                                (0, -2, 0.3),
                   (-1, -1, 0.5), (0, -1, 0.8), (1, -1, 0.5),
        (-2, 0, 0.3), (-1,  0, 0.8), (0,  0, 1.0), (1,  0, 0.8), (2, 0, 0.3),
                   (-1,  1, 0.5), (0,  1, 0.8), (1,  1, 0.5),
                                (0,  2, 0.3),
        ],
    ];
    let trail_count = snap.trail_xs.len();
    if trail_count > 1 {
        for trail_idx in 0..(trail_count - 1) {
            let frac = (trail_idx + 1) as f64 / trail_count as f64;
            let alpha = frac * 0.8;
            // Shape index: oldest=0 (small), newest=2 (large)
            let shape_idx = (frac * 3.0).min(2.0) as usize;
            let shape = TRAIL_SHAPES[shape_idx];
            let txs = &snap.trail_xs[trail_idx];
            let tys = &snap.trail_ys[trail_idx];

            for p in 0..txs.len() {
                let base_cx = (txs[p] * sx) as isize;
                let cy = (((N - 1) as f64 - tys[p]) * sy) as isize;

                for tile in 0..tiles {
                    let cx = base_cx + tile_width_px * tile as isize;

                    // Sample background luminance at center for adaptive contrast
                    let ux = (cx as usize).min(dw.saturating_sub(1));
                    let uy = (cy as usize).min(dh.saturating_sub(1));
                    let bg_sample = (uy * frame_width + ux) * 4;
                    let lum = buf[bg_sample] as f64 * 0.3 + buf[bg_sample + 1] as f64 * 0.59 + buf[bg_sample + 2] as f64 * 0.11;
                    let tf = ((lum - 80.0) / 100.0).clamp(0.0, 1.0);
                    let dr = CORE_BRIGHT[0] + tf * (CORE_DARK[0] - CORE_BRIGHT[0]);
                    let dg = CORE_BRIGHT[1] + tf * (CORE_DARK[1] - CORE_BRIGHT[1]);
                    let db = CORE_BRIGHT[2] + tf * (CORE_DARK[2] - CORE_BRIGHT[2]);

                    for &(dx, dy, weight) in shape {
                        let px = (cx + dx) as usize;
                        let py = (cy + dy) as usize;
                        if px < dw && py < dh {
                            let off = (py * frame_width + px) * 4;
                            let a = alpha * weight;
                            let bg_r = buf[off] as f64;
                            let bg_g = buf[off + 1] as f64;
                            let bg_b = buf[off + 2] as f64;
                            buf[off] = (bg_r * (1.0 - a) + dr * a) as u8;
                            buf[off + 1] = (bg_g * (1.0 - a) + dg * a) as u8;
                            buf[off + 2] = (bg_b * (1.0 - a) + db * a) as u8;
                        }
                    }
                }
            }
        }
    }
    for i in 0..snap.particles_x.len() {
        let sim_x = snap.particles_x[i];
        let sim_y = snap.particles_y[i];

        // Base screen position (left copy)
        let base_cx = (sim_x * sx) as isize;
        let cy = (((N - 1) as f64 - sim_y) * sy) as isize;

        // Draw particle at all tile copies
        for t in 0..tiles {
            let copy_offset = tile_width_px * t as isize;
            let cx = base_cx + copy_offset;

            let ux = (cx as usize).min(dw - 1);
            let uy = (cy as usize).min(dh - 1);
            let bg_off = (uy * frame_width + ux) * 4;
            let lum = buf[bg_off] as f64 * 0.3 + buf[bg_off + 1] as f64 * 0.59 + buf[bg_off + 2] as f64 * 0.11;

            let t = ((lum - 80.0) / 100.0).clamp(0.0, 1.0);
            let core = [
                CORE_BRIGHT[0] + t * (CORE_DARK[0] - CORE_BRIGHT[0]),
                CORE_BRIGHT[1] + t * (CORE_DARK[1] - CORE_BRIGHT[1]),
                CORE_BRIGHT[2] + t * (CORE_DARK[2] - CORE_BRIGHT[2]),
            ];

            for &(dx, dy, is_core) in &DIAMOND {
                let px = (cx + dx) as usize;
                let py = (cy + dy) as usize;
                if px < dw && py < dh {
                    let offset = (py * frame_width + px) * 4;
                    if is_core {
                        buf[offset] = core[0] as u8;
                        buf[offset + 1] = core[1] as u8;
                        buf[offset + 2] = core[2] as u8;
                    } else {
                        let bg_r = buf[offset] as f64;
                        let bg_g = buf[offset + 1] as f64;
                        let bg_b = buf[offset + 2] as f64;
                        buf[offset] = (bg_r * 0.5 + core[0] * 0.5) as u8;
                        buf[offset + 1] = (bg_g * 0.5 + core[1] * 0.5) as u8;
                        buf[offset + 2] = (bg_b * 0.5 + core[2] * 0.5) as u8;
                    }
                    buf[offset + 3] = 255;
                }
            }
        }
    }

}

/// Render field + color bar to a new RGBA buffer.
/// When `show_vorticity` is true, renders vorticity instead of temperature/dye.
pub fn render(snap: &FrameSnapshot, cfg: &RenderConfig, show_vorticity: bool) -> Vec<u8> {
    let mut buf = Vec::new();
    render_into(&mut buf, snap, cfg, show_vorticity);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SimState;

    fn test_config() -> RenderConfig {
        RenderConfig::fit(542, 512, 3, N)
    }

    #[test]
    fn test_color_cold_is_navy() {
        let rgba = temperature_to_rgba(0.0);
        assert_eq!(rgba[0], 26, "R should be 26");
        assert_eq!(rgba[1], 27, "G should be 27");
        assert_eq!(rgba[2], 38, "B should be 38");
        assert_eq!(rgba[3], 255, "A should be 255");
    }

    #[test]
    fn test_color_hot_is_orange() {
        let rgba = temperature_to_rgba(1.0);
        assert_eq!(rgba[0], 255, "R should be 255");
        assert_eq!(rgba[1], 158, "G should be 158");
        assert_eq!(rgba[2], 100, "B should be 100");
    }

    #[test]
    fn test_color_mid_is_purple() {
        let rgba = temperature_to_rgba(0.5);
        assert_eq!(rgba[0], 187, "R should be 187");
        assert_eq!(rgba[1], 154, "G should be 154");
        assert_eq!(rgba[2], 247, "B should be 247");
    }

    #[test]
    fn test_color_clamp() {
        let lo = temperature_to_rgba(-1.0);
        let hi = temperature_to_rgba(2.0);
        assert_eq!(lo, temperature_to_rgba(0.0));
        assert_eq!(hi, temperature_to_rgba(1.0));
    }

    #[test]
    fn test_gradient_continuity() {
        let steps = 256;
        for i in 1..steps {
            let t0 = (i - 1) as f64 / (steps - 1) as f64;
            let t1 = i as f64 / (steps - 1) as f64;
            let c0 = temperature_to_rgba(t0);
            let c1 = temperature_to_rgba(t1);
            for ch in 0..3 {
                let diff = (c1[ch] as i32 - c0[ch] as i32).abs();
                assert!(
                    diff <= 5,
                    "Color channel {} jumped by {} between t={} and t={}",
                    ch, diff, t0, t1
                );
            }
        }
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
        let buf = render(&snap, &cfg, false);
        assert_eq!(buf.len(), cfg.frame_width * cfg.frame_height * 4);
    }

    #[test]
    fn test_render_y_flip() {
        let snap = SimState::new(400, 0.15, N).snapshot();
        let cfg = test_config();
        let buf = render(&snap, &cfg, false);

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
        let buf = render(&snap, &cfg, false);

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
    fn test_color_bar_gradient() {
        let snap = SimState::new(400, 0.15, N).snapshot();
        let cfg = test_config();
        let buf = render(&snap, &cfg, false);

        let bar_x = cfg.display_width + BAR_GAP + BAR_WIDTH / 2;
        let top_offset = bar_x * 4;
        let bot_offset = ((cfg.display_height - 1) * cfg.frame_width + bar_x) * 4;
        let top_bright = buf[top_offset] as u32 + buf[top_offset + 1] as u32 + buf[top_offset + 2] as u32;
        let bot_bright = buf[bot_offset] as u32 + buf[bot_offset + 1] as u32 + buf[bot_offset + 2] as u32;
        assert!(top_bright > bot_bright, "Bar top (1.0/hot) should be brighter than bottom (0.0/cold)");
    }

    #[test]
    fn test_render_status_draws_text() {
        let cfg = test_config();
        let mut buf = vec![0u8; cfg.frame_width * cfg.frame_height * 4];
        render_status(&mut buf, &cfg, "test");

        // Status area should have non-zero pixels (background + text)
        let status_start = cfg.display_height * cfg.frame_width * 4;
        let status_area = &buf[status_start..];
        let has_content = status_area.iter().any(|&b| b != 0);
        assert!(has_content, "Status bar should have rendered content");
    }

    #[test]
    fn test_render_cylinder_gray() {
        let mut snap = SimState::new(0, 0.15, N).snapshot();
        // Add a cylinder at center
        snap.cylinder = Some((64.0, 64.0, 20.0));
        snap.temperature.fill(0.5); // uniform temp

        let cfg = test_config();
        let buf = render(&snap, &cfg, false);

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
        let buf = render(&snap, &cfg, false);
        // Just check it doesn't crash and has correct size
        assert_eq!(buf.len(), cfg.frame_width * cfg.frame_height * 4);
    }

    #[test]
    fn test_glyph_dash_arrow() {
        // '-' and '>' should have non-zero bitmaps
        let dash = super::glyph(b'-');
        let has_dash = dash.iter().any(|&row| row != 0);
        assert!(has_dash, "'-' glyph should have non-zero bits");

        let arrow = super::glyph(b'>');
        let has_arrow = arrow.iter().any(|&row| row != 0);
        assert!(has_arrow, "'>' glyph should have non-zero bits");
    }

    #[test]
    fn test_draw_text_returns_end_position() {
        let cfg = test_config();
        let mut buf = vec![0u8; cfg.frame_width * cfg.frame_height * 4];
        let color = [0xFF, 0xFF, 0xFF];
        let end_x = super::draw_text(&mut buf, cfg.frame_width, 10, 10, "hello", color);
        // "hello" = 5 chars, each FONT_WIDTH + 1 pixel spacing = 6 * 5 = 30
        let expected = 10 + 5 * (super::FONT_WIDTH + 1);
        assert_eq!(end_x, expected, "draw_text should return cursor position after text");

        // Verify some pixels were drawn (non-zero in the text area)
        let mut found = false;
        for y in 10..10 + super::FONT_HEIGHT {
            for x in 10..end_x {
                let off = (y * cfg.frame_width + x) * 4;
                if buf[off] != 0 {
                    found = true;
                    break;
                }
            }
        }
        assert!(found, "draw_text should have drawn some pixels");
    }

    #[test]
    fn test_render_status_separator_line() {
        let cfg = test_config();
        let mut buf = vec![0u8; cfg.frame_width * cfg.frame_height * 4];
        render_status(&mut buf, &cfg, "hello");

        // First row of status area should be separator (#333333)
        let sep_offset = (cfg.display_height * cfg.frame_width + 0) * 4;
        assert_eq!(buf[sep_offset], 0x33);
        assert_eq!(buf[sep_offset + 1], 0x33);
        assert_eq!(buf[sep_offset + 2], 0x33);
    }

    #[test]
    fn test_render_nonsquare() {
        // Test with nx=256 (wider than N=128)
        let nx = 256;
        let snap = SimState::new(0, 0.15, nx).snapshot();
        assert_eq!(snap.nx, nx);
        let cfg = RenderConfig::fit(800, 512, 1, nx);
        let buf = render(&snap, &cfg, false);
        assert_eq!(buf.len(), cfg.frame_width * cfg.frame_height * 4);
    }

    #[test]
    fn test_color_bar_labels() {
        let snap = SimState::new(0, 0.15, N).snapshot();
        let cfg = test_config();
        let buf = render(&snap, &cfg, false);

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
        let buf_with_trail = render(&snap, &cfg, false);

        // Compare against render without trail
        let mut no_trail_snap = snap;
        no_trail_snap.trail_xs.clear();
        no_trail_snap.trail_ys.clear();
        let buf_no_trail = render(&no_trail_snap, &cfg, false);

        // The two buffers should differ (trail pixels changed some values)
        let diffs: usize = buf_with_trail.iter().zip(buf_no_trail.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(diffs > 0, "Trail rendering should modify some pixels");
    }
}
