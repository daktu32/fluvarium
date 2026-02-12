use crate::state::{idx, FrameSnapshot, N};

/// Blackbody-inspired color stops for temperature mapping.
/// Mostly black, only hot areas glow: black -> ember -> crimson -> orange -> amber
const COLOR_STOPS: [(f64, f64, f64); 5] = [
    (6.0, 6.0, 6.0),       // #060606 black        (0.00)
    (14.0, 6.0, 6.0),      // #0e0606 barely warm  (0.25)
    (72.0, 12.0, 12.0),    // #480c0c ember         (0.50)
    (204.0, 44.0, 20.0),   // #cc2c14 crimson       (0.75)
    (255.0, 190.0, 30.0),  // #ffbe1e amber         (1.00)
];

/// Convert temperature [0.0, 1.0] to RGBA color (blackbody palette).
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
const BAR_TOTAL: usize = BAR_GAP + BAR_WIDTH + TICK_LEN;

/// Dynamic render layout computed from terminal pixel size.
pub struct RenderConfig {
    pub display_width: usize,
    pub display_height: usize,
    pub frame_width: usize,
    pub frame_height: usize,
}

impl RenderConfig {
    /// Compute layout to fit the given pixel dimensions.
    /// Fills full width and height independently (non-square).
    pub fn fit(pixel_width: usize, pixel_height: usize) -> Self {
        let display_width = pixel_width.saturating_sub(BAR_TOTAL).max(N);
        let display_height = pixel_height.max(N);
        Self {
            display_width,
            display_height,
            frame_width: display_width + BAR_TOTAL,
            frame_height: display_height,
        }
    }

    /// Fallback config when terminal size cannot be determined.
    #[cfg(test)]
    pub fn default_config() -> Self {
        Self::fit(542, 512)
    }

    /// Effective scale factors (float) for particle positioning.
    /// X maps two simulation domains across the display width.
    pub fn scale_x(&self) -> f64 {
        self.display_width as f64 / (2.0 * N as f64)
    }

    pub fn scale_y(&self) -> f64 {
        self.display_height as f64 / N as f64
    }
}

/// Render temperature field + color bar to RGBA buffer.
pub fn render(snap: &FrameSnapshot, cfg: &RenderConfig) -> Vec<u8> {
    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let frame_width = cfg.frame_width;
    let frame_height = cfg.frame_height;

    let mut buf = vec![0u8; frame_width * frame_height * 4];

    // Draw simulation doubled horizontally (periodic wraparound visible).
    // Maps display width to 2× the simulation domain, wrapping via mod N.
    for screen_y in 0..dh {
        let sim_y = (N - 1) - screen_y * N / dh;
        for screen_x in 0..dw {
            let sim_x = (screen_x * 2 * N / dw) % N;
            let t = snap.temperature[idx(sim_x as i32, sim_y as i32)];
            let rgba = temperature_to_rgba(t);
            let offset = (screen_y * frame_width + screen_x) * 4;
            buf[offset] = rgba[0];
            buf[offset + 1] = rgba[1];
            buf[offset + 2] = rgba[2];
            buf[offset + 3] = rgba[3];
        }
    }

    // Draw color bar: top=cold(0.0), bottom=hot(1.0)
    let bar_x = dw + BAR_GAP;
    for y in 0..frame_height {
        let t = y as f64 / (frame_height - 1) as f64;
        let rgba = temperature_to_rgba(t);
        for bx in 0..BAR_WIDTH {
            let offset = (y * frame_width + bar_x + bx) * 4;
            buf[offset] = rgba[0];
            buf[offset + 1] = rgba[1];
            buf[offset + 2] = rgba[2];
            buf[offset + 3] = rgba[3];
        }
    }

    // Draw tick marks at 0%, 25%, 50%, 75%, 100%
    let tick_x = bar_x + BAR_WIDTH;
    for tick in 0..5u32 {
        let y = (tick as usize) * (frame_height - 1) / 4;
        for dy in 0..2usize {
            let yy = (y + dy).min(frame_height - 1);
            for tx in 0..TICK_LEN {
                let offset = (yy * frame_width + tick_x + tx) * 4;
                buf[offset] = 255;
                buf[offset + 1] = 255;
                buf[offset + 2] = 255;
                buf[offset + 3] = 255;
            }
        }
    }

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

    let sx_f = cfg.scale_x();
    let sy_f = cfg.scale_y();
    let half_dw = dw as isize / 2;
    for i in 0..snap.particles_x.len() {
        let sim_x = snap.particles_x[i];
        let sim_y = snap.particles_y[i];

        // Base screen position (left copy)
        let base_cx = (sim_x * sx_f) as isize;
        let cy = (((N - 1) as f64 - sim_y) * sy_f) as isize;

        // Draw particle at both copies (left and right, shifted by half display width)
        for copy_offset in [0, half_dw] {
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

    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SimState;

    fn test_config() -> RenderConfig {
        RenderConfig::fit(542, 512)
    }

    #[test]
    fn test_color_cold_is_near_black() {
        let rgba = temperature_to_rgba(0.0);
        assert_eq!(rgba[0], 6, "R should be 6");
        assert_eq!(rgba[1], 6, "G should be 6");
        assert_eq!(rgba[2], 6, "B should be 6");
        assert_eq!(rgba[3], 255, "A should be 255");
    }

    #[test]
    fn test_color_hot_is_amber() {
        let rgba = temperature_to_rgba(1.0);
        assert_eq!(rgba[0], 255, "R should be 255");
        assert_eq!(rgba[1], 190, "G should be 190");
        assert_eq!(rgba[2], 30, "B should be 30");
    }

    #[test]
    fn test_color_mid_is_ember() {
        let rgba = temperature_to_rgba(0.5);
        assert_eq!(rgba[0], 72, "R should be 72");
        assert_eq!(rgba[1], 12, "G should be 12");
        assert_eq!(rgba[2], 12, "B should be 12");
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
        let cfg = RenderConfig::fit(1200, 800);
        assert_eq!(cfg.display_width, 1200 - BAR_TOTAL);
        assert_eq!(cfg.display_height, 800);
        assert_eq!(cfg.frame_width, cfg.display_width + BAR_TOTAL);
        assert_eq!(cfg.frame_height, 800);
    }

    #[test]
    fn test_render_config_small_terminal() {
        let cfg = RenderConfig::fit(200, 100);
        assert_eq!(cfg.display_width, 200 - BAR_TOTAL); // 170
        assert_eq!(cfg.display_height, N); // clamped to minimum N
    }

    #[test]
    fn test_render_buffer_size() {
        let snap = SimState::new().snapshot();
        let cfg = test_config();
        let buf = render(&snap, &cfg);
        assert_eq!(buf.len(), cfg.frame_width * cfg.frame_height * 4);
    }

    #[test]
    fn test_render_y_flip() {
        let snap = SimState::new().snapshot();
        let cfg = test_config();
        let buf = render(&snap, &cfg);

        let top = &buf[0..4];
        let bottom_row = cfg.display_height - 1;
        let bottom = &buf[(bottom_row * cfg.frame_width) * 4..(bottom_row * cfg.frame_width) * 4 + 4];

        let top_brightness = top[0] as u32 + top[1] as u32 + top[2] as u32;
        let bot_brightness = bottom[0] as u32 + bottom[1] as u32 + bottom[2] as u32;
        assert!(bot_brightness > top_brightness, "Bottom (hot) should be brighter than top (cold)");
    }

    #[test]
    fn test_particles_rendered_diamond_adaptive() {
        let mut snap = SimState::new().snapshot();
        snap.particles_x.clear();
        snap.particles_y.clear();
        let mid = (N / 2) as f64;
        snap.particles_x.push(mid);
        snap.particles_y.push(mid);

        let cfg = test_config();
        let buf = render(&snap, &cfg);

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
        let snap = SimState::new().snapshot();
        let cfg = test_config();
        let buf = render(&snap, &cfg);

        let bar_x = cfg.display_width + BAR_GAP + BAR_WIDTH / 2;
        let top_offset = bar_x * 4;
        let bot_offset = ((cfg.frame_height - 1) * cfg.frame_width + bar_x) * 4;
        let top_bright = buf[top_offset] as u32 + buf[top_offset + 1] as u32 + buf[top_offset + 2] as u32;
        let bot_bright = buf[bot_offset] as u32 + buf[bot_offset + 1] as u32 + buf[bot_offset + 2] as u32;
        assert!(bot_bright > top_bright, "Bar bottom (hot) should be brighter than top (cold)");
    }
}
