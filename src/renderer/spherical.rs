// Spherical data renderer: equirectangular and orthographic projections.

use super::color;
use super::font;
use crate::spherical::{interpolate_gauss, SphericalParticleSystem, SphericalSnapshot};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    Equirectangular,
    Orthographic,
}

impl Projection {
    pub fn toggle(self) -> Self {
        match self {
            Projection::Equirectangular => Projection::Orthographic,
            Projection::Orthographic => Projection::Equirectangular,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Projection::Equirectangular => "equirect",
            Projection::Orthographic => "ortho",
        }
    }
}

pub struct SphericalRenderConfig {
    pub display_width: usize,
    pub display_height: usize,
    pub frame_width: usize,
    pub frame_height: usize,
}

impl SphericalRenderConfig {
    pub fn equirectangular(win_width: usize, win_height: usize) -> Self {
        let bar_total = color::BAR_TOTAL;
        let status_h = font::STATUS_BAR_HEIGHT;
        let dw = win_width.saturating_sub(bar_total).max(4);
        let dh = win_height.saturating_sub(status_h).max(2);
        Self {
            display_width: dw,
            display_height: dh,
            frame_width: dw + bar_total,
            frame_height: dh + status_h,
        }
    }

    pub fn orthographic(win_width: usize, win_height: usize) -> Self {
        // Same layout; the rendering logic handles the circular viewport
        Self::equirectangular(win_width, win_height)
    }
}

/// Render equirectangular projection into RGBA buffer.
pub fn render_equirectangular(
    buf: &mut Vec<u8>,
    snap: &SphericalSnapshot,
    gauss_nodes: &[f64],
    cfg: &SphericalRenderConfig,
    colormap: color::ColorMap,
) {
    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let fw = cfg.frame_width;
    let fh = cfg.frame_height;

    buf.resize(fw * fh * 4, 0);
    buf.fill(0);

    // Use global range if available, otherwise per-frame range
    let (vmin, vmax) = snap
        .global_range
        .unwrap_or_else(|| data_range(&snap.field_data));
    let range = vmax - vmin;
    let inv_range = if range > 1e-30 { 1.0 / range } else { 1.0 };

    for sy in 0..dh {
        // lat from +π/2 (top) to -π/2 (bottom)
        let lat = PI / 2.0 - (sy as f64 + 0.5) / dh as f64 * PI;
        let mu = lat.sin();

        for sx in 0..dw {
            let lon = (sx as f64 + 0.5) / dw as f64 * 2.0 * PI;

            let val = interpolate_gauss(
                &snap.field_data,
                snap.im,
                snap.jm,
                gauss_nodes,
                lon,
                mu,
            );

            let t = (val - vmin) * inv_range;
            let rgba = color::map_to_rgba(t, colormap);

            let off = (sy * fw + sx) * 4;
            buf[off] = rgba[0];
            buf[off + 1] = rgba[1];
            buf[off + 2] = rgba[2];
            buf[off + 3] = 255;
        }
    }

    draw_graticule_equirect(buf, cfg);
    render_color_bar(buf, cfg, colormap, vmin, vmax, &snap.field_name);
}

/// Render orthographic projection into RGBA buffer.
pub fn render_orthographic(
    buf: &mut Vec<u8>,
    snap: &SphericalSnapshot,
    gauss_nodes: &[f64],
    cfg: &SphericalRenderConfig,
    colormap: color::ColorMap,
    cam_lat: f64,
    cam_lon: f64,
) {
    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let fw = cfg.frame_width;
    let fh = cfg.frame_height;

    buf.resize(fw * fh * 4, 0);
    buf.fill(0);

    let (vmin, vmax) = snap
        .global_range
        .unwrap_or_else(|| data_range(&snap.field_data));
    let range = vmax - vmin;
    let inv_range = if range > 1e-30 { 1.0 / range } else { 1.0 };

    let radius = (dw.min(dh) as f64) * 0.45;
    let cx = dw as f64 / 2.0;
    let cy = dh as f64 / 2.0;

    // Camera rotation: rotate from view coords to sphere coords
    let sin_clat = cam_lat.sin();
    let cos_clat = cam_lat.cos();
    let sin_clon = cam_lon.sin();
    let cos_clon = cam_lon.cos();

    // Background color (dark)
    let bg = [8u8, 10, 20, 255];

    for sy in 0..dh {
        for sx in 0..dw {
            let off = (sy * fw + sx) * 4;

            let nx = (sx as f64 - cx) / radius;
            let ny = (cy - sy as f64) / radius; // y up
            let r2 = nx * nx + ny * ny;

            if r2 > 1.0 {
                buf[off] = bg[0];
                buf[off + 1] = bg[1];
                buf[off + 2] = bg[2];
                buf[off + 3] = bg[3];
                continue;
            }

            let nz = (1.0 - r2).sqrt();

            // Inverse camera rotation: view → world
            // Camera looks at (cam_lon, cam_lat) along -z
            // Rotation: Rz(-cam_lon) * Ry(-cam_lat) applied to (nx, ny, nz)
            let x1 = nx;
            let y1 = ny * cos_clat + nz * sin_clat;
            let z1 = -ny * sin_clat + nz * cos_clat;

            let world_x = x1 * cos_clon + z1 * sin_clon;
            let _world_y = y1;
            let world_z = -x1 * sin_clon + z1 * cos_clon;

            // World xyz → lon, lat
            let lat = y1.asin();
            let lon = world_x.atan2(world_z);
            let lon = if lon < 0.0 { lon + 2.0 * PI } else { lon };
            let mu = lat.sin();

            let val = interpolate_gauss(
                &snap.field_data,
                snap.im,
                snap.jm,
                gauss_nodes,
                lon,
                mu,
            );

            let t = (val - vmin) * inv_range;
            let rgba = color::map_to_rgba(t, colormap);

            // Limb darkening
            let limb = nz.powf(0.3);

            let mut r = (rgba[0] as f64 * limb) as u8;
            let mut g = (rgba[1] as f64 * limb) as u8;
            let mut b = (rgba[2] as f64 * limb) as u8;

            // Graticule overlay
            let pixel_deg = 180.0 / (radius * 2.0).max(1.0);
            let threshold = pixel_deg * 0.8;
            if let Some(is_eq) = graticule_hit(lon, lat, 30.0, 30.0, threshold) {
                let gray = if is_eq { 220.0 } else { 160.0 };
                let a = if is_eq { 0.5 } else { 0.35 };
                r = (r as f64 * (1.0 - a) + gray * a) as u8;
                g = (g as f64 * (1.0 - a) + gray * a) as u8;
                b = (b as f64 * (1.0 - a) + gray * a) as u8;
            }

            buf[off] = r;
            buf[off + 1] = g;
            buf[off + 2] = b;
            buf[off + 3] = 255;
        }
    }

    render_color_bar(buf, cfg, colormap, vmin, vmax, &snap.field_name);
}

/// Check if (lon, lat) in radians is near a graticule line.
/// Returns Some(true) for equator, Some(false) for other grid lines, None if not near any.
fn graticule_hit(lon: f64, lat: f64, lon_step: f64, lat_step: f64, threshold: f64) -> Option<bool> {
    let lat_deg = lat.to_degrees();
    let lon_deg = lon.to_degrees();

    // Latitude lines
    let lat_rem = ((lat_deg % lat_step) + lat_step) % lat_step;
    if lat_rem < threshold || lat_rem > lat_step - threshold {
        // Check if equator
        let is_equator = lat_deg.abs() < threshold;
        return Some(is_equator);
    }

    // Longitude lines
    let lon_rem = ((lon_deg % lon_step) + lon_step) % lon_step;
    if lon_rem < threshold || lon_rem > lon_step - threshold {
        return Some(false);
    }

    None
}

/// Blend a graticule line pixel over existing RGBA.
fn blend_line(buf: &mut [u8], off: usize, gray: f64, alpha: f64) {
    buf[off] = (buf[off] as f64 * (1.0 - alpha) + gray * alpha) as u8;
    buf[off + 1] = (buf[off + 1] as f64 * (1.0 - alpha) + gray * alpha) as u8;
    buf[off + 2] = (buf[off + 2] as f64 * (1.0 - alpha) + gray * alpha) as u8;
}

/// Draw text with dark shadow for readability over variable backgrounds.
fn draw_text_shadow(buf: &mut [u8], fw: usize, x: usize, y: usize, text: &str, color: [u8; 3]) {
    font::draw_text(buf, fw, x + 1, y + 1, text, [0, 0, 0]);
    font::draw_text(buf, fw, x, y, text, color);
}

/// Draw sized text with dark shadow for readability.
fn draw_text_shadow_sized(buf: &mut [u8], fw: usize, x: usize, y: usize, text: &str, color: [u8; 3], cw: usize, ch: usize) {
    font::draw_text_sized(buf, fw, x + 1, y + 1, text, [0, 0, 0], cw, ch);
    font::draw_text_sized(buf, fw, x, y, text, color, cw, ch);
}

/// Render a field-name badge at the top-left of the display area.
///
/// Layout:
/// ```text
///   ┌────────────────────────┐
///   │  vort anomaly     1/6  │
///   └────────────────────────┘
/// ```
pub fn render_field_badge(
    buf: &mut [u8],
    cfg: &SphericalRenderConfig,
    field_name: &str,
    field_index: usize,
    field_count: usize,
) {
    let fw = cfg.frame_width;
    let display_name = field_display_name(field_name);

    // Badge font sizes
    let big_cw: usize = 8;
    let big_ch: usize = 11;
    let big_step = big_cw + big_cw / 5 + 1; // matches draw_text_sized spacing
    let small_step = font::FONT_WIDTH + 1;

    // Index text: "1/6"
    let index_text = format!("{}/{}", field_index + 1, field_count);

    // Compute badge dimensions
    let name_w = display_name.len() * big_step;
    let idx_w = index_text.len() * small_step;
    let gap = big_step; // gap between name and index
    let pad_x: usize = 8;
    let pad_y: usize = 5;

    let content_w = name_w + gap + idx_w;
    let badge_w = pad_x * 2 + content_w;
    let badge_h = pad_y * 2 + big_ch;

    let margin: usize = 10;
    let badge_x = margin;
    let badge_y = margin;

    // Draw semi-transparent dark background (70% alpha blend)
    let bg_r = 10.0_f64;
    let bg_g = 12.0;
    let bg_b = 20.0;
    let alpha = 0.7;

    for y in badge_y..badge_y + badge_h {
        for x in badge_x..badge_x + badge_w {
            if x >= fw || y >= cfg.display_height {
                continue;
            }
            let off = (y * fw + x) * 4;
            if off + 3 < buf.len() {
                let r = buf[off] as f64;
                let g = buf[off + 1] as f64;
                let b = buf[off + 2] as f64;
                buf[off] = (r * (1.0 - alpha) + bg_r * alpha) as u8;
                buf[off + 1] = (g * (1.0 - alpha) + bg_g * alpha) as u8;
                buf[off + 2] = (b * (1.0 - alpha) + bg_b * alpha) as u8;
            }
        }
    }

    // Draw field display name (8x11 sized font with shadow)
    let text_x = badge_x + pad_x;
    let text_y = badge_y + pad_y;
    let name_color = [0xCC, 0xCC, 0xD0];
    draw_text_shadow_sized(buf, fw, text_x, text_y, display_name, name_color, big_cw, big_ch);

    // Draw index "1/6" (5x7 normal font, dimmer)
    let idx_x = text_x + name_w + gap;
    let idx_y = text_y + (big_ch.saturating_sub(font::FONT_HEIGHT)) / 2; // vertically center
    let idx_color = [0x55, 0x55, 0x55];
    font::draw_text(buf, fw, idx_x, idx_y, &index_text, idx_color);
}

/// Draw graticule (lat/lon grid lines + labels) on equirectangular projection.
fn draw_graticule_equirect(buf: &mut [u8], cfg: &SphericalRenderConfig) {
    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let fw = cfg.frame_width;
    let fh = cfg.frame_height;

    let label_color = [0xCC, 0xCC, 0xCC];
    let char_h = font::FONT_HEIGHT;
    let char_w = font::FONT_WIDTH + 1;

    // Latitude lines every 30° (60N, 30N, EQ, 30S, 60S)
    let lat_lines: [(i32, &str); 5] = [
        (60, "60N"),
        (30, "30N"),
        (0, "EQ"),
        (-30, "30S"),
        (-60, "60S"),
    ];

    for &(lat_deg, label) in &lat_lines {
        let lat = lat_deg as f64 * PI / 180.0;
        let sy = ((PI / 2.0 - lat) / PI * dh as f64) as usize;
        if sy >= dh {
            continue;
        }

        let is_equator = lat_deg == 0;
        let (gray, alpha) = if is_equator {
            (220.0, 0.45)
        } else {
            (160.0, 0.3)
        };

        for sx in 0..dw {
            let off = (sy * fw + sx) * 4;
            if off + 3 < buf.len() {
                blend_line(buf, off, gray, alpha);
            }
        }

        // Label on left edge
        let ly = if sy >= char_h / 2 {
            sy - char_h / 2
        } else {
            0
        };
        let ly = ly.min(dh.saturating_sub(char_h));
        if ly * fw * 4 + label.len() * char_w * 4 < buf.len() {
            draw_text_shadow(buf, fw, 2, ly, label, label_color);
        }
    }

    // Longitude lines every 30°
    let lon_labels: [(u32, &str); 6] = [
        (0, "0"),
        (60, "60E"),
        (120, "120E"),
        (180, "180"),
        (240, "120W"),
        (300, "60W"),
    ];

    for lon_deg_i in 0..12u32 {
        let lon_deg = lon_deg_i * 30;
        let sx = (lon_deg as f64 / 360.0 * dw as f64) as usize;
        if sx >= dw {
            continue;
        }

        for sy in 0..dh {
            let off = (sy * fw + sx) * 4;
            if off + 3 < buf.len() {
                blend_line(buf, off, 160.0, 0.3);
            }
        }

        // Label at bottom edge (only every 60°)
        if let Some((_, label)) = lon_labels.iter().find(|(d, _)| *d == lon_deg) {
            let lw = label.len() * char_w;
            let lx = if sx >= lw / 2 { sx - lw / 2 } else { 0 };
            let lx = lx.min(dw.saturating_sub(lw));
            let ly = dh.saturating_sub(char_h + 2);
            if ly < fh && lx < fw {
                draw_text_shadow(buf, fw, lx, ly, label, label_color);
            }
        }
    }
}

/// Map internal field names to human-readable display names.
pub fn field_display_name(raw: &str) -> &str {
    match raw {
        "dvort" => "vort anomaly",
        "dphi" => "height anomaly",
        "vort" => "vorticity",
        "phi" => "geopotential",
        "topo" => "topography",
        "u_cos" => "zonal wind",
        "v_cos" => "merid. wind",
        other => other,
    }
}

fn data_range(data: &[f64]) -> (f64, f64) {
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for &v in data {
        if v < vmin {
            vmin = v;
        }
        if v > vmax {
            vmax = v;
        }
    }
    if !vmin.is_finite() {
        vmin = 0.0;
    }
    if !vmax.is_finite() {
        vmax = 1.0;
    }
    if (vmax - vmin).abs() < 1e-30 {
        vmin -= 0.5;
        vmax += 0.5;
    }
    // For diverging data (vmin < 0 < vmax), center at zero
    if vmin < 0.0 && vmax > 0.0 {
        let abs_max = vmin.abs().max(vmax.abs());
        vmin = -abs_max;
        vmax = abs_max;
    }
    // Pad range for fields with small variation relative to their mean
    let mean = (vmin + vmax) * 0.5;
    let current_range = vmax - vmin;
    let min_range = mean.abs() * 0.01;
    if current_range < min_range && mean.abs() > 1e-10 {
        vmin = mean - min_range * 0.5;
        vmax = mean + min_range * 0.5;
    }
    (vmin, vmax)
}

fn render_color_bar(
    buf: &mut [u8],
    cfg: &SphericalRenderConfig,
    colormap: color::ColorMap,
    vmin: f64,
    vmax: f64,
    field_name: &str,
) {
    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let fw = cfg.frame_width;

    let bar_gap = color::BAR_GAP;
    let bar_width = color::BAR_WIDTH;
    let tick_len = color::TICK_LEN;
    let label_gap = color::LABEL_GAP;
    let bar_x = dw + bar_gap;

    // Draw bar gradient
    for y in 0..dh {
        let t = 1.0 - y as f64 / (dh.max(1) - 1).max(1) as f64;
        let rgba = color::map_to_rgba(t, colormap);
        for bx in 0..bar_width {
            let off = (y * fw + bar_x + bx) * 4;
            if off + 3 < buf.len() {
                buf[off] = rgba[0];
                buf[off + 1] = rgba[1];
                buf[off + 2] = rgba[2];
                buf[off + 3] = 255;
            }
        }
    }

    // Tick marks and labels
    let tick_x = bar_x + bar_width;
    let label_x = tick_x + tick_len + label_gap;
    let label_color = [0x88u8, 0x88, 0x88];

    let tick_values = [1.0, 0.75, 0.5, 0.25, 0.0];
    for (i, &tv) in tick_values.iter().enumerate() {
        let y = i * (dh.max(1) - 1) / 4;
        let actual_val = vmin + (vmax - vmin) * tv;
        let label = format_value(actual_val);

        for dy in 0..2usize {
            let yy = (y + dy).min(dh.saturating_sub(1));
            for tx in 0..tick_len {
                let off = (yy * fw + tick_x + tx) * 4;
                if off + 3 < buf.len() {
                    buf[off] = 255;
                    buf[off + 1] = 255;
                    buf[off + 2] = 255;
                    buf[off + 3] = 255;
                }
            }
        }

        let fh = font::FONT_HEIGHT;
        let label_y = if y >= fh / 2 { y - fh / 2 } else { 0 };
        let label_y = label_y.min(dh.saturating_sub(fh));
        font::draw_text(buf, fw, label_x, label_y, &label, label_color);
    }

    // Field name label (use display name)
    let display_name = field_display_name(field_name);
    let name_y = 2.min(dh.saturating_sub(font::FONT_HEIGHT));
    let char_step = font::FONT_WIDTH + 1;
    let name_w = display_name.len() * char_step;
    // Left-align at bar_x if name is wider than the bar
    let name_x = if name_w <= bar_width {
        // Center within bar
        if bar_x + bar_width / 2 >= name_w / 2 {
            bar_x + bar_width / 2 - name_w / 2
        } else {
            bar_x
        }
    } else {
        bar_x
    };
    font::draw_text(buf, fw, name_x, name_y, display_name, [0xAA, 0xAA, 0xAA]);
}

// --- Particle rendering helpers (local copies from renderer/mod.rs) ---

const METEOR_COLORS: [[f64; 3]; 4] = [
    [255.0, 240.0, 200.0], // warm white (newest)
    [180.0, 220.0, 255.0], // light cyan
    [80.0, 140.0, 255.0],  // blue
    [30.0, 50.0, 120.0],   // deep blue (oldest)
];

fn meteor_color(t: f64) -> [f64; 3] {
    let t = t.clamp(0.0, 1.0);
    let seg = t * (METEOR_COLORS.len() - 1) as f64;
    let i = (seg as usize).min(METEOR_COLORS.len() - 2);
    let f = seg - i as f64;
    let c0 = &METEOR_COLORS[METEOR_COLORS.len() - 1 - i];
    let c1 = &METEOR_COLORS[(METEOR_COLORS.len() - 2).saturating_sub(i)];
    [
        c0[0] + f * (c1[0] - c0[0]),
        c0[1] + f * (c1[1] - c0[1]),
        c0[2] + f * (c1[2] - c0[2]),
    ]
}

#[inline]
fn screen_blend(buf: &mut [u8], off: usize, r: f64, g: f64, b: f64, alpha: f64) {
    let br = buf[off] as f64;
    let bg = buf[off + 1] as f64;
    let bb = buf[off + 2] as f64;
    let fr = (r * alpha).min(255.0);
    let fg = (g * alpha).min(255.0);
    let fb = (b * alpha).min(255.0);
    buf[off] = (255.0 - (255.0 - br) * (255.0 - fr) / 255.0) as u8;
    buf[off + 1] = (255.0 - (255.0 - bg) * (255.0 - fg) / 255.0) as u8;
    buf[off + 2] = (255.0 - (255.0 - bb) * (255.0 - fb) / 255.0) as u8;
}

const GLOW_3X3: &[(isize, isize, f64)] = &[
                  (0, -1, 0.5),
    (-1, 0, 0.5), (0,  0, 1.0), (1, 0, 0.5),
                  (0,  1, 0.5),
];

/// Render particles on equirectangular projection.
pub fn render_particles_equirect(
    buf: &mut [u8],
    ps: &SphericalParticleSystem,
    cfg: &SphericalRenderConfig,
) {
    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let fw = cfg.frame_width;

    // Draw trails (oldest → newest)
    let trails = ps.ordered_trails();
    let n_trails = trails.len();
    for (ti, (lons, lats)) in trails.iter().enumerate() {
        let frac = if n_trails > 1 {
            ti as f64 / (n_trails - 1) as f64
        } else {
            1.0
        };
        let color = meteor_color(frac);
        let alpha = 0.15 + 0.35 * frac;

        for i in 0..lons.len() {
            let sx = (lons[i] / (2.0 * PI) * dw as f64) as isize;
            let sy = ((PI / 2.0 - lats[i]) / PI * dh as f64) as isize;
            if sx >= 0 && sx < dw as isize && sy >= 0 && sy < dh as isize {
                let off = (sy as usize * fw + sx as usize) * 4;
                if off + 3 < buf.len() {
                    screen_blend(buf, off, color[0], color[1], color[2], alpha);
                }
            }
        }
    }

    // Draw particle heads with 3x3 soft glow
    let head_color = [255.0, 240.0, 210.0];
    for p in &ps.particles {
        let sx = (p.lon / (2.0 * PI) * dw as f64) as isize;
        let sy = ((PI / 2.0 - p.lat) / PI * dh as f64) as isize;

        for &(dx, dy, weight) in GLOW_3X3 {
            let px = sx + dx;
            let py = sy + dy;
            if px >= 0 && px < dw as isize && py >= 0 && py < dh as isize {
                let off = (py as usize * fw + px as usize) * 4;
                if off + 3 < buf.len() {
                    screen_blend(buf, off, head_color[0], head_color[1], head_color[2], weight * 0.8);
                }
            }
        }
    }
}

/// Render particles on orthographic projection.
pub fn render_particles_ortho(
    buf: &mut [u8],
    ps: &SphericalParticleSystem,
    cfg: &SphericalRenderConfig,
    cam_lat: f64,
    cam_lon: f64,
) {
    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let fw = cfg.frame_width;
    let radius = (dw.min(dh) as f64) * 0.45;
    let cx = dw as f64 / 2.0;
    let cy = dh as f64 / 2.0;

    let sin_clat = cam_lat.sin();
    let cos_clat = cam_lat.cos();
    let sin_clon = cam_lon.sin();
    let cos_clon = cam_lon.cos();

    // Project (lon, lat) → screen, returning None if on back side
    let project = |lon: f64, lat: f64| -> Option<(isize, isize, f64)> {
        let cos_lat = lat.cos();
        let sin_lat = lat.sin();
        let x = cos_lat * lon.sin();
        let y = sin_lat;
        let z = cos_lat * lon.cos();

        // Camera rotation: world → view
        let rx = x * cos_clon + z * sin_clon;
        let rz = -x * sin_clon + z * cos_clon;
        let nx = rx;
        let ny = y * cos_clat - rz * sin_clat;
        let nz = y * sin_clat + rz * cos_clat;

        if nz < 0.0 {
            return None; // back side
        }

        let screen_x = (cx + nx * radius) as isize;
        let screen_y = (cy - ny * radius) as isize;
        Some((screen_x, screen_y, nz))
    };

    // Draw trails
    let trails = ps.ordered_trails();
    let n_trails = trails.len();
    for (ti, (lons, lats)) in trails.iter().enumerate() {
        let frac = if n_trails > 1 {
            ti as f64 / (n_trails - 1) as f64
        } else {
            1.0
        };
        let color = meteor_color(frac);
        let alpha = 0.15 + 0.35 * frac;

        for i in 0..lons.len() {
            if let Some((sx, sy, nz)) = project(lons[i], lats[i]) {
                if sx >= 0 && sx < dw as isize && sy >= 0 && sy < dh as isize {
                    let limb = nz.powf(0.3);
                    let off = (sy as usize * fw + sx as usize) * 4;
                    if off + 3 < buf.len() {
                        screen_blend(buf, off, color[0] * limb, color[1] * limb, color[2] * limb, alpha);
                    }
                }
            }
        }
    }

    // Draw heads
    let head_color = [255.0, 240.0, 210.0];
    for p in &ps.particles {
        if let Some((sx, sy, nz)) = project(p.lon, p.lat) {
            let limb = nz.powf(0.3);
            for &(dx, dy, weight) in GLOW_3X3 {
                let px = sx + dx;
                let py = sy + dy;
                if px >= 0 && px < dw as isize && py >= 0 && py < dh as isize {
                    let off = (py as usize * fw + px as usize) * 4;
                    if off + 3 < buf.len() {
                        screen_blend(
                            buf, off,
                            head_color[0] * limb, head_color[1] * limb, head_color[2] * limb,
                            weight * 0.8,
                        );
                    }
                }
            }
        }
    }
}

fn format_value(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        "0".to_string()
    } else if abs >= 100.0 {
        format!("{:.0}", v)
    } else if abs >= 1.0 {
        format!("{:.1}", v)
    } else if abs >= 0.01 {
        format!("{:.2}", v)
    } else {
        format!("{:.1e}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equirectangular_buffer_size() {
        let snap = SphericalSnapshot {
            im: 8,
            jm: 4,
            step: 0,
            time: 0.0,
            field_name: "test".to_string(),
            field_data: vec![1.0; 32],
            field_names: vec!["test".to_string()],
            field_index: 0,
            global_range: None,
        };
        let nodes = vec![-0.6, -0.2, 0.2, 0.6];
        let cfg = SphericalRenderConfig::equirectangular(200, 100);
        let mut buf = Vec::new();
        render_equirectangular(
            &mut buf,
            &snap,
            &nodes,
            &cfg,
            color::ColorMap::OceanLava,
        );
        assert_eq!(buf.len(), cfg.frame_width * cfg.frame_height * 4);
    }

    #[test]
    fn test_orthographic_buffer_size() {
        let snap = SphericalSnapshot {
            im: 8,
            jm: 4,
            step: 0,
            time: 0.0,
            field_name: "test".to_string(),
            field_data: vec![1.0; 32],
            field_names: vec!["test".to_string()],
            field_index: 0,
            global_range: None,
        };
        let nodes = vec![-0.6, -0.2, 0.2, 0.6];
        let cfg = SphericalRenderConfig::orthographic(200, 200);
        let mut buf = Vec::new();
        render_orthographic(
            &mut buf,
            &snap,
            &nodes,
            &cfg,
            color::ColorMap::OceanLava,
            0.0,
            0.0,
        );
        assert_eq!(buf.len(), cfg.frame_width * cfg.frame_height * 4);
    }

    #[test]
    fn test_data_range_diverging() {
        let (vmin, vmax) = data_range(&[-3.0, -1.0, 0.5, 2.0]);
        assert!((vmin - (-3.0)).abs() < 1e-10);
        assert!((vmax - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_data_range_positive() {
        let (vmin, vmax) = data_range(&[10.0, 20.0, 30.0]);
        assert!((vmin - 10.0).abs() < 1e-10);
        assert!((vmax - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_projection_toggle() {
        assert_eq!(
            Projection::Equirectangular.toggle(),
            Projection::Orthographic
        );
        assert_eq!(
            Projection::Orthographic.toggle(),
            Projection::Equirectangular
        );
    }

    #[test]
    fn test_graticule_hit_equator() {
        // Near equator (lat ≈ 0)
        let hit = graticule_hit(1.0, 0.005_f64.to_radians(), 30.0, 30.0, 0.5);
        assert_eq!(hit, Some(true), "should detect equator");
    }

    #[test]
    fn test_graticule_hit_lat_line() {
        // Near 30N
        let hit = graticule_hit(1.0, 30.1_f64.to_radians(), 30.0, 30.0, 0.5);
        assert_eq!(hit, Some(false), "should detect 30N lat line");
    }

    #[test]
    fn test_graticule_hit_lon_line() {
        // Near 60E longitude, away from any lat line
        let hit = graticule_hit(60.1_f64.to_radians(), 15.0_f64.to_radians(), 30.0, 30.0, 0.5);
        assert_eq!(hit, Some(false), "should detect 60E lon line");
    }

    #[test]
    fn test_graticule_hit_none() {
        // Middle of a grid cell — no line
        let hit = graticule_hit(45.0_f64.to_radians(), 15.0_f64.to_radians(), 30.0, 30.0, 0.5);
        assert_eq!(hit, None, "should not detect any grid line");
    }

    #[test]
    fn test_blue_white_red_endpoints() {
        let cold = color::map_to_rgba(0.0, color::ColorMap::BlueWhiteRed);
        let hot = color::map_to_rgba(1.0, color::ColorMap::BlueWhiteRed);
        let mid = color::map_to_rgba(0.5, color::ColorMap::BlueWhiteRed);
        // Cold should be blue-ish
        assert!(cold[2] > cold[0], "cold end should be blue");
        // Hot should be red-ish
        assert!(hot[0] > hot[2], "hot end should be red");
        // Mid should be near-white
        assert!(mid[0] > 200 && mid[1] > 200 && mid[2] > 200, "mid should be near-white");
    }

    #[test]
    fn test_blue_white_red_gradient_continuity() {
        let steps = 256;
        for i in 1..steps {
            let t0 = (i - 1) as f64 / (steps - 1) as f64;
            let t1 = i as f64 / (steps - 1) as f64;
            let c0 = color::map_to_rgba(t0, color::ColorMap::BlueWhiteRed);
            let c1 = color::map_to_rgba(t1, color::ColorMap::BlueWhiteRed);
            for ch in 0..3 {
                let diff = (c1[ch] as i32 - c0[ch] as i32).abs();
                assert!(
                    diff <= 5,
                    "BlueWhiteRed channel {} jumped by {} between t={:.3} and t={:.3}",
                    ch, diff, t0, t1
                );
            }
        }
    }
}
