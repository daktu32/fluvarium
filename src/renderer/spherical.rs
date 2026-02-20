// Spherical data renderer: equirectangular and orthographic projections.

use super::color;
use super::font;
use crate::spherical::{SphericalParticleSystem, SphericalSnapshot};
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
    use crate::spherical::find_gauss_neighbors;

    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let fw = cfg.frame_width;
    let fh = cfg.frame_height;
    let im = snap.im;
    let data = &snap.field_data;

    buf.resize(fw * fh * 4, 0);
    buf.fill(0);

    let (vmin, vmax) = snap
        .global_range
        .unwrap_or_else(|| data_range(data));
    let range = vmax - vmin;
    let inv_range = if range > 1e-30 { 1.0 / range } else { 1.0 };

    // Pre-compute longitude fractional indices for each column
    let lon_table: Vec<(usize, usize, f64)> = (0..dw)
        .map(|sx| {
            let lon_frac = (sx as f64 + 0.5) / dw as f64 * im as f64;
            let i0 = lon_frac.floor() as usize % im;
            let i1 = (i0 + 1) % im;
            let wx = lon_frac - lon_frac.floor();
            (i0, i1, wx)
        })
        .collect();

    for sy in 0..dh {
        let lat = PI / 2.0 - (sy as f64 + 0.5) / dh as f64 * PI;
        let mu = lat.sin();

        // Hoist gauss neighbor lookup per row (same lat for all columns)
        let (j0, j1, wy) = find_gauss_neighbors(gauss_nodes, mu);
        let j0_base = j0 * im;
        let j1_base = j1 * im;
        let wy_inv = 1.0 - wy;

        let row_off = sy * fw;
        for sx in 0..dw {
            let (i0, i1, wx) = lon_table[sx];
            let wx_inv = 1.0 - wx;

            let val = data[j0_base + i0] * wx_inv * wy_inv
                + data[j0_base + i1] * wx * wy_inv
                + data[j1_base + i0] * wx_inv * wy
                + data[j1_base + i1] * wx * wy;

            let t = (val - vmin) * inv_range;
            let rgba = color::map_to_rgba(t, colormap);

            let off = (row_off + sx) * 4;
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
    use crate::spherical::find_gauss_neighbors;

    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let fw = cfg.frame_width;
    let fh = cfg.frame_height;
    let im = snap.im;
    let data = &snap.field_data;

    buf.resize(fw * fh * 4, 0);
    buf.fill(0);

    let (vmin, vmax) = snap
        .global_range
        .unwrap_or_else(|| data_range(data));
    let range = vmax - vmin;
    let inv_range = if range > 1e-30 { 1.0 / range } else { 1.0 };

    let radius = (dw.min(dh) as f64) * 0.45;
    let inv_radius = 1.0 / radius;
    let cx = dw as f64 / 2.0;
    let cy = dh as f64 / 2.0;

    let sin_clat = cam_lat.sin();
    let cos_clat = cam_lat.cos();
    let sin_clon = cam_lon.sin();
    let cos_clon = cam_lon.cos();

    let bg = [8u8, 10, 20, 255];

    // Graticule threshold (constant for all pixels)
    let pixel_deg = 180.0 / (radius * 2.0).max(1.0);
    let threshold = pixel_deg * 0.8;

    // Precompute mu→gauss neighbor lookup table to avoid binary search per pixel.
    // 1024 entries covering mu ∈ [-1, 1].
    const MU_TABLE_SIZE: usize = 1024;
    let mu_table: Vec<(usize, usize)> = (0..MU_TABLE_SIZE)
        .map(|k| {
            let mu = -1.0 + 2.0 * (k as f64 + 0.5) / MU_TABLE_SIZE as f64;
            let (j0, j1, _) = find_gauss_neighbors(gauss_nodes, mu);
            (j0, j1)
        })
        .collect();

    let im_inv_2pi = im as f64 / (2.0 * PI);

    for sy in 0..dh {
        let ny_base = (cy - sy as f64) * inv_radius;
        let row_off = sy * fw;

        for sx in 0..dw {
            let off = (row_off + sx) * 4;

            let nx = (sx as f64 - cx) * inv_radius;
            let ny = ny_base;
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
            let x1 = nx;
            let y1 = ny * cos_clat + nz * sin_clat;
            let z1 = -ny * sin_clat + nz * cos_clat;

            let world_x = x1 * cos_clon + z1 * sin_clon;
            let world_z = -x1 * sin_clon + z1 * cos_clon;

            // mu = sin(lat) = y1 (avoids asin+sin round-trip)
            let mu = y1;

            // lon via atan2
            let lon = {
                let l = world_x.atan2(world_z);
                if l < 0.0 { l + 2.0 * PI } else { l }
            };

            // Inline interpolation using precomputed mu table
            let mu_idx = ((mu + 1.0) * (MU_TABLE_SIZE as f64 * 0.5)) as usize;
            let (j0, j1) = mu_table[mu_idx.min(MU_TABLE_SIZE - 1)];
            let wy = if j0 == j1 {
                0.0
            } else {
                ((mu - gauss_nodes[j0]) / (gauss_nodes[j1] - gauss_nodes[j0])).clamp(0.0, 1.0)
            };
            let wy_inv = 1.0 - wy;

            let lon_frac = lon * im_inv_2pi;
            let i0 = lon_frac as usize % im;
            let i1 = (i0 + 1) % im;
            let wx = lon_frac - lon_frac.floor();
            let wx_inv = 1.0 - wx;

            let j0_base = j0 * im;
            let j1_base = j1 * im;
            let val = data[j0_base + i0] * wx_inv * wy_inv
                + data[j0_base + i1] * wx * wy_inv
                + data[j1_base + i0] * wx_inv * wy
                + data[j1_base + i1] * wx * wy;

            let t = (val - vmin) * inv_range;
            let rgba = color::map_to_rgba(t, colormap);

            // Limb darkening
            let limb = nz.powf(0.3);

            let mut r = (rgba[0] as f64 * limb) as u8;
            let mut g = (rgba[1] as f64 * limb) as u8;
            let mut b = (rgba[2] as f64 * limb) as u8;

            // Graticule overlay (compute lat only when needed)
            let lat = y1.asin();
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
    font::draw_text_shadow(buf, fw, x, y, text, color);
}

/// Render a field-name badge at the top-left of the display area.
///
/// Delegates to `font::render_field_badge` with the display name resolved
/// from the raw field name.
pub fn render_field_badge(
    buf: &mut [u8],
    cfg: &SphericalRenderConfig,
    field_name: &str,
    field_index: usize,
    field_count: usize,
) {
    let display_name = field_display_name(field_name);
    font::render_field_badge(buf, cfg.frame_width, cfg.display_height, display_name, field_index, field_count);
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

        // Forward camera rotation: world → view
        let rx = x * cos_clon - z * sin_clon;
        let rz = x * sin_clon + z * cos_clon;
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

/// Outline + core offsets for a 3px-wide outlined track line.
/// First pass: dark border (±1 pixel around center).
/// Screen-blend a glow pixel: result = 1 - (1 - dst) * (1 - src * alpha).
/// Always brightens; never darkens the background.
fn glow_pixel(buf: &mut [u8], off: usize, color: [u8; 3], alpha: f64) {
    let r = buf[off] as f64 / 255.0;
    let g = buf[off + 1] as f64 / 255.0;
    let b = buf[off + 2] as f64 / 255.0;
    let sr = color[0] as f64 / 255.0 * alpha;
    let sg = color[1] as f64 / 255.0 * alpha;
    let sb = color[2] as f64 / 255.0 * alpha;
    buf[off]     = ((1.0 - (1.0 - r) * (1.0 - sr)) * 255.0) as u8;
    buf[off + 1] = ((1.0 - (1.0 - g) * (1.0 - sg)) * 255.0) as u8;
    buf[off + 2] = ((1.0 - (1.0 - b) * (1.0 - sb)) * 255.0) as u8;
}

/// Draw a Bresenham line with screen-blended glow (3px wide: core + 1px soft fringe).
fn draw_glow_bresenham(
    buf: &mut [u8],
    fw: usize,
    dw: usize,
    dh: usize,
    sx0: isize, sy0: isize,
    sx1: isize, sy1: isize,
    color: [u8; 3],
    intensity: f64,
) {
    let in_bounds = |x: isize, y: isize| -> bool {
        x >= 0 && x < dw as isize && y >= 0 && y < dh as isize
    };

    let dx = (sx1 - sx0).abs();
    let dy = -(sy1 - sy0).abs();
    let step_x: isize = if sx0 < sx1 { 1 } else { -1 };
    let step_y: isize = if sy0 < sy1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut cx = sx0;
    let mut cy = sy0;
    let max_steps = (dx.abs() + dy.abs() + 1) as usize;

    let fringe_alpha = intensity * 0.35;
    let core_alpha = intensity;

    for _ in 0..max_steps {
        // Soft fringe (±1 pixel)
        for &(ox, oy) in &[(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
            let px = cx + ox as isize;
            let py = cy + oy as isize;
            if in_bounds(px, py) {
                let off = (py as usize * fw + px as usize) * 4;
                if off + 3 < buf.len() {
                    glow_pixel(buf, off, color, fringe_alpha);
                }
            }
        }
        // Core pixel
        if in_bounds(cx, cy) {
            let off = (cy as usize * fw + cx as usize) * 4;
            if off + 3 < buf.len() {
                glow_pixel(buf, off, color, core_alpha);
            }
        }
        if cx == sx1 && cy == sy1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; cx += step_x; }
        if e2 <= dx { err += dx; cy += step_y; }
    }
}

/// Draw a bright marker dot with radial glow at the current gyre position.
fn draw_glow_marker(
    buf: &mut [u8],
    fw: usize,
    dw: usize,
    dh: usize,
    sx: isize,
    sy: isize,
    color: [u8; 3],
) {
    let in_bounds = |x: isize, y: isize| -> bool {
        x >= 0 && x < dw as isize && y >= 0 && y < dh as isize
    };

    // Radial glow (radius 5)
    let r_max = 5isize;
    let r_max_sq = (r_max * r_max) as f64;
    for dy in -r_max..=r_max {
        for ddx in -r_max..=r_max {
            let dist_sq = (ddx * ddx + dy * dy) as f64;
            if dist_sq <= r_max_sq {
                let px = sx + ddx;
                let py = sy + dy;
                if in_bounds(px, py) {
                    let off = (py as usize * fw + px as usize) * 4;
                    if off + 3 < buf.len() {
                        let alpha = (1.0 - dist_sq / r_max_sq).powi(2);
                        glow_pixel(buf, off, color, alpha);
                    }
                }
            }
        }
    }
}

/// Gradient color for gyre track: cyan (tail) → gold (head).
fn track_color(t: f64) -> [u8; 3] {
    // t = 0.0 (oldest) → 1.0 (newest)
    let r = (80.0 + 175.0 * t) as u8;
    let g = (200.0 + 55.0 * t) as u8;
    let b = (255.0 * (1.0 - t * 0.7)) as u8;
    [r, g, b]
}

/// Render gyre center trajectory on equirectangular projection.
/// `track` contains (lon, lat) pairs in radians, up to the current frame.
pub fn render_gyre_track_equirect(
    buf: &mut [u8],
    track: &[(f64, f64)],
    cfg: &SphericalRenderConfig,
    current_frame: usize,
) {
    if track.len() < 2 {
        return;
    }

    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let fw = cfg.frame_width;
    let n = track.len() - 1;
    let inv_n = if n > 0 { 1.0 / n as f64 } else { 1.0 };

    let to_screen = |lon: f64, lat: f64| -> (isize, isize) {
        let sx = (lon / (2.0 * PI) * dw as f64) as isize;
        let sy = ((PI / 2.0 - lat) / PI * dh as f64) as isize;
        (sx, sy)
    };

    for i in 0..track.len() - 1 {
        let (lon0, lat0) = track[i];
        let (lon1, lat1) = track[i + 1];

        // Skip segments that cross the date line (Δlon > π)
        if (lon1 - lon0).abs() > PI {
            continue;
        }

        let t = i as f64 * inv_n;
        let color = track_color(t);
        let intensity = 0.3 + 0.7 * t; // fade in from tail to head

        let (sx0, sy0) = to_screen(lon0, lat0);
        let (sx1, sy1) = to_screen(lon1, lat1);
        draw_glow_bresenham(buf, fw, dw, dh, sx0, sy0, sx1, sy1, color, intensity);
    }

    // Current position marker (bright gold)
    if let Some(&(lon, lat)) = track.get(current_frame) {
        let (sx, sy) = to_screen(lon, lat);
        draw_glow_marker(buf, fw, dw, dh, sx, sy, [255, 240, 100]);
    }
}

/// Render gyre center trajectory on orthographic projection.
pub fn render_gyre_track_ortho(
    buf: &mut [u8],
    track: &[(f64, f64)],
    cfg: &SphericalRenderConfig,
    cam_lat: f64,
    cam_lon: f64,
    current_frame: usize,
) {
    if track.len() < 2 {
        return;
    }

    let dw = cfg.display_width;
    let dh = cfg.display_height;
    let fw = cfg.frame_width;
    let radius = (dw.min(dh) as f64) * 0.45;
    let cx = dw as f64 / 2.0;
    let cy = dh as f64 / 2.0;
    let n = track.len() - 1;
    let inv_n = if n > 0 { 1.0 / n as f64 } else { 1.0 };

    let sin_clat = cam_lat.sin();
    let cos_clat = cam_lat.cos();
    let sin_clon = cam_lon.sin();
    let cos_clon = cam_lon.cos();

    let project = |lon: f64, lat: f64| -> Option<(isize, isize)> {
        let cos_lat = lat.cos();
        let sin_lat = lat.sin();
        let x = cos_lat * lon.sin();
        let y = sin_lat;
        let z = cos_lat * lon.cos();

        // Forward camera rotation: R_lat^-1 * R_lon^-1 * world
        let rx = x * cos_clon - z * sin_clon;
        let rz = x * sin_clon + z * cos_clon;
        let nx = rx;
        let ny = y * cos_clat - rz * sin_clat;
        let nz = y * sin_clat + rz * cos_clat;

        if nz < 0.0 {
            return None;
        }

        let screen_x = (cx + nx * radius) as isize;
        let screen_y = (cy - ny * radius) as isize;
        Some((screen_x, screen_y))
    };

    for i in 0..track.len() - 1 {
        let p0 = project(track[i].0, track[i].1);
        let p1 = project(track[i + 1].0, track[i + 1].1);

        if let (Some((sx0, sy0)), Some((sx1, sy1))) = (p0, p1) {
            let t = i as f64 * inv_n;
            let color = track_color(t);
            let intensity = 0.3 + 0.7 * t;
            draw_glow_bresenham(buf, fw, dw, dh, sx0, sy0, sx1, sy1, color, intensity);
        }
    }

    // Current position marker (bright gold)
    if let Some(&(lon, lat)) = track.get(current_frame) {
        if let Some((sx, sy)) = project(lon, lat) {
            draw_glow_marker(buf, fw, dw, dh, sx, sy, [255, 240, 100]);
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
