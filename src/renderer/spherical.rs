// Spherical data renderer: equirectangular and orthographic projections.

use super::color;
use super::font;
use crate::spherical::{interpolate_gauss, SphericalSnapshot};
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

    // Compute data range for color mapping
    let (vmin, vmax) = data_range(&snap.field_data);
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

    let (vmin, vmax) = data_range(&snap.field_data);
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

            buf[off] = (rgba[0] as f64 * limb) as u8;
            buf[off + 1] = (rgba[1] as f64 * limb) as u8;
            buf[off + 2] = (rgba[2] as f64 * limb) as u8;
            buf[off + 3] = 255;
        }
    }

    render_color_bar(buf, cfg, colormap, vmin, vmax, &snap.field_name);
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

    // Field name label
    let name_y = 2.min(dh.saturating_sub(font::FONT_HEIGHT));
    let name_w = field_name.len() * (font::FONT_WIDTH + 1);
    let name_x = if bar_x + bar_width / 2 >= name_w / 2 {
        bar_x + bar_width / 2 - name_w / 2
    } else {
        bar_x
    };
    font::draw_text(buf, fw, name_x, name_y, field_name, [0xAA, 0xAA, 0xAA]);
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
