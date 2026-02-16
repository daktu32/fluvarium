use crate::renderer::{self, FONT_HEIGHT};
use crate::solver::SolverParams;
use crate::state::FluidModel;

/// Panel layout constants.
const GAUGE_WIDTH: usize = 8;

/// Overlay panel state.
pub struct OverlayState {
    pub visible: bool,
    pub selected: usize,
}

impl OverlayState {
    pub fn new() -> Self {
        Self {
            visible: false,
            selected: 0,
        }
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    pub fn navigate(&mut self, delta: isize, model: FluidModel) {
        let count = param_defs(model).len() as isize;
        self.selected = ((self.selected as isize + delta).rem_euclid(count)) as usize;
    }
}

/// Definition of an adjustable parameter.
pub struct ParamDef {
    pub name: &'static str,
    pub short: &'static str,
    pub desc: &'static str,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub fine_step: f64,
    pub default: f64,
    pub get: fn(&SolverParams) -> f64,
    pub set: fn(&mut SolverParams, f64),
}

/// Helper macro to reduce `ParamDef` boilerplate.
/// Usage: `param!(name, short, desc, field, min, max, step, fine_step, default)`
macro_rules! param {
    ($name:expr, $short:expr, $desc:expr, $field:ident, $min:expr, $max:expr, $step:expr, $fine:expr, $default:expr) => {
        ParamDef {
            name: $name,
            short: $short,
            desc: $desc,
            min: $min,
            max: $max,
            step: $step,
            fine_step: $fine,
            default: $default,
            get: |p| p.$field,
            set: |p, v| p.$field = v,
        }
    };
}

/// Rayleigh-Benard adjustable parameters.
const PARAM_DEFS_RB: [ParamDef; 7] = [
    param!("visc", "viscosity", "velocity viscosity coefficient",    visc,            0.0,  0.1,   0.001,  0.0001, 0.008),
    param!("diff", "diffusion", "thermal diffusion rate",            diff,            0.0,  0.05,  0.001,  0.0001, 0.002),
    param!("dt",   "timestep",  "simulation timestep",               dt,              0.0005, 0.02, 0.0005, 0.0001, 0.003),
    param!("buoy", "buoyancy",  "buoyancy force coefficient",        heat_buoyancy,   0.0,  50.0,  0.5,    0.1,    8.0),
    param!("src",  "heat src",  "heat source intensity",             source_strength, 0.0,  50.0,  0.5,    0.1,    10.0),
    param!("cool", "cooling",   "newtonian cooling rate",            cool_rate,       0.0,  30.0,  0.5,    0.1,    8.0),
    param!("base", "base temp", "bottom base temperature",           bottom_base,     0.0,  0.5,   0.01,   0.001,  0.15),
];

/// Karman vortex adjustable parameters.
const PARAM_DEFS_KARMAN: [ParamDef; 5] = [
    param!("visc", "viscosity", "velocity viscosity coefficient",    visc,        0.0005, 0.05, 0.0005, 0.0001, 0.015),
    param!("diff", "diffusion", "dye diffusion rate",                diff,        0.0,    0.01, 0.0005, 0.0001, 0.003),
    param!("dt",   "timestep",  "simulation timestep",               dt,          0.005,  0.1,  0.005,  0.001,  0.06),
    param!("u0",   "inflow",    "inflow velocity",                   inflow_vel,  0.01,   0.3,  0.01,   0.002,  0.1),
    param!("conf", "confine",   "vorticity confinement strength",    confinement, 0.0,    30.0, 0.5,    0.1,    3.0),
];

/// Kelvin-Helmholtz adjustable parameters.
const PARAM_DEFS_KH: [ParamDef; 7] = [
    param!("visc", "viscosity", "velocity viscosity coefficient",   visc,            0.0,    0.01,   0.0001, 0.00005, 0.001),
    param!("diff", "diffusion", "tracer diffusion rate",            diff,            0.0,    0.005,  0.0001, 0.00005, 0.0005),
    param!("dt",   "timestep",  "simulation timestep",              dt,              0.005,  0.2,    0.005,  0.001,   0.05),
    param!("shear","shear vel", "shear flow velocity magnitude",    shear_velocity,  0.01,   0.3,    0.005,  0.001,   0.08),
    param!("conf", "confine",   "vorticity confinement strength",   confinement,     0.0,    30.0,   0.5,    0.1,     0.0),
    param!("relax","relaxation","shear flow relaxation rate",       shear_relax,     0.0,    5.0,    0.1,    0.05,    1.0),
    param!("thick","thickness", "shear layer half-width in cells",  shear_thickness, 1.0,    10.0,   0.5,    0.1,     3.0),
];

/// Lid-Driven Cavity adjustable parameters.
const PARAM_DEFS_CAVITY: [ParamDef; 4] = [
    param!("visc", "viscosity", "velocity viscosity coefficient",  visc,          0.001,  0.1,   0.001,  0.0001,  0.01),
    param!("diff", "diffusion", "dye diffusion rate",              diff,          0.0,    0.01,  0.0005, 0.0001,  0.001),
    param!("dt",   "timestep",  "simulation timestep",             dt,            0.005,  0.2,   0.005,  0.001,   0.05),
    param!("lid",  "lid vel",   "lid (top wall) velocity",         lid_velocity,  0.1,    5.0,   0.1,    0.01,    1.0),
];

/// Get parameter definitions for the given fluid model.
pub fn param_defs(model: FluidModel) -> &'static [ParamDef] {
    match model {
        FluidModel::RayleighBenard => &PARAM_DEFS_RB,
        FluidModel::KarmanVortex => &PARAM_DEFS_KARMAN,
        FluidModel::KelvinHelmholtz => &PARAM_DEFS_KH,
        FluidModel::LidDrivenCavity => &PARAM_DEFS_CAVITY,
    }
}

/// Adjust a parameter by delta steps (positive = increase, negative = decrease).
/// If `fine` is true, use fine_step instead of step.
/// Returns true if the value actually changed.
pub fn adjust_param(params: &mut SolverParams, selected: usize, delta: i32, fine: bool, model: FluidModel) -> bool {
    let def = &param_defs(model)[selected];
    let old = (def.get)(params);
    let step = if fine { def.fine_step } else { def.step };
    let new_val = (old + delta as f64 * step).clamp(def.min, def.max);
    (def.set)(params, new_val);
    (new_val - old).abs() > f64::EPSILON
}

/// Reset a parameter to its default value.
pub fn reset_param(params: &mut SolverParams, selected: usize, model: FluidModel) {
    let def = &param_defs(model)[selected];
    (def.set)(params, def.default);
}

/// Colors used in the overlay panel.
mod colors {
    pub const BORDER: [u8; 3] = [0x44, 0x44, 0x44];
    pub const HEADER: [u8; 3] = [0x00, 0xBB, 0xBB];
    pub const LABEL_NORMAL: [u8; 3] = [0x88, 0x88, 0x88];
    pub const LABEL_SELECTED: [u8; 3] = [0xFF, 0xFF, 0xFF];
    pub const VALUE: [u8; 3] = [0xCC, 0xCC, 0xCC];
    pub const DESC_NORMAL: [u8; 3] = [0x66, 0x66, 0x66];
    pub const DESC_SELECTED: [u8; 3] = [0xAA, 0xAA, 0xAA];
    pub const HINT: [u8; 3] = [0x44, 0x88, 0x88];
    pub const CURSOR: [u8; 3] = [0x00, 0xFF, 0xFF];
}

/// Darken a rectangular region of the buffer by multiplying RGB by `factor`.
fn darken_rect(buf: &mut [u8], frame_width: usize, x0: usize, y0: usize, w: usize, h: usize, factor: f64) {
    for dy in 0..h {
        let y = y0 + dy;
        for dx in 0..w {
            let x = x0 + dx;
            let off = (y * frame_width + x) * 4;
            if off + 3 < buf.len() {
                buf[off] = (buf[off] as f64 * factor) as u8;
                buf[off + 1] = (buf[off + 1] as f64 * factor) as u8;
                buf[off + 2] = (buf[off + 2] as f64 * factor) as u8;
            }
        }
    }
}

/// Draw a 1px border rectangle.
fn draw_rect_border(buf: &mut [u8], frame_width: usize, x0: usize, y0: usize, w: usize, h: usize, color: [u8; 3]) {
    // Top and bottom edges
    for dx in 0..w {
        for &y in &[y0, y0 + h - 1] {
            let off = (y * frame_width + x0 + dx) * 4;
            if off + 3 < buf.len() {
                buf[off] = color[0];
                buf[off + 1] = color[1];
                buf[off + 2] = color[2];
                buf[off + 3] = 255;
            }
        }
    }
    // Left and right edges
    for dy in 0..h {
        for &x in &[x0, x0 + w - 1] {
            let off = ((y0 + dy) * frame_width + x) * 4;
            if off + 3 < buf.len() {
                buf[off] = color[0];
                buf[off + 1] = color[1];
                buf[off + 2] = color[2];
                buf[off + 3] = 255;
            }
        }
    }
}

/// Draw a gauge bar with teal gradient fill at custom pixel dimensions.
fn draw_gauge_scaled(buf: &mut [u8], frame_width: usize, x: usize, y: usize, ratio: f64, width_chars: usize, char_step: usize, height: usize) {
    let total_px = width_chars * char_step;
    let filled_px = ((ratio * total_px as f64).round() as usize).min(total_px);

    for dy in 0..height {
        for dx in 0..total_px {
            let off = ((y + dy) * frame_width + x + dx) * 4;
            if off + 3 < buf.len() {
                if dx < filled_px {
                    let t = dx as f64 / total_px as f64;
                    buf[off] = 0;
                    buf[off + 1] = (0x55 as f64 + t * (0xCC - 0x55) as f64) as u8;
                    buf[off + 2] = (0x55 as f64 + t * (0xCC - 0x55) as f64) as u8;
                } else {
                    buf[off] = 0x22;
                    buf[off + 1] = 0x22;
                    buf[off + 2] = 0x22;
                }
                buf[off + 3] = 255;
            }
        }
    }
}

/// Render the overlay panel onto the frame buffer.
/// Does nothing if `state.visible` is false.
pub fn render_overlay(
    buf: &mut [u8],
    frame_width: usize,
    display_width: usize,
    display_height: usize,
    state: &OverlayState,
    params: &SolverParams,
    model: FluidModel,
) {
    if !state.visible {
        return;
    }

    let defs = param_defs(model);
    let param_count = defs.len();

    // Font: 2/3 of 2x → 7×9 pixels (nearest-neighbor resize from 5×7)
    let cw: usize = 7;
    let ch: usize = 9;
    let sc = cw + cw / 5 + 1;          // char step = 9px (proportional spacing)
    let row_h = ch + 4;                // row height = 13px
    let pad = 10;                       // inner padding

    // Compute panel width from content
    // "> visc    ████████  0.008  viscosity"
    // 2 + 6(pad to 8) + 8(gauge) + 1 + 6(val) + 1 + 9(short) = 33 chars
    let content_chars = 35;
    let panel_w = content_chars * sc + pad * 2;

    // Panel height
    let panel_h = pad
        + row_h                         // header
        + 4                             // gap after header
        + param_count * row_h           // param rows
        + 6                             // gap
        + row_h                         // description
        + 4                             // gap
        + (FONT_HEIGHT + 2)             // hints at 1x
        + pad;

    // Center the panel (clamp to display area)
    let panel_w = panel_w.min(display_width.saturating_sub(4));
    let panel_h = panel_h.min(display_height.saturating_sub(4));
    let px = display_width.saturating_sub(panel_w) / 2;
    let py = display_height.saturating_sub(panel_h).saturating_sub(8);

    // Darken background
    darken_rect(buf, frame_width, px, py, panel_w, panel_h, 0.25);

    // Border
    draw_rect_border(buf, frame_width, px, py, panel_w, panel_h, colors::BORDER);

    let left = px + pad;
    let mut cy = py + pad;

    // Header
    let header = match model {
        FluidModel::RayleighBenard => "rayleigh benard".to_string(),
        FluidModel::KarmanVortex => {
            let re = params.inflow_vel * (params.cylinder_radius * 2.0) / params.visc;
            format!("karman vortex  re={:.0}", re)
        }
        FluidModel::KelvinHelmholtz => "kelvin helmholtz".to_string(),
        FluidModel::LidDrivenCavity => {
            let re = params.lid_velocity * (crate::state::N as f64) / params.visc;
            format!("lid-driven cavity  re={:.0}", re)
        }
    };
    renderer::draw_text_sized(buf, frame_width, left, cy, &header, colors::HEADER, cw, ch);
    cy += row_h + 4;

    // Parameter rows
    for (i, def) in defs.iter().enumerate() {
        let is_sel = i == state.selected;
        let label_color = if is_sel { colors::LABEL_SELECTED } else { colors::LABEL_NORMAL };
        let desc_color = if is_sel { colors::DESC_SELECTED } else { colors::DESC_NORMAL };

        // Cursor ">"
        let mut cx = left;
        if is_sel {
            renderer::draw_text_sized(buf, frame_width, cx, cy, ">", colors::CURSOR, cw, ch);
        }
        cx += 2 * sc;

        // Name (4 chars, padded to column 8)
        renderer::draw_text_sized(buf, frame_width, cx, cy, def.name, label_color, cw, ch);
        cx = left + 8 * sc;

        // Gauge bar
        let val = (def.get)(params);
        let ratio = if (def.max - def.min).abs() > f64::EPSILON {
            ((val - def.min) / (def.max - def.min)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        draw_gauge_scaled(buf, frame_width, cx, cy, ratio, GAUGE_WIDTH, sc, ch);
        cx += GAUGE_WIDTH * sc + sc;

        // Value
        let val_str = if def.step >= 0.1 {
            format!("{:.2}", val)
        } else if def.step >= 0.001 {
            format!("{:.3}", val)
        } else {
            format!("{:.4}", val)
        };
        cx = renderer::draw_text_sized(buf, frame_width, cx, cy, &val_str, colors::VALUE, cw, ch);
        cx += sc;

        // Short description
        renderer::draw_text_sized(buf, frame_width, cx, cy, def.short, desc_color, cw, ch);

        cy += row_h;
    }

    cy += 6;

    // Selected parameter description
    let sel_def = &defs[state.selected];
    renderer::draw_text_sized(buf, frame_width, left, cy, sel_def.desc, colors::DESC_SELECTED, cw, ch);
    cy += row_h + 4;

    // Key hints (1x — smaller for visual hierarchy)
    renderer::draw_text(
        buf,
        frame_width,
        left,
        cy,
        "space=close  ud=nav  lr=adj  ,.=fine  r=reset",
        colors::HINT,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::{FONT_HEIGHT, FONT_WIDTH};
    use crate::solver::SolverParams;

    const CHAR_STEP: usize = FONT_WIDTH + 1;

    #[test]
    fn test_overlay_toggle() {
        let mut state = OverlayState::new();
        assert!(!state.visible);
        state.toggle();
        assert!(state.visible);
        state.toggle();
        assert!(!state.visible);
    }

    #[test]
    fn test_navigate_wraps_rb() {
        let mut state = OverlayState::new();
        let model = FluidModel::RayleighBenard;
        let count = param_defs(model).len();
        assert_eq!(state.selected, 0);
        state.navigate(-1, model);
        assert_eq!(state.selected, count - 1, "Should wrap to last");
        state.navigate(1, model);
        assert_eq!(state.selected, 0, "Should wrap back to first");
    }

    #[test]
    fn test_navigate_wraps_karman() {
        let mut state = OverlayState::new();
        let model = FluidModel::KarmanVortex;
        let count = param_defs(model).len();
        assert_eq!(count, 5);
        state.navigate(-1, model);
        assert_eq!(state.selected, 4, "Should wrap to last Karman param");
        state.navigate(1, model);
        assert_eq!(state.selected, 0, "Should wrap back to first");
    }

    #[test]
    fn test_param_get_set_roundtrip() {
        let mut params = SolverParams::default();
        let defs = param_defs(FluidModel::RayleighBenard);
        for (i, def) in defs.iter().enumerate() {
            let orig = (def.get)(&params);
            let new_val = (def.min + def.max) / 2.0;
            (def.set)(&mut params, new_val);
            let read_back = (defs[i].get)(&params);
            assert!(
                (read_back - new_val).abs() < 1e-10,
                "Param {} get/set roundtrip failed",
                def.name
            );
            (def.set)(&mut params, orig); // restore
        }
    }

    #[test]
    fn test_param_defaults_match_solver_rb() {
        let defaults = SolverParams::default();
        for def in param_defs(FluidModel::RayleighBenard) {
            let solver_val = (def.get)(&defaults);
            assert!(
                (solver_val - def.default).abs() < 1e-10,
                "PARAM_DEFS_RB.default for {} ({}) doesn't match SolverParams::default() ({})",
                def.name,
                def.default,
                solver_val
            );
        }
    }

    #[test]
    fn test_param_defaults_match_solver_karman() {
        let defaults = SolverParams::default_karman();
        for def in param_defs(FluidModel::KarmanVortex) {
            let solver_val = (def.get)(&defaults);
            assert!(
                (solver_val - def.default).abs() < 1e-10,
                "PARAM_DEFS_KARMAN.default for {} ({}) doesn't match SolverParams::default_karman() ({})",
                def.name,
                def.default,
                solver_val
            );
        }
    }

    #[test]
    fn test_darken_reduces_brightness() {
        let w = 10;
        let h = 10;
        let mut buf = vec![128u8; w * h * 4]; // all channels at 128
        // Set alpha to 255
        for i in 0..w * h {
            buf[i * 4 + 3] = 255;
        }

        darken_rect(&mut buf, w, 2, 2, 4, 4, 0.25);

        // Darkened area should be ~32
        let off = (3 * w + 3) * 4;
        assert!(buf[off] < 40, "R should be darkened: got {}", buf[off]);
        assert!(buf[off + 1] < 40, "G should be darkened: got {}", buf[off + 1]);

        // Outside darkened area should be unchanged
        let off2 = (0 * w + 0) * 4;
        assert_eq!(buf[off2], 128, "Outside area should be unchanged");
    }

    #[test]
    fn test_gauge_empty_full() {
        let w = 200;
        let h = 20;
        let mut buf_empty = vec![0u8; w * h * 4];
        let mut buf_full = vec![0u8; w * h * 4];

        draw_gauge_scaled(&mut buf_empty, w, 4, 4, 0.0, GAUGE_WIDTH, CHAR_STEP, FONT_HEIGHT);
        draw_gauge_scaled(&mut buf_full, w, 4, 4, 1.0, GAUGE_WIDTH, CHAR_STEP, FONT_HEIGHT);

        // Empty gauge: all pixels should be dark (#222222)
        let off = (4 * w + 4) * 4;
        assert_eq!(buf_empty[off], 0x22, "Empty gauge should be #22 at start");

        // Full gauge: first pixel should have teal color (G/B > 0x22)
        assert!(buf_full[off + 1] > 0x22, "Full gauge should have teal fill");
    }

    #[test]
    fn test_overlay_invisible_noop() {
        let cfg = crate::renderer::RenderConfig::fit(542, 512, 3, crate::state::N);
        let mut buf = vec![42u8; cfg.frame_width * cfg.frame_height * 4];
        let orig = buf.clone();
        let state = OverlayState::new(); // visible = false
        let params = SolverParams::default();

        render_overlay(&mut buf, cfg.frame_width, cfg.display_width, cfg.display_height, &state, &params, FluidModel::RayleighBenard);

        assert_eq!(buf, orig, "Invisible overlay should not modify buffer");
    }

    #[test]
    fn test_adjust_clamps() {
        let mut params = SolverParams::default();
        let model = FluidModel::RayleighBenard;

        // Try to decrease visc below min (0.0)
        params.visc = 0.0;
        let changed = adjust_param(&mut params, 0, -1, false, model);
        assert!(!changed, "Should not change when at min");
        assert!((params.visc - 0.0).abs() < f64::EPSILON, "visc should stay at 0.0");

        // Try to increase visc above max (0.1)
        params.visc = 0.1;
        let changed = adjust_param(&mut params, 0, 1, false, model);
        assert!(!changed, "Should not change when at max");
        assert!((params.visc - 0.1).abs() < f64::EPSILON, "visc should stay at 0.1");
    }

    #[test]
    fn test_reset_restores_default() {
        let mut params = SolverParams::default();
        params.visc = 0.05;
        reset_param(&mut params, 0, FluidModel::RayleighBenard);
        assert!(
            (params.visc - 0.008).abs() < 1e-10,
            "visc should be reset to default 0.008, got {}",
            params.visc
        );
    }

    #[test]
    fn test_qa_param_defaults_match_solver_kh() {
        let defaults = SolverParams::default_kh();
        for def in param_defs(FluidModel::KelvinHelmholtz) {
            let solver_val = (def.get)(&defaults);
            assert!(
                (solver_val - def.default).abs() < 1e-10,
                "PARAM_DEFS_KH.default for {} ({}) doesn't match SolverParams::default_kh() ({})",
                def.name,
                def.default,
                solver_val
            );
        }
    }

    #[test]
    fn test_qa_navigate_wraps_kh() {
        let mut state = OverlayState::new();
        let model = FluidModel::KelvinHelmholtz;
        let count = param_defs(model).len();
        assert_eq!(count, 7, "KH should have 7 params");
        assert_eq!(state.selected, 0);
        state.navigate(-1, model);
        assert_eq!(state.selected, count - 1, "Should wrap to last KH param");
        state.navigate(1, model);
        assert_eq!(state.selected, 0, "Should wrap back to first");
    }
}
