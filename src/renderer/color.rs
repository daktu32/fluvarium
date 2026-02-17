/// Selects which color palette to use for field rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorMap {
    /// Tokyo Night: navy -> blue -> purple -> pink -> orange (for RB).
    TokyoNight,
    /// Ocean & Lava: deep blue -> blue -> white -> orange -> red (for KH).
    OceanLava,
    /// Solar Wind: deep space -> indigo -> violet -> magenta -> plasma gold (for Karman).
    SolarWind,
    /// Arctic Ice: deep void -> teal -> cyan -> white -> bright mint (for Cavity).
    ArcticIce,
    /// Blue-White-Red: diverging colormap for signed data (vorticity etc).
    BlueWhiteRed,
}

/// Tokyo Night-inspired color stops for field mapping.
/// Deep navy -> blue -> purple -> pink -> orange
pub(crate) const COLOR_STOPS: [(f64, f64, f64); 5] = [
    (26.0, 27.0, 38.0),    // #1a1b26 navy         (0.00)
    (122.0, 162.0, 247.0), // #7aa2f7 blue         (0.25)
    (187.0, 154.0, 247.0), // #bb9af7 purple       (0.50)
    (247.0, 118.0, 142.0), // #f7768e pink         (0.75)
    (255.0, 158.0, 100.0), // #ff9e64 orange       (1.00)
];

/// Ocean & Lava color stops: deep blue -> blue -> near-white -> orange -> deep red.
/// Designed for KH instability: blue bottom fluid, red top fluid, white interface.
/// The 0.25/0.75 stops are kept dark so only the 0.5 midpoint glows white.
pub(crate) const OCEAN_LAVA_STOPS: [(f64, f64, f64); 5] = [
    (10.0, 30.0, 120.0),   // deep ocean blue      (0.00)
    (20.0, 90.0, 200.0),   // medium blue           (0.25)
    (250.0, 250.0, 240.0), // bright white interface (0.50)
    (220.0, 80.0, 10.0),   // medium orange          (0.75)
    (160.0, 20.0, 20.0),   // deep lava red         (1.00)
];

/// Solar Wind color stops: void -> indigo -> violet -> magenta -> plasma gold.
/// Designed for Karman vortex: dye wake glows like solar plasma against deep space.
pub(crate) const SOLAR_WIND_STOPS: [(f64, f64, f64); 5] = [
    (5.0, 5.0, 20.0),     // deep space void       (0.00)
    (20.0, 20.0, 80.0),   // dark indigo            (0.25)
    (80.0, 30.0, 180.0),  // violet nebula          (0.50)
    (220.0, 80.0, 160.0), // hot magenta            (0.75)
    (255.0, 220.0, 120.0),// solar plasma gold      (1.00)
];

/// Arctic Ice color stops: void -> teal -> cyan -> white -> bright mint.
/// Designed for Lid-Driven Cavity: velocity magnitude glows like flowing arctic currents.
pub(crate) const ARCTIC_ICE_STOPS: [(f64, f64, f64); 5] = [
    (8.0, 10.0, 25.0),    // deep void              (0.00)
    (10.0, 60.0, 90.0),   // dark teal              (0.25)
    (30.0, 180.0, 200.0), // bright cyan             (0.50)
    (200.0, 240.0, 250.0),// near white              (0.75)
    (120.0, 255.0, 200.0),// bright mint             (1.00)
];

/// Blue-White-Red diverging colormap: deep blue -> blue -> white -> red -> deep red.
/// Designed for signed data like vorticity: blue=negative, white=zero, red=positive.
pub(crate) const BLUE_WHITE_RED_STOPS: [(f64, f64, f64); 5] = [
    (10.0, 30.0, 150.0),  // deep blue              (0.00)
    (80.0, 130.0, 230.0), // medium blue            (0.25)
    (245.0, 245.0, 245.0),// near white             (0.50)
    (230.0, 100.0, 70.0), // medium red             (0.75)
    (150.0, 20.0, 20.0),  // deep red               (1.00)
];

/// Convert temperature [0.0, 1.0] to RGBA color (Tokyo Night palette).
#[cfg(test)]
pub fn temperature_to_rgba(t: f64) -> [u8; 4] {
    map_to_rgba(t, ColorMap::TokyoNight)
}

/// Convert a [0.0, 1.0] value to RGBA using the specified color map.
pub fn map_to_rgba(t: f64, colormap: ColorMap) -> [u8; 4] {
    let stops = match colormap {
        ColorMap::TokyoNight => &COLOR_STOPS,
        ColorMap::OceanLava => &OCEAN_LAVA_STOPS,
        ColorMap::SolarWind => &SOLAR_WIND_STOPS,
        ColorMap::ArcticIce => &ARCTIC_ICE_STOPS,
        ColorMap::BlueWhiteRed => &BLUE_WHITE_RED_STOPS,
    };

    let t = t.clamp(0.0, 1.0);
    let seg = t * 4.0;
    let i = (seg as usize).min(3);
    let s = seg - i as f64;

    let (r0, g0, b0) = stops[i];
    let (r1, g1, b1) = stops[i + 1];

    [
        (r0 + s * (r1 - r0)) as u8,
        (g0 + s * (g1 - g0)) as u8,
        (b0 + s * (b1 - b0)) as u8,
        255,
    ]
}

/// Color bar layout constants.
pub(crate) const BAR_GAP: usize = 6;
pub(crate) const BAR_WIDTH: usize = 20;
pub(crate) const TICK_LEN: usize = 4;
pub(crate) const LABEL_GAP: usize = 2;
pub(crate) const LABEL_WIDTH: usize = 24;
pub(crate) const BAR_TOTAL: usize = BAR_GAP + BAR_WIDTH + TICK_LEN + LABEL_GAP + LABEL_WIDTH;

#[cfg(test)]
mod tests {
    use super::*;

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

    // --- Ocean & Lava tests ---

    #[test]
    fn test_ocean_lava_cold_is_deep_blue() {
        let rgba = map_to_rgba(0.0, ColorMap::OceanLava);
        assert_eq!(rgba[0], 10, "R");
        assert_eq!(rgba[1], 30, "G");
        assert_eq!(rgba[2], 120, "B");
        assert_eq!(rgba[3], 255, "A");
    }

    #[test]
    fn test_ocean_lava_hot_is_deep_red() {
        let rgba = map_to_rgba(1.0, ColorMap::OceanLava);
        assert_eq!(rgba[0], 160, "R");
        assert_eq!(rgba[1], 20, "G");
        assert_eq!(rgba[2], 20, "B");
    }

    #[test]
    fn test_ocean_lava_mid_is_near_white() {
        let rgba = map_to_rgba(0.5, ColorMap::OceanLava);
        assert_eq!(rgba[0], 250, "R");
        assert_eq!(rgba[1], 250, "G");
        assert_eq!(rgba[2], 240, "B");
    }

    #[test]
    fn test_ocean_lava_gradient_continuity() {
        let steps = 256;
        for i in 1..steps {
            let t0 = (i - 1) as f64 / (steps - 1) as f64;
            let t1 = i as f64 / (steps - 1) as f64;
            let c0 = map_to_rgba(t0, ColorMap::OceanLava);
            let c1 = map_to_rgba(t1, ColorMap::OceanLava);
            for ch in 0..3 {
                let diff = (c1[ch] as i32 - c0[ch] as i32).abs();
                assert!(
                    diff <= 5,
                    "OceanLava channel {} jumped by {} between t={} and t={}",
                    ch, diff, t0, t1
                );
            }
        }
    }

    // --- Solar Wind tests ---

    #[test]
    fn test_solar_wind_cold_is_deep_space() {
        let rgba = map_to_rgba(0.0, ColorMap::SolarWind);
        assert_eq!(rgba[0], 5, "R");
        assert_eq!(rgba[1], 5, "G");
        assert_eq!(rgba[2], 20, "B");
    }

    #[test]
    fn test_solar_wind_hot_is_plasma_gold() {
        let rgba = map_to_rgba(1.0, ColorMap::SolarWind);
        assert_eq!(rgba[0], 255, "R");
        assert_eq!(rgba[1], 220, "G");
        assert_eq!(rgba[2], 120, "B");
    }

    #[test]
    fn test_solar_wind_gradient_continuity() {
        let steps = 256;
        for i in 1..steps {
            let t0 = (i - 1) as f64 / (steps - 1) as f64;
            let t1 = i as f64 / (steps - 1) as f64;
            let c0 = map_to_rgba(t0, ColorMap::SolarWind);
            let c1 = map_to_rgba(t1, ColorMap::SolarWind);
            for ch in 0..3 {
                let diff = (c1[ch] as i32 - c0[ch] as i32).abs();
                assert!(
                    diff <= 5,
                    "SolarWind channel {} jumped by {} between t={} and t={}",
                    ch, diff, t0, t1
                );
            }
        }
    }

    // --- Arctic Ice tests ---

    #[test]
    fn test_arctic_ice_cold_is_deep_void() {
        let rgba = map_to_rgba(0.0, ColorMap::ArcticIce);
        assert_eq!(rgba[0], 8, "R");
        assert_eq!(rgba[1], 10, "G");
        assert_eq!(rgba[2], 25, "B");
    }

    #[test]
    fn test_arctic_ice_hot_is_bright_mint() {
        let rgba = map_to_rgba(1.0, ColorMap::ArcticIce);
        assert_eq!(rgba[0], 120, "R");
        assert_eq!(rgba[1], 255, "G");
        assert_eq!(rgba[2], 200, "B");
    }

    #[test]
    fn test_arctic_ice_gradient_continuity() {
        let steps = 256;
        for i in 1..steps {
            let t0 = (i - 1) as f64 / (steps - 1) as f64;
            let t1 = i as f64 / (steps - 1) as f64;
            let c0 = map_to_rgba(t0, ColorMap::ArcticIce);
            let c1 = map_to_rgba(t1, ColorMap::ArcticIce);
            for ch in 0..3 {
                let diff = (c1[ch] as i32 - c0[ch] as i32).abs();
                assert!(
                    diff <= 5,
                    "ArcticIce channel {} jumped by {} between t={} and t={}",
                    ch, diff, t0, t1
                );
            }
        }
    }

    #[test]
    fn test_map_to_rgba_matches_temperature_to_rgba() {
        // map_to_rgba with TokyoNight should match temperature_to_rgba
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            assert_eq!(temperature_to_rgba(t), map_to_rgba(t, ColorMap::TokyoNight));
        }
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
}
