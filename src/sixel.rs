use std::io::{self, Write};

use crate::renderer::temperature_to_rgba;

/// Encode RGBA buffer to Sixel format using icy_sixel.
#[cfg(test)]
pub fn encode_sixel(rgba: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
    // icy_sixel expects a flat pixel buffer
    let sixel_output = icy_sixel::sixel_string(
        rgba,
        width as i32,
        height as i32,
        icy_sixel::PixelFormat::RGBA8888,
        icy_sixel::DiffusionMethod::None,
        icy_sixel::MethodForLargest::Auto,
        icy_sixel::MethodForRep::Auto,
        icy_sixel::Quality::LOW,
    )
    .map_err(|e| format!("Sixel encoding error: {}", e))?;

    Ok(sixel_output.into_bytes())
}

/// Output a Sixel-encoded frame to stdout.
/// Uses synchronized output (DEC 2026) and single write to minimize flicker.
pub fn output_frame(sixel_data: &[u8]) -> io::Result<()> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let mut buf = Vec::with_capacity(10 + 3 + sixel_data.len() + 10);
    buf.extend_from_slice(b"\x1b[?2026h"); // begin synchronized update
    buf.extend_from_slice(b"\x1b[H"); // cursor home
    buf.extend_from_slice(sixel_data);
    buf.extend_from_slice(b"\x1b[?2026l"); // end synchronized update
    handle.write_all(&buf)?;
    handle.flush()?;
    Ok(())
}

/// Number of palette colors: gradient colors + 1 white for tick marks.
const PALETTE_SIZE: usize = 64;

/// Fast Sixel encoder with a fixed palette.
///
/// Palette: (PALETTE_SIZE-1) temperature gradient colors + 1 white (for tick marks).
/// Uses a 32×32×32 RGB→palette LUT for O(1) per-pixel color mapping,
/// bypassing the expensive color quantization that icy_sixel performs.
pub struct SixelEncoder {
    palette_def: Vec<u8>,
    lut: Vec<u8>,
}

impl SixelEncoder {
    pub fn new() -> Self {
        let grad_colors = PALETTE_SIZE - 1; // gradient entries
        let mut palette_rgb = Vec::with_capacity(PALETTE_SIZE);
        for i in 0..grad_colors as u32 {
            let t = i as f64 / (grad_colors - 1) as f64;
            let rgba = temperature_to_rgba(t);
            palette_rgb.push([rgba[0], rgba[1], rgba[2]]);
        }
        palette_rgb.push([255u8, 255, 255]); // last index = white

        // Pre-render palette definition string
        let mut palette_def = Vec::with_capacity(2048);
        for (idx, &[r, g, b]) in palette_rgb.iter().enumerate() {
            let _ = write!(
                palette_def,
                "#{};2;{};{};{}",
                idx,
                r as u32 * 100 / 255,
                g as u32 * 100 / 255,
                b as u32 * 100 / 255
            );
        }

        // Build 32×32×32 RGB → palette index lookup table
        let mut lut = vec![0u8; 32 * 32 * 32];
        for ri in 0u32..32 {
            for gi in 0u32..32 {
                for bi in 0u32..32 {
                    let r = (ri * 255 / 31) as i32;
                    let g = (gi * 255 / 31) as i32;
                    let b = (bi * 255 / 31) as i32;
                    let mut best_idx = 0u8;
                    let mut best_dist = i32::MAX;
                    for (idx, &[pr, pg, pb]) in palette_rgb.iter().enumerate() {
                        let dr = r - pr as i32;
                        let dg = g - pg as i32;
                        let db = b - pb as i32;
                        let dist = dr * dr + dg * dg + db * db;
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = idx as u8;
                        }
                    }
                    lut[(ri * 1024 + gi * 32 + bi) as usize] = best_idx;
                }
            }
        }

        Self { palette_def, lut }
    }

    #[inline]
    fn lookup(&self, r: u8, g: u8, b: u8) -> u8 {
        self.lut[((r as usize) >> 3 << 10) | ((g as usize) >> 3 << 5) | ((b as usize) >> 3)]
    }

    /// Encode RGBA buffer to Sixel with optional top padding for vertical centering.
    /// `top_pad` is the number of blank pixel rows to prepend.
    pub fn encode(&self, rgba: &[u8], width: usize, height: usize, top_pad: usize) -> Vec<u8> {
        let num_pixels = width * height;

        // Map every pixel to a palette index via LUT
        let mut indices = vec![0u8; num_pixels];
        for i in 0..num_pixels {
            let off = i * 4;
            indices[i] = self.lookup(rgba[off], rgba[off + 1], rgba[off + 2]);
        }

        let mut out = Vec::with_capacity(num_pixels);

        // DCS header + raster attributes (total height includes padding)
        let total_height = height + top_pad;
        let _ = write!(out, "\x1bP0;0;0q\"1;1;{};{}", width, total_height);

        // Pre-rendered palette definitions
        out.extend_from_slice(&self.palette_def);

        // Empty bands for top padding
        let pad_bands = (top_pad + 5) / 6;
        for _ in 0..pad_bands {
            out.push(b'-');
        }

        // Encode sixel bands (6 rows each)
        let num_bands = (height + 5) / 6;
        for band in 0..num_bands {
            let y_start = band * 6;
            let y_end = (y_start + 6).min(height);
            let band_rows = y_end - y_start;

            // Find which colors appear in this band
            let mut used = [false; PALETTE_SIZE];
            for y in y_start..y_end {
                for &idx in &indices[y * width..(y + 1) * width] {
                    used[idx as usize] = true;
                }
            }

            let mut first_color = true;
            for color in 0..PALETTE_SIZE {
                if !used[color] {
                    continue;
                }

                if !first_color {
                    out.push(b'$'); // carriage return within band
                }
                first_color = false;

                // Select this color
                let _ = write!(out, "#{}", color);

                // Emit sixel characters across all columns with RLE
                let mut run_char = 0u8;
                let mut run_len = 0u32;

                for x in 0..width {
                    let mut bits = 0u8;
                    for dy in 0..band_rows {
                        if indices[(y_start + dy) * width + x] as usize == color {
                            bits |= 1 << dy;
                        }
                    }
                    let ch = 63 + bits;

                    if ch == run_char {
                        run_len += 1;
                    } else {
                        flush_run(&mut out, run_char, run_len);
                        run_char = ch;
                        run_len = 1;
                    }
                }
                flush_run(&mut out, run_char, run_len);
            }

            // Move to next band (except after last)
            if band < num_bands - 1 {
                out.push(b'-');
            }
        }

        // String Terminator
        out.extend_from_slice(b"\x1b\\");
        out
    }
}

#[inline]
fn flush_run(out: &mut Vec<u8>, ch: u8, len: u32) {
    if len == 0 {
        return;
    }
    if len >= 4 {
        let _ = write!(out, "!{}{}", len, ch as char);
    } else {
        for _ in 0..len {
            out.push(ch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_sixel_not_empty() {
        // Create a small 8x6 red image
        let width = 8;
        let height = 6;
        let mut rgba = vec![0u8; width * height * 4];
        for i in 0..(width * height) {
            rgba[i * 4] = 255; // R
            rgba[i * 4 + 1] = 0; // G
            rgba[i * 4 + 2] = 0; // B
            rgba[i * 4 + 3] = 255; // A
        }

        let result = encode_sixel(&rgba, width, height);
        assert!(result.is_ok(), "Encoding should succeed");
        let data = result.unwrap();
        assert!(!data.is_empty(), "Output should not be empty");
    }

    #[test]
    fn test_sixel_dcs_header() {
        let width = 8;
        let height = 6;
        let rgba = vec![128u8; width * height * 4];

        let data = encode_sixel(&rgba, width, height).unwrap();
        let s = String::from_utf8_lossy(&data);

        // Sixel data should start with DCS (ESC P or \x90)
        assert!(
            s.starts_with("\x1bP") || s.starts_with("\u{0090}") || data[0] == 0x90,
            "Should start with DCS header, got: {:?}",
            &data[..data.len().min(10)]
        );
    }

    #[test]
    fn test_sixel_st_footer() {
        let width = 8;
        let height = 6;
        let rgba = vec![128u8; width * height * 4];

        let data = encode_sixel(&rgba, width, height).unwrap();
        let s = String::from_utf8_lossy(&data);

        // Sixel data should end with ST (ESC \ or \x9c)
        assert!(
            s.ends_with("\x1b\\") || s.ends_with("\u{009c}") || *data.last().unwrap() == 0x9c,
            "Should end with ST footer, got: {:?}",
            &data[data.len().saturating_sub(10)..]
        );
    }

    #[test]
    fn test_custom_encoder_dcs_header() {
        let encoder = SixelEncoder::new();
        let width = 8;
        let height = 6;
        let rgba = vec![128u8; width * height * 4];
        let data = encoder.encode(&rgba, width, height, 0);
        assert!(data.starts_with(b"\x1bP"), "Should start with DCS header");
    }

    #[test]
    fn test_custom_encoder_st_footer() {
        let encoder = SixelEncoder::new();
        let width = 8;
        let height = 6;
        let rgba = vec![128u8; width * height * 4];
        let data = encoder.encode(&rgba, width, height, 0);
        assert!(data.ends_with(b"\x1b\\"), "Should end with ST footer");
    }

    #[test]
    fn test_custom_encoder_not_empty() {
        let encoder = SixelEncoder::new();
        let width = 16;
        let height = 12;
        let rgba = vec![100u8; width * height * 4];
        let data = encoder.encode(&rgba, width, height, 0);
        assert!(data.len() > 100, "Encoded data should have substantial content");
    }

    #[test]
    fn test_custom_encoder_lut_deterministic() {
        let encoder = SixelEncoder::new();
        // Same color should always map to same index
        let idx1 = encoder.lookup(200, 100, 50);
        let idx2 = encoder.lookup(200, 100, 50);
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_custom_encoder_white_maps_to_last() {
        let encoder = SixelEncoder::new();
        let idx = encoder.lookup(255, 255, 255);
        assert_eq!(idx, (PALETTE_SIZE - 1) as u8, "Pure white should map to last palette index");
    }
}
