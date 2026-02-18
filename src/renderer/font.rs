use super::RenderConfig;

/// Status bar layout constants.
pub(crate) const FONT_WIDTH: usize = 5;
pub(crate) const FONT_HEIGHT: usize = 7;
pub(crate) const STATUS_PAD_TOP: usize = 3;
pub(crate) const STATUS_PAD_BOTTOM: usize = 2;
pub(crate) const STATUS_BAR_HEIGHT: usize = STATUS_PAD_TOP + FONT_HEIGHT + STATUS_PAD_BOTTOM;

/// 5x7 bitmap font glyph lookup. Each row is a u8 with lower 5 bits = pixels (bit4=left).
pub(crate) const fn glyph(ch: u8) -> [u8; FONT_HEIGHT] {
    match ch {
        b' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        b'.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00],
        b'-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        b'/' => [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
        b'>' => [0x10, 0x08, 0x04, 0x02, 0x04, 0x08, 0x10],
        b'=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        b'[' => [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E],
        b']' => [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E],
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

pub(crate) fn draw_char(buf: &mut [u8], frame_width: usize, x: usize, y: usize, ch: u8, color: [u8; 3]) {
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

/// Draw a character at (x, y) resized to target (cw x ch) pixels via nearest-neighbor.
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

/// Draw a string of text at (x, y) with each character sized to (cw x ch) pixels.
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
    let mut cx = 4 + cfg.display_x_offset; // left padding, aligned with display area
    for &ch in text.as_bytes() {
        if cx + FONT_WIDTH > fw {
            break;
        }
        draw_char(buf, fw, cx, text_y, ch, text_color);
        cx += char_step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::N;

    fn test_config() -> RenderConfig {
        RenderConfig::fit(542, 512, 3, N)
    }

    #[test]
    fn test_glyph_dash_arrow() {
        // '-' and '>' should have non-zero bitmaps
        let dash = glyph(b'-');
        let has_dash = dash.iter().any(|&row| row != 0);
        assert!(has_dash, "'-' glyph should have non-zero bits");

        let arrow = glyph(b'>');
        let has_arrow = arrow.iter().any(|&row| row != 0);
        assert!(has_arrow, "'>' glyph should have non-zero bits");
    }

    #[test]
    fn test_draw_text_returns_end_position() {
        let cfg = test_config();
        let mut buf = vec![0u8; cfg.frame_width * cfg.frame_height * 4];
        let color = [0xFF, 0xFF, 0xFF];
        let end_x = draw_text(&mut buf, cfg.frame_width, 10, 10, "hello", color);
        // "hello" = 5 chars, each FONT_WIDTH + 1 pixel spacing = 6 * 5 = 30
        let expected = 10 + 5 * (FONT_WIDTH + 1);
        assert_eq!(end_x, expected, "draw_text should return cursor position after text");

        // Verify some pixels were drawn (non-zero in the text area)
        let mut found = false;
        for y in 10..10 + FONT_HEIGHT {
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
}
