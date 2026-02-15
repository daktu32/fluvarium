/// iTerm2 Graphics Protocol encoder for headless terminal rendering.
///
/// Encodes RGBA pixel buffers into iTerm2 inline image escape sequences
/// using a custom uncompressed PNG encoder (stored deflate) for real-time
/// performance — ~5-10x faster than png crate's Compression::Fast.

// --- PNG CRC-32 lookup table (compile-time) ---

const fn make_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            if c & 1 != 0 {
                c = 0xEDB88320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
}

static CRC_TABLE: [u32; 256] = make_crc_table();

fn png_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc = CRC_TABLE[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Adler-32 checksum for the zlib wrapper inside PNG IDAT.
fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    const NMAX: usize = 5552; // max bytes before modulo overflow
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for chunk in data.chunks(NMAX) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

/// Write a PNG chunk: [length(4)][type(4)][data][crc32(4)].
fn write_png_chunk(buf: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let crc_start = buf.len();
    buf.extend_from_slice(chunk_type);
    buf.extend_from_slice(data);
    let crc = png_crc32(&buf[crc_start..]);
    buf.extend_from_slice(&crc.to_be_bytes());
}

/// Encode RGBA pixels as an uncompressed PNG (stored deflate blocks).
///
/// Bypasses zlib/deflate compression entirely — just wraps raw pixel data
/// in the deflate "stored" block format. This is ~5-10x faster than even
/// `Compression::Fast` because there's zero hash-table lookups or LZ77
/// matching. The trade-off is ~2x larger file size, but for real-time
/// terminal rendering the CPU savings far outweigh the extra I/O.
fn encode_png_stored(
    rgba: &[u8],
    width: usize,
    height: usize,
    raw_buf: &mut Vec<u8>,
    png_buf: &mut Vec<u8>,
) {
    let row_payload = width * 4;
    let row_len = 1 + row_payload; // filter byte + RGBA
    let raw_len = row_len * height;

    // Build raw image data: [0x00(NoFilter) R G B A ...] per row.
    // RGBA avoids per-pixel RGB strip — entire row is a single bulk memcpy.
    raw_buf.clear();
    raw_buf.reserve(raw_len);
    for y in 0..height {
        raw_buf.push(0); // NoFilter
        let row_start = y * row_payload;
        raw_buf.extend_from_slice(&rgba[row_start..row_start + row_payload]);
    }

    let adler = adler32(raw_buf);

    // Pre-compute IDAT data length for chunk header
    let num_blocks = if raw_len == 0 { 1 } else { (raw_len + 65534) / 65535 };
    let idat_len = 2 + num_blocks * 5 + raw_len + 4; // zlib_hdr + block_hdrs + data + adler32
    let total = 8 + 25 + (12 + idat_len) + 12; // sig + IHDR + IDAT + IEND

    png_buf.clear();
    png_buf.reserve(total);

    // PNG signature
    png_buf.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    // IHDR chunk
    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&(width as u32).to_be_bytes());
    ihdr[4..8].copy_from_slice(&(height as u32).to_be_bytes());
    ihdr[8] = 8; // bit depth
    ihdr[9] = 6; // color type: RGBA
    write_png_chunk(png_buf, b"IHDR", &ihdr);

    // IDAT chunk — written inline to avoid a separate buffer
    png_buf.extend_from_slice(&(idat_len as u32).to_be_bytes());
    let crc_start = png_buf.len();
    png_buf.extend_from_slice(b"IDAT");

    // Zlib header: CMF=0x78 (deflate, 32K window), FLG=0x01 (check bits)
    png_buf.extend_from_slice(&[0x78, 0x01]);

    // Stored deflate blocks (max 65535 bytes each)
    let mut offset = 0;
    let mut remaining = raw_len;
    loop {
        let block_len = remaining.min(65535);
        let is_last = remaining <= 65535;
        png_buf.push(if is_last { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00
        png_buf.extend_from_slice(&(block_len as u16).to_le_bytes()); // LEN
        png_buf.extend_from_slice(&(!(block_len as u16)).to_le_bytes()); // NLEN
        png_buf.extend_from_slice(&raw_buf[offset..offset + block_len]);
        offset += block_len;
        remaining -= block_len;
        if is_last {
            break;
        }
    }

    // Adler-32 (big-endian)
    png_buf.extend_from_slice(&adler.to_be_bytes());

    // CRC over chunk type + data
    let crc = png_crc32(&png_buf[crc_start..]);
    png_buf.extend_from_slice(&crc.to_be_bytes());

    // IEND chunk
    write_png_chunk(png_buf, b"IEND", &[]);
}

pub struct Iterm2Encoder {
    png_buf: Vec<u8>,
    raw_buf: Vec<u8>,
    b64_buf: String,
    seq_buf: Vec<u8>,
}

impl Iterm2Encoder {
    pub fn new() -> Self {
        Self {
            png_buf: Vec::new(),
            raw_buf: Vec::new(),
            b64_buf: String::new(),
            seq_buf: Vec::new(),
        }
    }

    /// Encode an RGBA pixel buffer into an iTerm2 inline image escape sequence.
    ///
    /// `width`/`height` are the actual image dimensions (PNG resolution).
    /// `disp_w`/`disp_h` are the display dimensions in the escape sequence
    /// (the terminal scales the image to these pixel dimensions).
    /// Returns the complete escape sequence bytes ready for stdout.
    pub fn encode(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
        disp_w: usize,
        disp_h: usize,
    ) -> &[u8] {
        use base64::Engine;
        use std::io::Write;

        // 1. Encode to uncompressed PNG (stored deflate — no compression)
        encode_png_stored(rgba, width, height, &mut self.raw_buf, &mut self.png_buf);

        // 2. Base64 encode the PNG bytes
        self.b64_buf.clear();
        base64::engine::general_purpose::STANDARD.encode_string(&self.png_buf, &mut self.b64_buf);

        // 3. Build escape sequence (display dimensions for terminal scaling)
        self.seq_buf.clear();
        write!(
            self.seq_buf,
            "\x1b]1337;File=inline=1;size={};width={}px;height={}px;preserveAspectRatio=0:{}\x07",
            self.png_buf.len(),
            disp_w,
            disp_h,
            self.b64_buf,
        )
        .expect("escape sequence write");

        &self.seq_buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_rgba(width: usize, height: usize) -> Vec<u8> {
        let mut rgba = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 4;
                rgba[i] = (x * 255 / width.max(1)) as u8; // R
                rgba[i + 1] = (y * 255 / height.max(1)) as u8; // G
                rgba[i + 2] = 128; // B
                rgba[i + 3] = 255; // A
            }
        }
        rgba
    }

    #[test]
    fn test_encode_produces_valid_escape_sequence() {
        let mut encoder = Iterm2Encoder::new();
        let rgba = make_test_rgba(32, 16);
        let result = encoder.encode(&rgba, 32, 16, 32, 16);

        assert!(!result.is_empty(), "Encoded output should not be empty");

        // Must start with ESC ]1337;File=
        let prefix = b"\x1b]1337;File=";
        assert!(
            result.starts_with(prefix),
            "Should start with iTerm2 escape prefix, got {:?}",
            &result[..result.len().min(20)]
        );

        // Must end with BEL (\x07)
        assert_eq!(
            result.last(),
            Some(&0x07),
            "Should end with BEL character"
        );

        // Must contain inline=1
        let as_str = std::str::from_utf8(result).expect("Should be valid UTF-8");
        assert!(as_str.contains("inline=1"), "Should contain inline=1");
    }

    #[test]
    fn test_encode_contains_base64_png() {
        let mut encoder = Iterm2Encoder::new();
        let rgba = make_test_rgba(16, 8);
        let result = encoder.encode(&rgba, 16, 8, 16, 8);
        let as_str = std::str::from_utf8(result).expect("Should be valid UTF-8");

        // Extract base64 payload after the colon separator
        let colon_pos = as_str.rfind(':').expect("Should contain colon separator");
        let b64_data = &as_str[colon_pos + 1..as_str.len() - 1]; // strip trailing BEL

        // Decode base64
        use base64::Engine;
        let png_bytes = base64::engine::general_purpose::STANDARD
            .decode(b64_data)
            .expect("Base64 payload should decode");

        // Verify PNG header
        assert!(
            png_bytes.starts_with(&[0x89, b'P', b'N', b'G']),
            "Decoded payload should be a valid PNG (starts with PNG header)"
        );
    }

    #[test]
    fn test_encode_reuse_buffers() {
        let mut encoder = Iterm2Encoder::new();

        // First encode
        let rgba1 = make_test_rgba(8, 6);
        let result1 = encoder.encode(&rgba1, 8, 6, 8, 6);
        assert!(!result1.is_empty());
        let len1 = result1.len();

        // Second encode with different dimensions
        let rgba2 = make_test_rgba(16, 12);
        let result2 = encoder.encode(&rgba2, 16, 12, 16, 12);
        assert!(!result2.is_empty());

        // Second should be larger (more pixels → more data)
        assert!(
            result2.len() > len1,
            "Larger image should produce larger output"
        );

        // Both should be valid escape sequences
        assert!(result2.starts_with(b"\x1b]1337;File="));
        assert_eq!(result2.last(), Some(&0x07));
    }

    #[test]
    fn test_encode_small_image() {
        let mut encoder = Iterm2Encoder::new();
        let rgba = make_test_rgba(8, 6);
        let result = encoder.encode(&rgba, 8, 6, 8, 6);

        assert!(!result.is_empty(), "Should encode even small images");
        assert!(result.starts_with(b"\x1b]1337;File="));
        assert_eq!(result.last(), Some(&0x07));

        // Verify dimensions are in the header
        let as_str = std::str::from_utf8(result).expect("Valid UTF-8");
        assert!(as_str.contains("width=8px"), "Should specify width");
        assert!(as_str.contains("height=6px"), "Should specify height");
    }

    #[test]
    fn test_encode_display_dimensions_differ_from_image() {
        let mut encoder = Iterm2Encoder::new();
        let rgba = make_test_rgba(16, 8);
        // Image is 16×8 but display at 640×320
        let result = encoder.encode(&rgba, 16, 8, 640, 320);

        let as_str = std::str::from_utf8(result).expect("Valid UTF-8");
        // Escape sequence should use display dimensions, not image dimensions
        assert!(as_str.contains("width=640px"), "Should use display width");
        assert!(as_str.contains("height=320px"), "Should use display height");
        assert!(!as_str.contains("width=16px"), "Should NOT use image width");
    }

    #[test]
    fn test_png_stored_decodable() {
        // Verify our custom PNG encoder produces output the png crate can decode
        let mut encoder = Iterm2Encoder::new();
        let rgba = make_test_rgba(32, 16);
        let result = encoder.encode(&rgba, 32, 16, 32, 16);

        let as_str = std::str::from_utf8(result).unwrap();
        let colon_pos = as_str.rfind(':').unwrap();
        let b64_data = &as_str[colon_pos + 1..as_str.len() - 1];

        use base64::Engine;
        let png_bytes = base64::engine::general_purpose::STANDARD
            .decode(b64_data)
            .unwrap();

        let decoder = png::Decoder::new(std::io::Cursor::new(&png_bytes));
        let mut reader = decoder.read_info().expect("PNG should be decodable");
        let info = reader.info();
        assert_eq!(info.width, 32);
        assert_eq!(info.height, 16);
        assert_eq!(info.color_type, png::ColorType::Rgba);

        let mut buf = vec![0u8; reader.output_buffer_size()];
        reader
            .next_frame(&mut buf)
            .expect("PNG frame should be readable");

        // Verify pixel data matches (RGBA passthrough)
        for y in 0..16 {
            for x in 0..32 {
                let i = (y * 32 + x) * 4;
                assert_eq!(buf[i], rgba[i], "R mismatch at ({x},{y})");
                assert_eq!(buf[i + 1], rgba[i + 1], "G mismatch at ({x},{y})");
                assert_eq!(buf[i + 2], rgba[i + 2], "B mismatch at ({x},{y})");
                assert_eq!(buf[i + 3], rgba[i + 3], "A mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_crc32_known_values() {
        assert_eq!(png_crc32(&[]), 0x00000000);
        assert_eq!(png_crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_adler32_known_values() {
        assert_eq!(adler32(&[]), 1);
        assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
    }
}
