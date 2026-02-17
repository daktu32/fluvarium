/// Terminal key event parsed from raw stdin bytes.
#[derive(Debug, Clone, PartialEq)]
pub enum TermKey {
    Space,
    Escape,
    Up,
    Down,
    Left,
    Right,
    Comma,
    Period,
    Char(char),
}

/// Parse a terminal key from raw bytes.
/// Returns `(parsed_key, bytes_consumed)`. `consumed=0` means need more data.
pub fn parse_key(buf: &[u8]) -> (Option<TermKey>, usize) {
    if buf.is_empty() {
        return (None, 0);
    }
    match buf[0] {
        0x1b => {
            // ESC sequence
            if buf.len() < 2 {
                return (Some(TermKey::Escape), 1);
            }
            if buf[1] != b'[' {
                return (Some(TermKey::Escape), 1);
            }
            // CSI sequence: ESC [ <final>
            if buf.len() < 3 {
                return (None, 0); // need more data
            }
            let key = match buf[2] {
                b'A' => Some(TermKey::Up),
                b'B' => Some(TermKey::Down),
                b'C' => Some(TermKey::Right),
                b'D' => Some(TermKey::Left),
                _ => None,
            };
            (key, 3)
        }
        0x20 => (Some(TermKey::Space), 1),
        b',' => (Some(TermKey::Comma), 1),
        b'.' => (Some(TermKey::Period), 1),
        b'q' | b'r' | b'm' | b'v' | b'a' | b'f' | b'p' | b'c'
        | b'[' | b']' => (Some(TermKey::Char(buf[0] as char)), 1),
        _ => (None, 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_key_empty() {
        let (key, consumed) = parse_key(&[]);
        assert_eq!(key, None);
        assert_eq!(consumed, 0);
    }

    #[test]
    fn test_parse_key_space() {
        let (key, consumed) = parse_key(&[0x20]);
        assert_eq!(key, Some(TermKey::Space));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_parse_key_comma() {
        let (key, consumed) = parse_key(&[b',']);
        assert_eq!(key, Some(TermKey::Comma));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_parse_key_period() {
        let (key, consumed) = parse_key(&[b'.']);
        assert_eq!(key, Some(TermKey::Period));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_parse_key_char_q() {
        let (key, consumed) = parse_key(&[b'q']);
        assert_eq!(key, Some(TermKey::Char('q')));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_parse_key_char_r() {
        let (key, consumed) = parse_key(&[b'r']);
        assert_eq!(key, Some(TermKey::Char('r')));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_parse_key_char_m() {
        let (key, consumed) = parse_key(&[b'm']);
        assert_eq!(key, Some(TermKey::Char('m')));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_parse_key_char_v() {
        let (key, consumed) = parse_key(&[b'v']);
        assert_eq!(key, Some(TermKey::Char('v')));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_parse_key_arrow_up() {
        let (key, consumed) = parse_key(&[0x1b, b'[', b'A']);
        assert_eq!(key, Some(TermKey::Up));
        assert_eq!(consumed, 3);
    }

    #[test]
    fn test_parse_key_arrow_down() {
        let (key, consumed) = parse_key(&[0x1b, b'[', b'B']);
        assert_eq!(key, Some(TermKey::Down));
        assert_eq!(consumed, 3);
    }

    #[test]
    fn test_parse_key_arrow_right() {
        let (key, consumed) = parse_key(&[0x1b, b'[', b'C']);
        assert_eq!(key, Some(TermKey::Right));
        assert_eq!(consumed, 3);
    }

    #[test]
    fn test_parse_key_arrow_left() {
        let (key, consumed) = parse_key(&[0x1b, b'[', b'D']);
        assert_eq!(key, Some(TermKey::Left));
        assert_eq!(consumed, 3);
    }

    #[test]
    fn test_parse_key_escape_alone() {
        // ESC followed by non-bracket byte -> Escape key, consume 1
        let (key, consumed) = parse_key(&[0x1b, b'x']);
        assert_eq!(key, Some(TermKey::Escape));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_parse_key_escape_only_byte() {
        // Single ESC at end of buffer
        let (key, consumed) = parse_key(&[0x1b]);
        assert_eq!(key, Some(TermKey::Escape));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_parse_key_incomplete_csi() {
        // ESC [ but no final byte -> need more data
        let (key, consumed) = parse_key(&[0x1b, b'[']);
        assert_eq!(key, None);
        assert_eq!(consumed, 0);
    }

    #[test]
    fn test_parse_key_unknown_csi() {
        // ESC [ with unknown final byte -> skip 3 bytes
        let (key, consumed) = parse_key(&[0x1b, b'[', b'Z']);
        assert_eq!(key, None);
        assert_eq!(consumed, 3);
    }

    #[test]
    fn test_parse_key_unknown_byte() {
        let (key, consumed) = parse_key(&[0x01]);
        assert_eq!(key, None);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_parse_key_sequence_in_buffer() {
        // Arrow up followed by space -- parse_key should only consume the arrow
        let buf = [0x1b, b'[', b'A', 0x20];
        let (key, consumed) = parse_key(&buf);
        assert_eq!(key, Some(TermKey::Up));
        assert_eq!(consumed, 3);
        // Second parse should get space
        let (key2, consumed2) = parse_key(&buf[consumed..]);
        assert_eq!(key2, Some(TermKey::Space));
        assert_eq!(consumed2, 1);
    }
}
