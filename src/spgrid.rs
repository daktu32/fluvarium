// .spg frame reader and manifest.json parser.
// Standalone copy of the read side from spmodel-rs io.rs.

use serde::Deserialize;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

const SPG_MAGIC: &[u8; 4] = b"SPG\0";
const SPG_VERSION: u32 = 1;
const FIELD_NAME_ENTRY_SIZE: usize = 16;

#[derive(Debug, Clone)]
pub struct SpgFrame {
    pub step: u64,
    pub time: f64,
    pub fields: Vec<(String, Vec<f64>)>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct GridInfo {
    pub im: usize,
    pub jm: usize,
    pub nm: usize,
    #[serde(rename = "type")]
    pub grid_type: String,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct TimeInfo {
    pub dt: f64,
    pub output_interval: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct FrameEntry {
    pub step: u64,
    pub time: f64,
    pub file: String,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Manifest {
    pub version: u32,
    pub model: String,
    pub grid: GridInfo,
    pub time: TimeInfo,
    pub fields: Vec<String>,
    #[serde(default)]
    pub params: serde_json::Value,
    pub frames: Vec<FrameEntry>,
}

pub struct SpgReader {
    dir: PathBuf,
    pub manifest: Manifest,
}

impl SpgReader {
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let json = fs::read_to_string(dir.join("manifest.json"))?;
        let manifest: Manifest = serde_json::from_str(&json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Self { dir, manifest })
    }

    pub fn frame_count(&self) -> usize {
        self.manifest.frames.len()
    }

    pub fn read_frame(&self, index: usize) -> io::Result<SpgFrame> {
        if index >= self.manifest.frames.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("frame index {} out of range", index),
            ));
        }
        let entry = &self.manifest.frames[index];
        let path = self.dir.join(&entry.file);
        let mut file = fs::File::open(&path)?;
        read_spg_frame(&mut file)
    }

    pub fn read_all_frames(&self) -> io::Result<Vec<SpgFrame>> {
        (0..self.frame_count())
            .map(|i| self.read_frame(i))
            .collect()
    }
}

fn read_spg_frame<R: Read>(r: &mut R) -> io::Result<SpgFrame> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != SPG_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid SPG magic: {:?}", magic),
        ));
    }

    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != SPG_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported SPG version: {}", version),
        ));
    }

    r.read_exact(&mut buf4)?;
    let im = u32::from_le_bytes(buf4) as usize;
    r.read_exact(&mut buf4)?;
    let jm = u32::from_le_bytes(buf4) as usize;
    let grid_size = im * jm;

    r.read_exact(&mut buf4)?;
    let field_count = u32::from_le_bytes(buf4) as usize;

    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8)?;
    let step = u64::from_le_bytes(buf8);
    r.read_exact(&mut buf8)?;
    let time = f64::from_le_bytes(buf8);

    let mut field_names = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        let mut entry = [0u8; FIELD_NAME_ENTRY_SIZE];
        r.read_exact(&mut entry)?;
        let len = entry[0] as usize;
        let name = std::str::from_utf8(&entry[1..1 + len])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
            .to_string();
        field_names.push(name);
    }

    let mut fields = Vec::with_capacity(field_count);
    for name in field_names {
        let mut data = vec![0.0f64; grid_size];
        for val in &mut data {
            r.read_exact(&mut buf8)?;
            *val = f64::from_le_bytes(buf8);
        }
        fields.push((name, data));
    }

    Ok(SpgFrame { step, time, fields })
}
