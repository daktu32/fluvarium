use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    pub physics: PhysicsConfig,
    pub display: DisplayConfig,
    pub particles: usize,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct PhysicsConfig {
    pub visc: f64,
    pub diff: f64,
    pub dt: f64,
    pub diffuse_iter: usize,
    pub project_iter: usize,
    pub heat_buoyancy: f64,
    pub noise_amp: f64,
    pub source_strength: f64,
    pub cool_rate: f64,
    pub bottom_base: f64,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct DisplayConfig {
    pub width: usize,
    pub height: usize,
    pub tiles: usize,
    pub target_fps: usize,
    pub steps_per_frame: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            physics: PhysicsConfig::default(),
            display: DisplayConfig::default(),
            particles: 400,
        }
    }
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            visc: 0.008,
            diff: 0.002,
            dt: 0.003,
            diffuse_iter: 20,
            project_iter: 30,
            heat_buoyancy: 8.0,
            noise_amp: 0.0,
            source_strength: 10.0,
            cool_rate: 8.0,
            bottom_base: 0.15,
        }
    }
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 320,
            tiles: 3,
            target_fps: 60,
            steps_per_frame: 1,
        }
    }
}

pub fn load() -> Config {
    let path = std::path::Path::new("fluvarium.yaml");
    if path.exists() {
        match std::fs::read_to_string(path) {
            Ok(contents) => match serde_yaml::from_str(&contents) {
                Ok(cfg) => cfg,
                Err(e) => {
                    eprintln!("Warning: failed to parse fluvarium.yaml: {e}; using defaults");
                    Config::default()
                }
            },
            Err(e) => {
                eprintln!("Warning: failed to read fluvarium.yaml: {e}; using defaults");
                Config::default()
            }
        }
    } else {
        Config::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let cfg = Config::default();
        assert_eq!(cfg.physics.visc, 0.008);
        assert_eq!(cfg.physics.diff, 0.002);
        assert_eq!(cfg.physics.dt, 0.003);
        assert_eq!(cfg.physics.diffuse_iter, 20);
        assert_eq!(cfg.physics.project_iter, 30);
        assert_eq!(cfg.physics.heat_buoyancy, 8.0);
        assert_eq!(cfg.physics.noise_amp, 0.0);
        assert_eq!(cfg.physics.source_strength, 10.0);
        assert_eq!(cfg.physics.cool_rate, 8.0);
        assert_eq!(cfg.physics.bottom_base, 0.15);
        assert_eq!(cfg.display.width, 640);
        assert_eq!(cfg.display.height, 320);
        assert_eq!(cfg.display.tiles, 3);
        assert_eq!(cfg.display.target_fps, 60);
        assert_eq!(cfg.display.steps_per_frame, 1);
        assert_eq!(cfg.particles, 400);
    }

    #[test]
    fn test_partial_yaml() {
        let yaml = "physics:\n  visc: 0.01\nparticles: 200\n";
        let cfg: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.physics.visc, 0.01);
        assert_eq!(cfg.physics.diff, 0.002); // default
        assert_eq!(cfg.particles, 200);
        assert_eq!(cfg.display.width, 640); // default
    }

    #[test]
    fn test_full_yaml() {
        let yaml = r#"
physics:
  visc: 0.01
  diff: 0.005
  dt: 0.005
  diffuse_iter: 10
  project_iter: 20
  heat_buoyancy: 5.0
  noise_amp: 0.1
  source_strength: 15.0
  cool_rate: 6.0
  bottom_base: 0.2
display:
  width: 800
  height: 400
  tiles: 2
  target_fps: 30
  steps_per_frame: 2
particles: 600
"#;
        let cfg: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.physics.visc, 0.01);
        assert_eq!(cfg.physics.diff, 0.005);
        assert_eq!(cfg.physics.dt, 0.005);
        assert_eq!(cfg.physics.diffuse_iter, 10);
        assert_eq!(cfg.physics.project_iter, 20);
        assert_eq!(cfg.physics.heat_buoyancy, 5.0);
        assert_eq!(cfg.physics.noise_amp, 0.1);
        assert_eq!(cfg.physics.source_strength, 15.0);
        assert_eq!(cfg.physics.cool_rate, 6.0);
        assert_eq!(cfg.physics.bottom_base, 0.2);
        assert_eq!(cfg.display.width, 800);
        assert_eq!(cfg.display.height, 400);
        assert_eq!(cfg.display.tiles, 2);
        assert_eq!(cfg.display.target_fps, 30);
        assert_eq!(cfg.display.steps_per_frame, 2);
        assert_eq!(cfg.particles, 600);
    }

    #[test]
    fn test_load_missing_file() {
        // When no fluvarium.yaml exists, load() should return defaults
        let cfg = load();
        assert_eq!(cfg.physics.visc, 0.008);
        assert_eq!(cfg.particles, 400);
    }
}
