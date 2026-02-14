mod iterm2;
mod overlay;
mod renderer;
mod sixel;
mod solver;
mod state;

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use minifb::{Key, KeyRepeat, Window, WindowOptions};
use state::FrameSnapshot;

struct Defaults;

impl Defaults {
    const MODEL: state::FluidModel = state::FluidModel::KarmanVortex;
    const WIN_WIDTH: usize = 1280;
    const WIN_HEIGHT: usize = 640;
    const TARGET_FPS: usize = 60;
    const STEPS_PER_FRAME: usize = 1;
    const NUM_PARTICLES: usize = 400;
    const RB_TILES: usize = 1;
    const HEADLESS_WIDTH: usize = 1280;
    const HEADLESS_HEIGHT: usize = 640;
    const HEADLESS_FRAME_INTERVAL_MS: u64 = 33;
    /// Max pixel count for headless render resolution (~200K pixels).
    /// Terminal upscales via iTerm2's width/height parameters.
    const HEADLESS_MAX_RENDER_PIXELS: usize = 640 * 320;
}

/// Terminal key event parsed from raw stdin bytes.
#[derive(Debug, Clone, PartialEq)]
enum TermKey {
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
fn parse_key(buf: &[u8]) -> (Option<TermKey>, usize) {
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
        b'q' | b'r' | b'm' | b'v' => (Some(TermKey::Char(buf[0] as char)), 1),
        _ => (None, 1),
    }
}

/// Terminal raw mode via termios FFI (macOS only).
#[cfg(target_os = "macos")]
mod raw_term {
    #[repr(C)]
    struct Termios {
        c_iflag: u64,
        c_oflag: u64,
        c_cflag: u64,
        c_lflag: u64,
        c_cc: [u8; 20],
        c_ispeed: u64,
        c_ospeed: u64,
    }

    const ECHO: u64 = 0x08;
    const ICANON: u64 = 0x100;
    const VMIN: usize = 16;
    const VTIME: usize = 17;
    const TCSANOW: i32 = 0;
    const STDIN_FD: i32 = 0;

    unsafe extern "C" {
        fn tcgetattr(fd: i32, termios: *mut Termios) -> i32;
        fn tcsetattr(fd: i32, action: i32, termios: *const Termios) -> i32;
        fn read(fd: i32, buf: *mut u8, count: usize) -> isize;
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }

    /// RAII guard that restores original terminal settings on drop.
    pub struct RawTerminal {
        original: Termios,
    }

    impl RawTerminal {
        /// Enter raw mode: disable ICANON + ECHO, keep ISIG, set VMIN=0 VTIME=0.
        /// Returns `None` if tcgetattr fails (e.g. no real terminal in tests).
        pub fn enter() -> Option<Self> {
            unsafe {
                let mut original = std::mem::zeroed::<Termios>();
                if tcgetattr(STDIN_FD, &mut original) != 0 {
                    return None;
                }
                let mut raw = std::ptr::read(&original);
                raw.c_lflag &= !(ICANON | ECHO);
                raw.c_cc[VMIN] = 0;
                raw.c_cc[VTIME] = 0;
                if tcsetattr(STDIN_FD, TCSANOW, &raw) != 0 {
                    return None;
                }
                Some(Self { original })
            }
        }
    }

    impl Drop for RawTerminal {
        fn drop(&mut self) {
            unsafe {
                let _ = tcsetattr(STDIN_FD, TCSANOW, &self.original);
            }
        }
    }

    /// Non-blocking read from stdin. Returns number of bytes read (0 if nothing available).
    pub fn read_stdin(buf: &mut [u8]) -> usize {
        unsafe {
            // Set O_NONBLOCK via fcntl
            const F_GETFL: i32 = 3;
            const F_SETFL: i32 = 4;
            const O_NONBLOCK: i32 = 0x0004;

            let flags = fcntl(STDIN_FD, F_GETFL);
            let _ = fcntl(STDIN_FD, F_SETFL, flags | O_NONBLOCK);
            let n = read(STDIN_FD, buf.as_mut_ptr(), buf.len());
            let _ = fcntl(STDIN_FD, F_SETFL, flags); // restore
            if n > 0 { n as usize } else { 0 }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod raw_term {
    pub struct RawTerminal;

    impl RawTerminal {
        pub fn enter() -> Option<Self> {
            None
        }
    }

    pub fn read_stdin(_buf: &mut [u8]) -> usize {
        0
    }
}

/// Convert RGBA &[u8] buffer to 0RGB &[u32] buffer for minifb.
fn rgba_to_argb(rgba: &[u8], out: &mut [u32]) {
    for (i, pixel) in rgba.chunks_exact(4).enumerate() {
        out[i] = (pixel[0] as u32) << 16 | (pixel[1] as u32) << 8 | pixel[2] as u32;
    }
}

/// Compute simulation grid width from window dimensions and model.
/// Karman: NX = N × (win_w / win_h), clamped to at least N.
/// RB: always N (tiles handle horizontal extent).
fn compute_sim_nx(win_w: usize, win_h: usize, model: state::FluidModel) -> usize {
    match model {
        state::FluidModel::KarmanVortex => {
            let aspect = win_w as f64 / win_h as f64;
            let nx = (state::N as f64 * aspect).round() as usize;
            nx.max(state::N)
        }
        _ => state::N,
    }
}

/// Create a SimState for the given model, dispatching to the appropriate constructor.
fn create_sim_state(
    model: state::FluidModel,
    params: &solver::SolverParams,
    num_particles: usize,
    nx: usize,
) -> state::SimState {
    match model {
        state::FluidModel::KarmanVortex => state::SimState::new_karman(
            num_particles,
            params.inflow_vel,
            params.cylinder_x,
            params.cylinder_y,
            params.cylinder_radius,
            nx,
        ),
        _ => state::SimState::new(num_particles, params.bottom_base, nx),
    }
}

/// Per-model parameter storage with save/restore on model switch.
struct ModelParams {
    rb: solver::SolverParams,
    karman: solver::SolverParams,
}

impl ModelParams {
    fn new() -> Self {
        Self {
            rb: solver::SolverParams::default(),
            karman: solver::SolverParams::default_karman(),
        }
    }

    fn get(&self, model: state::FluidModel) -> &solver::SolverParams {
        match model {
            state::FluidModel::KarmanVortex => &self.karman,
            _ => &self.rb,
        }
    }

    /// Save current params for old_model, toggle to new model, return (new_model, restored_params).
    fn save_and_switch(
        &mut self,
        current: &solver::SolverParams,
        old_model: state::FluidModel,
    ) -> (state::FluidModel, solver::SolverParams) {
        match old_model {
            state::FluidModel::RayleighBenard => self.rb = current.clone(),
            state::FluidModel::KarmanVortex => self.karman = current.clone(),
        }
        let new_model = match old_model {
            state::FluidModel::RayleighBenard => state::FluidModel::KarmanVortex,
            state::FluidModel::KarmanVortex => state::FluidModel::RayleighBenard,
        };
        let restored = self.get(new_model).clone();
        (new_model, restored)
    }
}

/// Channels connecting the main (render) thread to the physics thread.
struct PhysicsChannels {
    param_tx: mpsc::Sender<solver::SolverParams>,
    reset_tx: mpsc::Sender<(state::FluidModel, state::SimState)>,
    snap_rx: mpsc::Receiver<FrameSnapshot>,
    snap_return_tx: mpsc::Sender<FrameSnapshot>,
}

/// Spawn the physics simulation thread and return its channels + join handle.
fn spawn_physics_thread(
    model: state::FluidModel,
    params: solver::SolverParams,
    num_particles: usize,
    sim_nx: usize,
    steps_per_frame: usize,
    running: Arc<AtomicBool>,
) -> (PhysicsChannels, std::thread::JoinHandle<()>) {
    let (param_tx, param_rx) = mpsc::channel::<solver::SolverParams>();
    let (reset_tx, reset_rx) = mpsc::channel::<(state::FluidModel, state::SimState)>();
    let (snap_tx, snap_rx) = mpsc::sync_channel::<FrameSnapshot>(1);
    let (snap_return_tx, snap_return_rx) = mpsc::channel::<FrameSnapshot>();

    let handle = std::thread::spawn(move || {
        let mut cur_model = model;
        let mut sim = create_sim_state(cur_model, &params, num_particles, sim_nx);
        let mut params = params;
        let mut snap_buf = FrameSnapshot::new_empty(sim.nx, state::N, num_particles);

        while running.load(Ordering::SeqCst) {
            while let Ok((new_model, new_sim)) = reset_rx.try_recv() {
                cur_model = new_model;
                sim = new_sim;
                snap_buf = FrameSnapshot::new_empty(sim.nx, state::N, num_particles);
            }
            while let Ok(new_params) = param_rx.try_recv() {
                params = new_params;
            }
            for _ in 0..steps_per_frame {
                match cur_model {
                    state::FluidModel::KarmanVortex => solver::fluid_step_karman(&mut sim, &params),
                    _ => solver::fluid_step(&mut sim, &params),
                }
            }
            sim.snapshot_into(&mut snap_buf);
            if snap_tx.send(snap_buf).is_err() {
                break;
            }
            let expected_len = sim.nx * state::N;
            snap_buf = snap_return_rx.try_recv()
                .ok()
                .filter(|b| b.temperature.len() == expected_len)
                .unwrap_or_else(|| FrameSnapshot::new_empty(sim.nx, state::N, num_particles));
        }
    });

    let channels = PhysicsChannels { param_tx, reset_tx, snap_rx, snap_return_tx };
    (channels, handle)
}

fn format_status(params: &solver::SolverParams, tiles: usize, num_particles: usize, panel_visible: bool, model: state::FluidModel, viz_mode: renderer::VizMode) -> String {
    if panel_visible {
        "space=close  ud=nav  lr=adj  ,.=fine  r=reset".to_string()
    } else {
        match model {
            state::FluidModel::RayleighBenard => format!(
                "visc={:.3} diff={:.3} dt={:.3} buoy={:.1} src={:.1} cool={:.1} base={:.2} | tiles={} p={} | space=params m=model",
                params.visc, params.diff, params.dt,
                params.heat_buoyancy, params.source_strength, params.cool_rate,
                params.bottom_base, tiles, num_particles,
            ),
            state::FluidModel::KarmanVortex => {
                let re = params.inflow_vel * (params.cylinder_radius * 2.0) / params.visc;
                let viz = match viz_mode {
                    renderer::VizMode::Field => "dye",
                    renderer::VizMode::Vorticity => "vorticity",
                    renderer::VizMode::Streamline => "streamline",
                    renderer::VizMode::None => "none",
                };
                format!(
                    "karman [{viz}] | visc={:.3} dt={:.3} u0={:.2} re={:.0} | p={} | space=params v=viz m=model",
                    params.visc, params.dt, params.inflow_vel, re, num_particles,
                )
            }
        }
    }
}

fn is_headless() -> bool {
    std::env::args().any(|a| a == "--headless")
}

/// Parse `--bgm <URL>` from CLI args.
fn parse_bgm_url() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find(|w| w[0] == "--bgm")
        .map(|w| w[1].clone())
}

/// Global PID for the bgm child process, so atexit/signal handlers can kill it.
static BGM_PID: AtomicI32 = AtomicI32::new(0);

unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    fn atexit(func: extern "C" fn()) -> i32;
}

/// Called by atexit and signal handlers — kills bgm process by stored PID.
extern "C" fn kill_bgm_process() {
    let pid = BGM_PID.swap(0, Ordering::SeqCst);
    if pid > 0 {
        unsafe {
            kill(pid, 9); // SIGKILL
            waitpid(pid, std::ptr::null_mut(), 0);
        }
    }
}

/// RAII guard that kills the bgm process on drop (backup for atexit).
struct BgmGuard;

impl Drop for BgmGuard {
    fn drop(&mut self) {
        kill_bgm_process();
    }
}

/// Spawn mpv for background music playback. Kills are guaranteed by three layers:
/// 1. `atexit` — runs even on `exit()` / framework-driven termination (macOS Cmd+Q)
/// 2. `ctrlc` handler — runs on SIGINT/SIGTERM
/// 3. `BgmGuard::drop` — runs on normal scope exit or panic unwind
fn spawn_bgm(url: &str) -> Option<BgmGuard> {
    let child = std::process::Command::new("mpv")
        .args(["--no-video", "--really-quiet", url])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    BGM_PID.store(child.id() as i32, Ordering::SeqCst);
    unsafe { atexit(kill_bgm_process); }
    Some(BgmGuard)
}

fn main() {
    if is_headless() {
        run_headless();
    } else {
        run_gui();
    }
}

fn run_gui() {
    let bgm_child = parse_bgm_url().and_then(|url| spawn_bgm(&url));

    let mut model = Defaults::MODEL;
    let win_width = Defaults::WIN_WIDTH;
    let win_height = Defaults::WIN_HEIGHT;
    let target_fps = Defaults::TARGET_FPS;
    let steps_per_frame = Defaults::STEPS_PER_FRAME;
    let num_particles = Defaults::NUM_PARTICLES;
    let rb_tiles = Defaults::RB_TILES;

    let mut tiles = 1; // Karman uses tiles=1

    let mut model_params = ModelParams::new();
    let mut current_params = model_params.get(model).clone();
    let mut viz_mode = renderer::VizMode::Field;
    let mut status_text = format_status(&current_params, tiles, num_particles, false, model, viz_mode);

    let sim_nx = compute_sim_nx(win_width, win_height, model);
    let mut render_cfg = renderer::RenderConfig::fit(win_width, win_height, tiles, sim_nx);
    let mut w = render_cfg.frame_width;
    let mut h = render_cfg.frame_height;

    let mut window = Window::new(
        "fludarium",
        w,
        h,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");

    window.set_target_fps(target_fps);

    // Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        kill_bgm_process();
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    let (channels, physics_thread) = spawn_physics_thread(
        model, current_params.clone(), num_particles, sim_nx, steps_per_frame, running.clone(),
    );
    let PhysicsChannels { param_tx, reset_tx, snap_rx, snap_return_tx } = channels;

    // Overlay state
    let mut overlay_state = overlay::OverlayState::new();

    // Main thread: render + display
    let mut framebuf = vec![0u32; w * h];
    let mut rgba_buf: Vec<u8> = Vec::new();
    let mut frame_count = 0u32;
    let mut last_fps_time = Instant::now();
    let mut display_fps: u32;
    let mut last_snap: Option<FrameSnapshot> = None;
    let mut needs_redraw = false;

    while window.is_open() && running.load(Ordering::SeqCst) {
        // --- Keyboard handling ---

        // Escape: close panel first, then quit app
        if window.is_key_pressed(Key::Escape, KeyRepeat::No) {
            if overlay_state.visible {
                overlay_state.visible = false;
                status_text = format_status(&current_params, tiles, num_particles, false, model, viz_mode);
                needs_redraw = true;
            } else {
                break;
            }
        }

        // Space: toggle overlay
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            overlay_state.toggle();
            status_text = format_status(&current_params, tiles, num_particles, overlay_state.visible, model, viz_mode);
            needs_redraw = true;
        }

        if overlay_state.visible {
            // Up/Down: navigate parameters
            if window.is_key_pressed(Key::Up, KeyRepeat::Yes) {
                overlay_state.navigate(-1, model);
                needs_redraw = true;
            }
            if window.is_key_pressed(Key::Down, KeyRepeat::Yes) {
                overlay_state.navigate(1, model);
                needs_redraw = true;
            }

            // Left/Right: adjust parameter (normal step)
            if window.is_key_pressed(Key::Left, KeyRepeat::Yes) {
                if overlay::adjust_param(&mut current_params, overlay_state.selected, -1, false, model) {
                    let _ = param_tx.send(current_params.clone());
                    status_text = format_status(&current_params, tiles, num_particles, true, model, viz_mode);
                }
                needs_redraw = true;
            }
            if window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
                if overlay::adjust_param(&mut current_params, overlay_state.selected, 1, false, model) {
                    let _ = param_tx.send(current_params.clone());
                    status_text = format_status(&current_params, tiles, num_particles, true, model, viz_mode);
                }
                needs_redraw = true;
            }

            // Comma/Period: fine adjust
            if window.is_key_pressed(Key::Comma, KeyRepeat::Yes) {
                if overlay::adjust_param(&mut current_params, overlay_state.selected, -1, true, model) {
                    let _ = param_tx.send(current_params.clone());
                    status_text = format_status(&current_params, tiles, num_particles, true, model, viz_mode);
                }
                needs_redraw = true;
            }
            if window.is_key_pressed(Key::Period, KeyRepeat::Yes) {
                if overlay::adjust_param(&mut current_params, overlay_state.selected, 1, true, model) {
                    let _ = param_tx.send(current_params.clone());
                    status_text = format_status(&current_params, tiles, num_particles, true, model, viz_mode);
                }
                needs_redraw = true;
            }

            // R: reset selected parameter to default
            if window.is_key_pressed(Key::R, KeyRepeat::No) {
                overlay::reset_param(&mut current_params, overlay_state.selected, model);
                let _ = param_tx.send(current_params.clone());
                status_text = format_status(&current_params, tiles, num_particles, true, model, viz_mode);
                needs_redraw = true;
            }
        }

        // M: switch fluid model
        if window.is_key_pressed(Key::M, KeyRepeat::No) {
            let (new_model, restored) = model_params.save_and_switch(&current_params, model);
            model = new_model;
            current_params = restored;
            tiles = match model {
                state::FluidModel::KarmanVortex => 1,
                _ => rb_tiles,
            };
            let (cur_w, cur_h) = window.get_size();
            let new_nx = compute_sim_nx(cur_w, cur_h, model);
            let new_sim = create_sim_state(model, &current_params, num_particles, new_nx);
            let _ = reset_tx.send((model, new_sim));
            let _ = param_tx.send(current_params.clone());
            overlay_state.selected = 0;
            // Recompute layout for new tile count and NX
            render_cfg = renderer::RenderConfig::fit(cur_w, cur_h, tiles, new_nx);
            w = render_cfg.frame_width;
            h = render_cfg.frame_height;
            framebuf = vec![0u32; w * h];
            status_text = format_status(&current_params, tiles, num_particles, overlay_state.visible, model, viz_mode);
            last_snap = None;
            needs_redraw = true;
        }

        // V: cycle visualization mode
        if window.is_key_pressed(Key::V, KeyRepeat::No) {
            viz_mode = viz_mode.next();
            status_text = format_status(&current_params, tiles, num_particles, overlay_state.visible, model, viz_mode);
            needs_redraw = true;
        }

        // --- Check for window resize ---
        let (new_w, new_h) = window.get_size();
        if new_w != w || new_h != h {
            let new_nx = compute_sim_nx(new_w, new_h, model);
            if new_nx != render_cfg.sim_nx {
                // Grid width changed — reset simulation to match new aspect ratio
                let new_sim = create_sim_state(model, &current_params, num_particles, new_nx);
                let _ = reset_tx.send((model, new_sim));
                last_snap = None;
            }
            render_cfg = renderer::RenderConfig::fit(new_w, new_h, tiles, new_nx);
            w = render_cfg.frame_width;
            h = render_cfg.frame_height;
            framebuf = vec![0u32; w * h];
            needs_redraw = true;
        }

        // --- Non-blocking: grab latest snapshot if available ---
        let mut snap = None;
        while let Ok(s) = snap_rx.try_recv() {
            snap = Some(s);
        }

        if let Some(s) = snap {
            renderer::render_into(&mut rgba_buf, &s, &render_cfg, viz_mode);
            renderer::render_status(&mut rgba_buf, &render_cfg, &status_text);
            overlay::render_overlay(
                &mut rgba_buf,
                render_cfg.frame_width,
                render_cfg.display_width,
                render_cfg.display_height,
                &overlay_state,
                &current_params,
                model,
            );
            rgba_to_argb(&rgba_buf, &mut framebuf);
            // Return old snapshot buffer to physics thread for reuse
            if let Some(old) = last_snap.take() {
                let _ = snap_return_tx.send(old);
            }
            last_snap = Some(s);
            needs_redraw = false;
        } else if needs_redraw {
            if let Some(ref s) = last_snap {
                renderer::render_into(&mut rgba_buf, s, &render_cfg, viz_mode);
                renderer::render_status(&mut rgba_buf, &render_cfg, &status_text);
                overlay::render_overlay(
                    &mut rgba_buf,
                    render_cfg.frame_width,
                    render_cfg.display_width,
                    render_cfg.display_height,
                    &overlay_state,
                    &current_params,
                    model,
                );
                rgba_to_argb(&rgba_buf, &mut framebuf);
            }
            needs_redraw = false;
        }

        window.update_with_buffer(&framebuf, w, h).unwrap();

        frame_count += 1;
        let now = Instant::now();
        if now.duration_since(last_fps_time) >= Duration::from_secs(1) {
            display_fps = frame_count;
            frame_count = 0;
            last_fps_time = now;
            window.set_title(&format!("fludarium — {display_fps} fps"));
        }
    }

    // Shutdown
    running.store(false, Ordering::SeqCst);
    drop(snap_rx);
    let _ = physics_thread.join();
    drop(bgm_child); // BgmGuard::drop kills mpv
}

/// Query terminal pixel dimensions via TIOCGWINSZ ioctl.
/// Returns (width_px, height_px) if the terminal reports pixel size.
#[cfg(target_os = "macos")]
fn query_terminal_pixel_size() -> Option<(usize, usize)> {
    #[repr(C)]
    struct Winsize {
        ws_row: u16,
        ws_col: u16,
        ws_xpixel: u16,
        ws_ypixel: u16,
    }

    unsafe extern "C" {
        fn ioctl(fd: i32, request: u64, ...) -> i32;
    }

    // TIOCGWINSZ on macOS = _IOR('t', 104, struct winsize) = 0x40087468
    const TIOCGWINSZ: u64 = 0x40087468;

    let mut ws = Winsize { ws_row: 0, ws_col: 0, ws_xpixel: 0, ws_ypixel: 0 };
    let ret = unsafe { ioctl(1, TIOCGWINSZ, &mut ws as *mut Winsize) };

    if ret == 0 && ws.ws_xpixel > 0 && ws.ws_ypixel > 0 {
        Some((ws.ws_xpixel as usize, ws.ws_ypixel as usize))
    } else {
        None
    }
}

#[cfg(not(target_os = "macos"))]
fn query_terminal_pixel_size() -> Option<(usize, usize)> {
    None
}

/// Compute capped render dimensions from terminal pixel size.
/// Returns (render_w, render_h, scale_factor).
fn headless_render_dims(term_w: usize, term_h: usize) -> (usize, usize, f64) {
    let actual = term_w * term_h;
    let max = Defaults::HEADLESS_MAX_RENDER_PIXELS;
    let scale = if actual > max {
        (max as f64 / actual as f64).sqrt()
    } else {
        1.0
    };
    let rw = ((term_w as f64 * scale) as usize).max(2);
    let rh = ((term_h as f64 * scale) as usize).max(2);
    (rw, rh, scale)
}

fn run_headless() {
    use std::io::Write;

    let bgm_child = parse_bgm_url().and_then(|url| spawn_bgm(&url));

    let mut model = Defaults::MODEL;
    let (mut term_width, mut term_height) = query_terminal_pixel_size()
        .unwrap_or((Defaults::HEADLESS_WIDTH, Defaults::HEADLESS_HEIGHT));
    let steps_per_frame = Defaults::STEPS_PER_FRAME;
    let num_particles = Defaults::NUM_PARTICLES;
    let rb_tiles = Defaults::RB_TILES;
    let mut tiles = match model {
        state::FluidModel::KarmanVortex => 1,
        _ => rb_tiles,
    };
    let frame_interval = Duration::from_millis(Defaults::HEADLESS_FRAME_INTERVAL_MS);

    // Cap render resolution for performance; terminal upscales via iTerm2 protocol
    let (mut render_w, mut render_h, mut render_scale) = headless_render_dims(term_width, term_height);

    let mut model_params = ModelParams::new();
    let mut current_params = model_params.get(model).clone();
    let mut viz_mode = renderer::VizMode::Field;
    let mut status_text = format_status(&current_params, tiles, num_particles, false, model, viz_mode);

    // sim_nx based on terminal aspect ratio (not reduced render dims)
    let sim_nx = compute_sim_nx(term_width, term_height, model);
    let mut render_cfg = renderer::RenderConfig::fit(render_w, render_h, tiles, sim_nx);
    if render_scale < 1.0 {
        render_cfg.particle_radius = 0; // single-pixel dots at reduced resolution
    }

    // Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        kill_bgm_process();
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    let (channels, physics_thread) = spawn_physics_thread(
        model, current_params.clone(), num_particles, sim_nx, steps_per_frame, running.clone(),
    );
    let PhysicsChannels { param_tx, reset_tx, snap_rx, snap_return_tx } = channels;

    // Overlay state
    let mut overlay_state = overlay::OverlayState::new();
    let mut last_snap: Option<FrameSnapshot> = None;
    let mut needs_redraw = false;

    // Terminal setup
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::with_capacity(4 * 1024 * 1024, stdout.lock());
    let _ = write!(out, "\x1b[?1049h"); // alternate screen
    let _ = write!(out, "\x1b[?25l"); // hide cursor
    let _ = write!(out, "\x1b[2J"); // clear screen
    let _ = out.flush();

    // Enter raw terminal mode (RAII — restored on drop)
    let _raw_guard = raw_term::RawTerminal::enter();

    let mut encoder = iterm2::Iterm2Encoder::new();
    let mut rgba_buf: Vec<u8> = Vec::new();
    let mut stdin_buf = [0u8; 64];
    let mut stdin_pending = 0usize; // bytes in stdin_buf not yet consumed

    while running.load(Ordering::SeqCst) {
        let frame_start = Instant::now();

        // --- Keyboard polling ---
        let n = raw_term::read_stdin(&mut stdin_buf[stdin_pending..]);
        stdin_pending += n;

        let mut cursor = 0;
        while cursor < stdin_pending {
            let (key, consumed) = parse_key(&stdin_buf[cursor..stdin_pending]);
            if consumed == 0 {
                break; // need more data
            }
            cursor += consumed;

            if let Some(k) = key {
                match k {
                    TermKey::Escape => {
                        if overlay_state.visible {
                            overlay_state.visible = false;
                            status_text = format_status(&current_params, tiles, num_particles, false, model, viz_mode);
                            needs_redraw = true;
                        } else {
                            running.store(false, Ordering::SeqCst);
                        }
                    }
                    TermKey::Char('q') => {
                        running.store(false, Ordering::SeqCst);
                    }
                    TermKey::Space => {
                        overlay_state.toggle();
                        status_text = format_status(&current_params, tiles, num_particles, overlay_state.visible, model, viz_mode);
                        needs_redraw = true;
                    }
                    TermKey::Up if overlay_state.visible => {
                        overlay_state.navigate(-1, model);
                        needs_redraw = true;
                    }
                    TermKey::Down if overlay_state.visible => {
                        overlay_state.navigate(1, model);
                        needs_redraw = true;
                    }
                    TermKey::Left if overlay_state.visible => {
                        if overlay::adjust_param(&mut current_params, overlay_state.selected, -1, false, model) {
                            let _ = param_tx.send(current_params.clone());
                            status_text = format_status(&current_params, tiles, num_particles, true, model, viz_mode);
                        }
                        needs_redraw = true;
                    }
                    TermKey::Right if overlay_state.visible => {
                        if overlay::adjust_param(&mut current_params, overlay_state.selected, 1, false, model) {
                            let _ = param_tx.send(current_params.clone());
                            status_text = format_status(&current_params, tiles, num_particles, true, model, viz_mode);
                        }
                        needs_redraw = true;
                    }
                    TermKey::Comma if overlay_state.visible => {
                        if overlay::adjust_param(&mut current_params, overlay_state.selected, -1, true, model) {
                            let _ = param_tx.send(current_params.clone());
                            status_text = format_status(&current_params, tiles, num_particles, true, model, viz_mode);
                        }
                        needs_redraw = true;
                    }
                    TermKey::Period if overlay_state.visible => {
                        if overlay::adjust_param(&mut current_params, overlay_state.selected, 1, true, model) {
                            let _ = param_tx.send(current_params.clone());
                            status_text = format_status(&current_params, tiles, num_particles, true, model, viz_mode);
                        }
                        needs_redraw = true;
                    }
                    TermKey::Char('r') if overlay_state.visible => {
                        overlay::reset_param(&mut current_params, overlay_state.selected, model);
                        let _ = param_tx.send(current_params.clone());
                        status_text = format_status(&current_params, tiles, num_particles, true, model, viz_mode);
                        needs_redraw = true;
                    }
                    TermKey::Char('m') => {
                        let (new_model, restored) = model_params.save_and_switch(&current_params, model);
                        model = new_model;
                        current_params = restored;
                        tiles = match model {
                            state::FluidModel::KarmanVortex => 1,
                            _ => rb_tiles,
                        };
                        let new_nx = compute_sim_nx(term_width, term_height, model);
                        let new_sim = create_sim_state(model, &current_params, num_particles, new_nx);
                        let _ = reset_tx.send((model, new_sim));
                        let _ = param_tx.send(current_params.clone());
                        overlay_state.selected = 0;
                        render_cfg = renderer::RenderConfig::fit(render_w, render_h, tiles, new_nx);
                        if render_scale < 1.0 {
                            render_cfg.particle_radius = 0;
                        }
                        status_text = format_status(&current_params, tiles, num_particles, overlay_state.visible, model, viz_mode);
                        last_snap = None;
                        needs_redraw = true;
                    }
                    TermKey::Char('v') => {
                        viz_mode = viz_mode.next();
                        status_text = format_status(&current_params, tiles, num_particles, overlay_state.visible, model, viz_mode);
                        needs_redraw = true;
                    }
                    _ => {}
                }
            }
        }
        // Shift unconsumed bytes to front
        if cursor > 0 && cursor < stdin_pending {
            stdin_buf.copy_within(cursor..stdin_pending, 0);
        }
        stdin_pending -= cursor;

        if !running.load(Ordering::SeqCst) {
            break;
        }

        // --- Check for terminal resize ---
        if let Some((new_tw, new_th)) = query_terminal_pixel_size() {
            if new_tw != term_width || new_th != term_height {
                term_width = new_tw;
                term_height = new_th;
                let (rw, rh, rs) = headless_render_dims(term_width, term_height);
                render_w = rw;
                render_h = rh;
                render_scale = rs;
                // Stretch to fit (keep sim_nx, like GUI mode)
                render_cfg = renderer::RenderConfig::fit(render_w, render_h, tiles, render_cfg.sim_nx);
                if render_scale < 1.0 {
                    render_cfg.particle_radius = 0;
                }
                needs_redraw = true;
            }
        }

        // --- Non-blocking: grab latest snapshot ---
        let mut snap = None;
        while let Ok(s) = snap_rx.try_recv() {
            snap = Some(s);
        }

        if let Some(s) = snap {
            renderer::render_into(&mut rgba_buf, &s, &render_cfg, viz_mode);
            renderer::render_status(&mut rgba_buf, &render_cfg, &status_text);
            overlay::render_overlay(
                &mut rgba_buf,
                render_cfg.frame_width,
                render_cfg.display_width,
                render_cfg.display_height,
                &overlay_state,
                &current_params,
                model,
            );

            let seq = encoder.encode(&rgba_buf, render_cfg.frame_width, render_cfg.frame_height, term_width, term_height);
            let _ = write!(out, "\x1b[H");
            let _ = out.write_all(seq);
            let _ = out.flush();
            // Return old snapshot buffer to physics thread for reuse
            if let Some(old) = last_snap.take() {
                let _ = snap_return_tx.send(old);
            }
            last_snap = Some(s);
            needs_redraw = false;
        } else if needs_redraw {
            if let Some(ref s) = last_snap {
                renderer::render_into(&mut rgba_buf, s, &render_cfg, viz_mode);
                renderer::render_status(&mut rgba_buf, &render_cfg, &status_text);
                overlay::render_overlay(
                    &mut rgba_buf,
                    render_cfg.frame_width,
                    render_cfg.display_width,
                    render_cfg.display_height,
                    &overlay_state,
                    &current_params,
                    model,
                );

                let seq = encoder.encode(&rgba_buf, render_cfg.frame_width, render_cfg.frame_height, term_width, term_height);
                let _ = write!(out, "\x1b[H");
                let _ = out.write_all(seq);
                let _ = out.flush();
            }
            needs_redraw = false;
        }

        // Rate limit to ~30fps
        let elapsed = frame_start.elapsed();
        if elapsed < frame_interval {
            std::thread::sleep(frame_interval - elapsed);
        }
    }

    // Terminal restore (raw mode restored by _raw_guard drop)
    let _ = write!(out, "\x1b[?25h"); // show cursor
    let _ = write!(out, "\x1b[?1049l"); // restore main screen
    let _ = out.flush();

    // Shutdown
    running.store(false, Ordering::SeqCst);
    drop(snap_rx);
    let _ = physics_thread.join();
    drop(bgm_child); // BgmGuard::drop kills mpv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_params_save_and_switch() {
        let mut mp = ModelParams::new();
        let mut current = mp.get(state::FluidModel::KarmanVortex).clone();
        // Modify a Karman param
        current.visc = 0.999;
        // Switch from Karman → RB
        let (new_model, restored) = mp.save_and_switch(&current, state::FluidModel::KarmanVortex);
        assert!(matches!(new_model, state::FluidModel::RayleighBenard));
        // Restored should be RB defaults
        assert!((restored.visc - solver::SolverParams::default().visc).abs() < 1e-10);
        // Saved Karman params should have our modification
        assert!((mp.karman.visc - 0.999).abs() < 1e-10);
        // Switch back from RB → Karman
        let (back_model, back_params) = mp.save_and_switch(&restored, new_model);
        assert!(matches!(back_model, state::FluidModel::KarmanVortex));
        // Should get our modified Karman visc back
        assert!((back_params.visc - 0.999).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_no_panic() {
        let mut sim = state::SimState::new(400, 0.15, state::N);
        let params = solver::SolverParams::default();
        let cfg = renderer::RenderConfig::default_config();

        for _ in 0..3 {
            solver::fluid_step(&mut sim, &params);
            let snap = sim.snapshot();
            let rgba = renderer::render(&snap, &cfg, renderer::VizMode::Field);
            let result = sixel::encode_sixel(&rgba, cfg.frame_width, cfg.frame_height);
            assert!(result.is_ok(), "Sixel encoding should succeed");
            let data = result.unwrap();
            assert!(!data.is_empty(), "Sixel output should not be empty");
        }
    }

    #[test]
    fn test_karman_pipeline_no_panic() {
        let params = solver::SolverParams::default_karman();
        let nx = 256; // wider than N for Karman
        let mut sim = state::SimState::new_karman(100, params.inflow_vel, params.cylinder_x, params.cylinder_y, params.cylinder_radius, nx);
        let cfg = renderer::RenderConfig::fit(542, 512, 1, nx);

        for _ in 0..3 {
            solver::fluid_step_karman(&mut sim, &params);
            let snap = sim.snapshot();
            let rgba = renderer::render(&snap, &cfg, renderer::VizMode::Field);
            assert_eq!(rgba.len(), cfg.frame_width * cfg.frame_height * 4);
        }
    }

    #[test]
    fn test_drain_latest_gets_newest() {
        let (tx, rx) = mpsc::sync_channel::<i32>(10);
        for i in 0..3 {
            tx.send(i).unwrap();
        }
        // drain
        let mut latest = rx.recv().unwrap();
        while let Ok(newer) = rx.try_recv() {
            latest = newer;
        }
        assert_eq!(latest, 2, "Should get the last item sent");
    }

    #[test]
    fn test_compute_sim_nx_rb() {
        let nx = compute_sim_nx(800, 600, state::FluidModel::RayleighBenard);
        assert_eq!(nx, state::N, "RB always uses N");
    }

    #[test]
    fn test_compute_sim_nx_karman() {
        let nx = compute_sim_nx(800, 400, state::FluidModel::KarmanVortex);
        // aspect = 800/400 = 2.0, so nx = N * 2
        assert_eq!(nx, state::N * 2);
    }

    #[test]
    fn test_compute_sim_nx_karman_min() {
        // Very tall window: aspect < 1
        let nx = compute_sim_nx(200, 800, state::FluidModel::KarmanVortex);
        assert_eq!(nx, state::N, "Karman NX should not go below N");
    }

    #[test]
    fn test_karman_resize_preserves_aspect_ratio() {
        // After resize, recomputing sim_nx should keep scale_x ≈ scale_y
        // Landscape/square windows where sim_nx >= N naturally holds
        let sizes: [(usize, usize); 4] = [
            (640, 320), (800, 400), (1024, 512), (800, 600),
        ];
        for (w, h) in sizes {
            let nx = compute_sim_nx(w, h, state::FluidModel::KarmanVortex);
            let cfg = renderer::RenderConfig::fit(w, h, 1, nx);
            let sx = cfg.scale_x();
            let sy = cfg.scale_y();
            let ratio = sx / sy;
            assert!((ratio - 1.0).abs() < 0.15,
                "scale_x/scale_y should be ~1.0 for {w}x{h}, got {ratio:.3} (sx={sx:.2}, sy={sy:.2}, nx={nx})");
        }
    }

    #[test]
    fn test_karman_stale_nx_causes_distortion() {
        // If sim_nx is NOT recalculated after resize, scales diverge
        let initial_nx = compute_sim_nx(640, 320, state::FluidModel::KarmanVortex);
        // Simulate resize to a very different aspect ratio without updating nx
        let cfg = renderer::RenderConfig::fit(1200, 400, 1, initial_nx);
        let ratio = cfg.scale_x() / cfg.scale_y();
        // With stale nx, ratio deviates significantly from 1.0
        assert!((ratio - 1.0).abs() > 0.3,
            "Stale sim_nx should cause distortion, got ratio={ratio:.3}");
    }

    #[test]
    fn test_headless_iterm2_pipeline() {
        // End-to-end: physics → render → iterm2 encode
        let params = solver::SolverParams::default_karman();
        let nx = compute_sim_nx(640, 320, state::FluidModel::KarmanVortex);
        let mut sim = state::SimState::new_karman(
            100, params.inflow_vel, params.cylinder_x,
            params.cylinder_y, params.cylinder_radius, nx,
        );
        let cfg = renderer::RenderConfig::fit(640, 320, 1, nx);

        solver::fluid_step_karman(&mut sim, &params);
        let snap = sim.snapshot();
        let rgba = renderer::render(&snap, &cfg, renderer::VizMode::Field);

        let mut encoder = iterm2::Iterm2Encoder::new();
        let seq = encoder.encode(&rgba, cfg.frame_width, cfg.frame_height, 640, 320);

        assert!(seq.starts_with(b"\x1b]1337;File="));
        assert_eq!(seq.last(), Some(&0x07));
    }

    #[test]
    fn test_query_terminal_pixel_size_no_panic() {
        // In test environments (no real terminal), should return None without panicking
        let result = query_terminal_pixel_size();
        // If it returns Some, dimensions should be positive
        if let Some((w, h)) = result {
            assert!(w > 0);
            assert!(h > 0);
        }
    }

    // --- parse_key tests ---

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
        // ESC followed by non-bracket byte → Escape key, consume 1
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
        // ESC [ but no final byte → need more data
        let (key, consumed) = parse_key(&[0x1b, b'[']);
        assert_eq!(key, None);
        assert_eq!(consumed, 0);
    }

    #[test]
    fn test_parse_key_unknown_csi() {
        // ESC [ with unknown final byte → skip 3 bytes
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
        // Arrow up followed by space — parse_key should only consume the arrow
        let buf = [0x1b, b'[', b'A', 0x20];
        let (key, consumed) = parse_key(&buf);
        assert_eq!(key, Some(TermKey::Up));
        assert_eq!(consumed, 3);
        // Second parse should get space
        let (key2, consumed2) = parse_key(&buf[consumed..]);
        assert_eq!(key2, Some(TermKey::Space));
        assert_eq!(consumed2, 1);
    }

    // --- raw_term tests ---

    #[test]
    fn test_raw_term_enter_in_test() {
        // In test environments (no real terminal), enter() should return None
        let guard = raw_term::RawTerminal::enter();
        // We don't assert None because CI might have a terminal,
        // but it should not panic either way.
        drop(guard);
    }

    #[test]
    fn test_raw_term_read_stdin_no_panic() {
        let mut buf = [0u8; 64];
        let n = raw_term::read_stdin(&mut buf);
        // In tests, stdin has no data, so n should be 0 (or small).
        assert!(n <= buf.len());
    }
}
