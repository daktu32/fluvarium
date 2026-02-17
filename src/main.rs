mod input;
mod iterm2;
mod overlay;
mod physics;
mod playback;
mod renderer;
mod solver;
mod spherical;
mod spgrid;
mod state;

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use input::{parse_key, TermKey};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use physics::{compute_sim_nx, create_sim_state, ModelParams, PhysicsChannels, spawn_physics_thread};
use renderer::ColorMap;
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
    const HEADLESS_FRAME_INTERVAL_MS: u64 = 16;
    /// Max pixel count for headless render resolution (~200K pixels).
    /// Terminal upscales via iTerm2's width/height parameters.
    const HEADLESS_MAX_RENDER_PIXELS: usize = 640 * 320;
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

/// Short display name for each model (used in window title).
fn model_label(model: state::FluidModel) -> &'static str {
    match model {
        state::FluidModel::RayleighBenard => "Rayleigh\u{2013}B\u{00e9}nard",
        state::FluidModel::KarmanVortex => "K\u{00e1}rm\u{00e1}n Vortex",
        state::FluidModel::KelvinHelmholtz => "Kelvin\u{2013}Helmholtz",
        state::FluidModel::LidDrivenCavity => "Lid-Driven Cavity",
    }
}

/// Format window title with model name and FPS.
fn format_title(model: state::FluidModel, fps: u32) -> String {
    format!("fludarium \u{2223} {} \u{00b7} {} fps", model_label(model), fps)
}

fn format_status(params: &solver::SolverParams, tiles: usize, num_particles: usize, panel_visible: bool, model: state::FluidModel, viz_mode: renderer::VizMode) -> String {
    if panel_visible {
        "space=close  ud=nav  lr=adj  ,.=fine  d=default".to_string()
    } else {
        match model {
            state::FluidModel::RayleighBenard => format!(
                "visc={:.3} diff={:.3} dt={:.3} buoy={:.1} src={:.1} cool={:.1} base={:.2} | tiles={} p={} | space=params a=arrows r=restart m=model",
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
                    "karman [{viz}] | visc={:.3} dt={:.3} u0={:.2} re={:.0} | p={} | space=params v=viz a=arrows m=model",
                    params.visc, params.dt, params.inflow_vel, re, num_particles,
                )
            }
            state::FluidModel::KelvinHelmholtz => format!(
                "kh | visc={:.4} dt={:.3} shear={:.3} conf={:.1} | p={} | space=params a=arrows r=restart m=model",
                params.visc, params.dt, params.shear_velocity, params.confinement, num_particles,
            ),
            state::FluidModel::LidDrivenCavity => {
                let re = params.lid_velocity * (state::N as f64) / params.visc;
                format!(
                    "cavity | visc={:.3} dt={:.3} lid={:.2} re={:.0} | p={} | space=params a=arrows r=restart m=model",
                    params.visc, params.dt, params.lid_velocity, re, num_particles,
                )
            }
        }
    }
}

/// Build the appropriate RenderConfig for the current model.
/// Cavity uses aspect-preserving fit_square; others use stretch-to-fit.
fn make_render_cfg(w: usize, h: usize, tiles: usize, sim_nx: usize, model: state::FluidModel) -> renderer::RenderConfig {
    match model {
        state::FluidModel::LidDrivenCavity => renderer::RenderConfig::fit_square(w, h, tiles, sim_nx),
        _ => renderer::RenderConfig::fit(w, h, tiles, sim_nx),
    }
}

fn is_headless() -> bool {
    std::env::args().any(|a| a == "--headless")
}

/// Parse `--playback <dir>` from CLI args.
fn parse_playback_dir() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find(|w| w[0] == "--playback")
        .map(|w| w[1].clone())
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

/// Deterministic socket path for mpv IPC (per-process to avoid collisions).
fn bgm_socket_path() -> String {
    format!("/tmp/fludarium-mpv-{}.sock", std::process::id())
}

/// RAII guard that kills the bgm process and cleans up the IPC socket on drop.
struct BgmGuard;

impl Drop for BgmGuard {
    fn drop(&mut self) {
        kill_bgm_process();
        let _ = std::fs::remove_file(bgm_socket_path());
    }
}

/// Spawn mpv paused for background music playback.
/// mpv loads and buffers while paused; call `unpause_bgm()` to start playback.
/// Kills are guaranteed by three layers:
/// 1. `atexit` — runs even on `exit()` / framework-driven termination (macOS Cmd+Q)
/// 2. `ctrlc` handler — runs on SIGINT/SIGTERM
/// 3. `BgmGuard::drop` — runs on normal scope exit or panic unwind
/// Send a JSON IPC command to mpv and return the response line.
fn mpv_ipc(sock: &str, cmd: &[u8]) -> Option<String> {
    use std::os::unix::net::UnixStream;
    use std::io::{BufRead, BufReader, Write};
    let mut stream = UnixStream::connect(sock).ok()?;
    stream.set_write_timeout(Some(Duration::from_secs(2))).ok();
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
    stream.write_all(cmd).ok()?;
    stream.flush().ok();
    let mut reader = BufReader::new(&stream);
    let mut resp = String::new();
    reader.read_line(&mut resp).ok();
    Some(resp)
}

fn spawn_bgm(url: &str) -> Option<BgmGuard> {
    let sock = bgm_socket_path();
    let _ = std::fs::remove_file(&sock); // remove stale socket
    let child = std::process::Command::new("mpv")
        .args(["--no-video", "--pause", "--volume=0",
               &format!("--input-ipc-server={}", sock), url])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    BGM_PID.store(child.id() as i32, Ordering::SeqCst);
    unsafe { atexit(kill_bgm_process); }

    // Wait for mpv IPC socket to appear.
    for _ in 0..60 {
        if std::path::Path::new(&sock).exists() {

            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    if !std::path::Path::new(&sock).exists() {

        return Some(BgmGuard);
    }

    // Pre-warm: unpause (muted) to trigger yt-dlp resolution + audio pipeline init.

    mpv_ipc(&sock, b"{\"command\":[\"set_property\",\"pause\",false]}\n");

    // Wait for playback position to become valid (= audio pipeline ready).
    let mut start_pos: f64 = 0.0;
    for _ in 0..100 {
        if let Some(resp) = mpv_ipc(&sock, b"{\"command\":[\"get_property\",\"time-pos\"]}\n") {
            // Parse {"data":6141.234,"request_id":0,"error":"success"}
            if let Some(data_start) = resp.find("\"data\":") {
                let after = &resp[data_start + 7..];
                if let Some(end) = after.find(|c: char| c == ',' || c == '}') {
                    if let Ok(pos) = after[..end].trim().parse::<f64>() {
                        start_pos = pos;

                        break;
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    // Re-pause, seek back to start position, restore volume.

    mpv_ipc(&sock, b"{\"command\":[\"set_property\",\"pause\",true]}\n");
    let seek_cmd = format!("{{\"command\":[\"seek\",{},\"absolute\"]}}\n", start_pos);
    mpv_ipc(&sock, seek_cmd.as_bytes());
    mpv_ipc(&sock, b"{\"command\":[\"set_property\",\"volume\",100]}\n");


    Some(BgmGuard)
}

/// Send unpause command to mpv via its IPC socket (synchronous, instant after pre-warm).
fn unpause_bgm() {
    let sock = bgm_socket_path();

    if let Some(_resp) = mpv_ipc(&sock, b"{\"command\":[\"set_property\",\"pause\",false]}\n") {

    }
}

fn main() {
    if let Some(dir) = parse_playback_dir() {
        run_gui_playback(&dir);
    } else if is_headless() {
        run_headless();
    } else {
        run_gui();
    }
}

fn run_gui() {
    let bgm_child = parse_bgm_url().and_then(|url| spawn_bgm(&url));
    let mut bgm_started = bgm_child.is_none(); // true if no BGM (nothing to unpause)

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
    let mut show_arrows = false;
    let mut colormap = match model {
        state::FluidModel::KelvinHelmholtz => ColorMap::OceanLava,
        state::FluidModel::KarmanVortex => ColorMap::SolarWind,
        state::FluidModel::LidDrivenCavity => ColorMap::ArcticIce,
        _ => ColorMap::TokyoNight,
    };
    let mut status_text = format_status(&current_params, tiles, num_particles, false, model, viz_mode);

    let sim_nx = compute_sim_nx(win_width, win_height, model);
    let mut render_cfg = make_render_cfg(win_width, win_height, tiles, sim_nx, model);
    let mut w = render_cfg.frame_width;
    let mut h = render_cfg.frame_height;

    let initial_title = format!("fludarium \u{2223} {}", model_label(model));
    let mut window = Window::new(
        &initial_title,
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

            // D: reset selected parameter to default
            if window.is_key_pressed(Key::D, KeyRepeat::No) {
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
            colormap = match model {
                state::FluidModel::KelvinHelmholtz => ColorMap::OceanLava,
                state::FluidModel::KarmanVortex => ColorMap::SolarWind,
                state::FluidModel::LidDrivenCavity => ColorMap::ArcticIce,
                _ => ColorMap::TokyoNight,
            };
            let (cur_w, cur_h) = window.get_size();
            let new_nx = compute_sim_nx(cur_w, cur_h, model);
            let new_sim = create_sim_state(model, &current_params, num_particles, new_nx);
            let _ = reset_tx.send((model, new_sim));
            let _ = param_tx.send(current_params.clone());
            overlay_state.selected = 0;
            // Recompute layout for new tile count and NX
            render_cfg = make_render_cfg(cur_w, cur_h, tiles, new_nx, model);
            w = render_cfg.frame_width;
            h = render_cfg.frame_height;
            framebuf = vec![0u32; w * h];
            status_text = format_status(&current_params, tiles, num_particles, overlay_state.visible, model, viz_mode);
            last_snap = None;
            needs_redraw = true;
        }

        // R: restart current model simulation (works with overlay open too)
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            let (cur_w, cur_h) = window.get_size();
            let new_nx = compute_sim_nx(cur_w, cur_h, model);
            let new_sim = create_sim_state(model, &current_params, num_particles, new_nx);
            let _ = reset_tx.send((model, new_sim));
            render_cfg = make_render_cfg(cur_w, cur_h, tiles, new_nx, model);
            w = render_cfg.frame_width;
            h = render_cfg.frame_height;
            framebuf = vec![0u32; w * h];
            last_snap = None;
            needs_redraw = true;
        }

        // V: cycle visualization mode
        if window.is_key_pressed(Key::V, KeyRepeat::No) {
            viz_mode = viz_mode.next();
            status_text = format_status(&current_params, tiles, num_particles, overlay_state.visible, model, viz_mode);
            needs_redraw = true;
        }

        // A: toggle arrow overlay
        if window.is_key_pressed(Key::A, KeyRepeat::No) {
            show_arrows = !show_arrows;
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
            render_cfg = make_render_cfg(new_w, new_h, tiles, new_nx, model);
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
            // Unpause BGM on first rendered frame for audio-visual sync
            if !bgm_started {
                bgm_started = true;
                unpause_bgm();
            }
            renderer::render_into(&mut rgba_buf, &s, &render_cfg, viz_mode, colormap, show_arrows);
            renderer::render_status(&mut rgba_buf, &render_cfg, &status_text);
            overlay::render_overlay(
                &mut rgba_buf,
                render_cfg.frame_width,
                render_cfg.display_width,
                render_cfg.display_height,
                render_cfg.display_x_offset,
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
                renderer::render_into(&mut rgba_buf, s, &render_cfg, viz_mode, colormap, show_arrows);
                renderer::render_status(&mut rgba_buf, &render_cfg, &status_text);
                overlay::render_overlay(
                    &mut rgba_buf,
                    render_cfg.frame_width,
                    render_cfg.display_width,
                    render_cfg.display_height,
                    render_cfg.display_x_offset,
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
            window.set_title(&format_title(model, display_fps));
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

/// Frame data for the headless background encoder thread.
struct HeadlessFrame {
    rgba: Vec<u8>,
    width: usize,
    height: usize,
    disp_w: usize,
    disp_h: usize,
}

fn run_headless() {
    use std::io::Write;

    let bgm_child = parse_bgm_url().and_then(|url| spawn_bgm(&url));
    let mut bgm_started = bgm_child.is_none();

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
    let mut show_arrows = false;
    let mut colormap = match model {
        state::FluidModel::KelvinHelmholtz => ColorMap::OceanLava,
        state::FluidModel::KarmanVortex => ColorMap::SolarWind,
        state::FluidModel::LidDrivenCavity => ColorMap::ArcticIce,
        _ => ColorMap::TokyoNight,
    };
    let mut status_text = format_status(&current_params, tiles, num_particles, false, model, viz_mode);

    // sim_nx based on terminal aspect ratio (not reduced render dims)
    let sim_nx = compute_sim_nx(term_width, term_height, model);
    let mut render_cfg = make_render_cfg(render_w, render_h, tiles, sim_nx, model);
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

    // Terminal setup (before encoder thread takes over stdout)
    {
        let mut out = std::io::stdout();
        let _ = write!(out, "\x1b[?1049h\x1b[?25l\x1b[2J");
        let _ = out.flush();
    }

    // Enter raw terminal mode (RAII — restored on drop)
    let _raw_guard = raw_term::RawTerminal::enter();

    // Background encoder thread: render frames are sent here for
    // PNG encode → base64 → iTerm2 escape → stdout, decoupling
    // I/O from the main loop for smooth frame pacing.
    let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel::<HeadlessFrame>(1);
    let (buf_return_tx, buf_return_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let encoder_thread = std::thread::spawn(move || {
        use std::io::Write;
        let mut out = std::io::BufWriter::with_capacity(2 * 1024 * 1024, std::io::stdout());
        let mut encoder = iterm2::Iterm2Encoder::new();
        while let Ok(frame) = frame_rx.recv() {
            let seq = encoder.encode(
                &frame.rgba,
                frame.width,
                frame.height,
                frame.disp_w,
                frame.disp_h,
            );
            let _ = write!(out, "\x1b[H");
            let _ = out.write_all(seq);
            let _ = out.flush();
            // Return RGBA buffer for reuse (avoids allocation per frame)
            let _ = buf_return_tx.send(frame.rgba);
        }
    });

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
                    TermKey::Char('d') if overlay_state.visible => {
                        overlay::reset_param(&mut current_params, overlay_state.selected, model);
                        let _ = param_tx.send(current_params.clone());
                        status_text = format_status(&current_params, tiles, num_particles, true, model, viz_mode);
                        needs_redraw = true;
                    }
                    TermKey::Char('r') => {
                        // R: restart current model simulation
                        let new_nx = compute_sim_nx(term_width, term_height, model);
                        let new_sim = create_sim_state(model, &current_params, num_particles, new_nx);
                        let _ = reset_tx.send((model, new_sim));
                        render_cfg = make_render_cfg(render_w, render_h, tiles, new_nx, model);
                        if render_scale < 1.0 {
                            render_cfg.particle_radius = 0;
                        }
                        last_snap = None;
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
                        colormap = match model {
                            state::FluidModel::KelvinHelmholtz => ColorMap::OceanLava,
                            state::FluidModel::KarmanVortex => ColorMap::SolarWind,
                            state::FluidModel::LidDrivenCavity => ColorMap::ArcticIce,
                            _ => ColorMap::TokyoNight,
                        };
                        let new_nx = compute_sim_nx(term_width, term_height, model);
                        let new_sim = create_sim_state(model, &current_params, num_particles, new_nx);
                        let _ = reset_tx.send((model, new_sim));
                        let _ = param_tx.send(current_params.clone());
                        overlay_state.selected = 0;
                        render_cfg = make_render_cfg(render_w, render_h, tiles, new_nx, model);
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
                    TermKey::Char('a') => {
                        show_arrows = !show_arrows;
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
                render_cfg = make_render_cfg(render_w, render_h, tiles, render_cfg.sim_nx, model);
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
            // Unpause BGM on first rendered frame for audio-visual sync
            if !bgm_started {
                bgm_started = true;
                unpause_bgm();
            }
            // Recover a recycled buffer from the encoder thread (avoids alloc)
            if rgba_buf.is_empty() {
                rgba_buf = buf_return_rx.try_recv().unwrap_or_default();
            }
            renderer::render_into(&mut rgba_buf, &s, &render_cfg, viz_mode, colormap, show_arrows);
            renderer::render_status(&mut rgba_buf, &render_cfg, &status_text);
            overlay::render_overlay(
                &mut rgba_buf,
                render_cfg.frame_width,
                render_cfg.display_width,
                render_cfg.display_height,
                render_cfg.display_x_offset,
                &overlay_state,
                &current_params,
                model,
            );

            let frame = HeadlessFrame {
                rgba: std::mem::take(&mut rgba_buf),
                width: render_cfg.frame_width,
                height: render_cfg.frame_height,
                disp_w: term_width,
                disp_h: term_height,
            };
            match frame_tx.try_send(frame) {
                Ok(()) => {} // sent; buffer returned via buf_return channel
                Err(std::sync::mpsc::TrySendError::Full(rejected)) => {
                    rgba_buf = rejected.rgba; // encoder busy, keep buffer
                }
                Err(_) => break, // encoder thread died
            }
            // Return old snapshot buffer to physics thread for reuse
            if let Some(old) = last_snap.take() {
                let _ = snap_return_tx.send(old);
            }
            last_snap = Some(s);
            needs_redraw = false;
        } else if needs_redraw {
            if let Some(ref s) = last_snap {
                if rgba_buf.is_empty() {
                    rgba_buf = buf_return_rx.try_recv().unwrap_or_default();
                }
                renderer::render_into(&mut rgba_buf, s, &render_cfg, viz_mode, colormap, show_arrows);
                renderer::render_status(&mut rgba_buf, &render_cfg, &status_text);
                overlay::render_overlay(
                    &mut rgba_buf,
                    render_cfg.frame_width,
                    render_cfg.display_width,
                    render_cfg.display_height,
                    render_cfg.display_x_offset,
                    &overlay_state,
                    &current_params,
                    model,
                );

                let frame = HeadlessFrame {
                    rgba: std::mem::take(&mut rgba_buf),
                    width: render_cfg.frame_width,
                    height: render_cfg.frame_height,
                    disp_w: term_width,
                    disp_h: term_height,
                };
                match frame_tx.try_send(frame) {
                    Ok(()) => {}
                    Err(std::sync::mpsc::TrySendError::Full(rejected)) => {
                        rgba_buf = rejected.rgba;
                    }
                    Err(_) => break,
                }
            }
            needs_redraw = false;
        }

        // Rate limit to ~30fps
        let elapsed = frame_start.elapsed();
        if elapsed < frame_interval {
            std::thread::sleep(frame_interval - elapsed);
        }
    }

    // Shutdown encoder thread (drop sender → recv returns Err → thread exits)
    drop(frame_tx);
    let _ = encoder_thread.join();

    // Terminal restore (raw mode restored by _raw_guard drop)
    {
        let mut out = std::io::stdout();
        let _ = write!(out, "\x1b[?25h\x1b[?1049l");
        let _ = out.flush();
    }

    // Shutdown
    running.store(false, Ordering::SeqCst);
    drop(snap_rx);
    let _ = physics_thread.join();
    drop(bgm_child); // BgmGuard::drop kills mpv
}

fn run_gui_playback(dir: &str) {
    use renderer::spherical::{
        Projection, SphericalRenderConfig, render_equirectangular, render_orthographic,
    };

    eprintln!("Loading {dir}...");
    let reader = spgrid::SpgReader::open(dir).unwrap_or_else(|e| {
        eprintln!("Error opening {dir}: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "  model={}, grid={}x{} T{}, {} frames, fields={:?}",
        reader.manifest.model,
        reader.manifest.grid.im,
        reader.manifest.grid.jm,
        reader.manifest.grid.nm,
        reader.manifest.frames.len(),
        reader.manifest.fields,
    );

    let mut pb = playback::PlaybackState::from_reader(&reader).unwrap_or_else(|e| {
        eprintln!("Error reading frames: {e}");
        std::process::exit(1);
    });
    eprintln!("  {} frames loaded.", pb.frame_count());

    let win_width = Defaults::WIN_WIDTH;
    let win_height = Defaults::WIN_HEIGHT;
    let target_fps = Defaults::TARGET_FPS;

    let mut projection = Projection::Equirectangular;
    let mut colormap = ColorMap::BlueWhiteRed;
    let mut cam_lat = 0.0_f64;
    let mut cam_lon = 0.0_f64;

    let title = format!(
        "fludarium \u{2223} {} \u{00b7} {}",
        pb.model_name,
        projection.label()
    );
    let mut window = Window::new(
        &title,
        win_width,
        win_height,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");

    window.set_target_fps(target_fps);

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    let colormaps = [
        ColorMap::BlueWhiteRed,
        ColorMap::OceanLava,
        ColorMap::TokyoNight,
        ColorMap::SolarWind,
        ColorMap::ArcticIce,
    ];
    let mut colormap_idx = 0;

    let mut framebuf = vec![0u32; win_width * win_height];
    let mut rgba_buf: Vec<u8> = Vec::new();
    let mut last_time = Instant::now();
    let mut frame_count = 0u32;
    let mut last_fps_time = Instant::now();
    #[allow(unused_assignments)]
    let mut display_fps: u32 = 0;
    let mut needs_redraw = true;

    while window.is_open() && running.load(Ordering::SeqCst) {
        let now = Instant::now();
        let dt = now.duration_since(last_time).as_secs_f64();
        last_time = now;

        // Keyboard
        if window.is_key_pressed(Key::Escape, KeyRepeat::No)
            || window.is_key_pressed(Key::Q, KeyRepeat::No)
        {
            break;
        }

        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            pb.toggle_play();
            needs_redraw = true;
        }

        if window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
            if projection == Projection::Orthographic {
                cam_lon += 10.0_f64.to_radians();
                needs_redraw = true;
            } else {
                pb.step_forward();
                needs_redraw = true;
            }
        }
        if window.is_key_pressed(Key::Left, KeyRepeat::Yes) {
            if projection == Projection::Orthographic {
                cam_lon -= 10.0_f64.to_radians();
                needs_redraw = true;
            } else {
                pb.step_backward();
                needs_redraw = true;
            }
        }
        if window.is_key_pressed(Key::Up, KeyRepeat::Yes) {
            if projection == Projection::Orthographic {
                cam_lat = (cam_lat + 5.0_f64.to_radians()).min(std::f64::consts::FRAC_PI_2);
                needs_redraw = true;
            }
        }
        if window.is_key_pressed(Key::Down, KeyRepeat::Yes) {
            if projection == Projection::Orthographic {
                cam_lat = (cam_lat - 5.0_f64.to_radians()).max(-std::f64::consts::FRAC_PI_2);
                needs_redraw = true;
            }
        }

        if window.is_key_pressed(Key::LeftBracket, KeyRepeat::No) {
            pb.speed_down();
            needs_redraw = true;
        }
        if window.is_key_pressed(Key::RightBracket, KeyRepeat::No) {
            pb.speed_up();
            needs_redraw = true;
        }

        if window.is_key_pressed(Key::F, KeyRepeat::No) {
            pb.next_field();
            needs_redraw = true;
        }

        if window.is_key_pressed(Key::P, KeyRepeat::No) {
            projection = projection.toggle();
            needs_redraw = true;
        }

        if window.is_key_pressed(Key::C, KeyRepeat::No) {
            colormap_idx = (colormap_idx + 1) % colormaps.len();
            colormap = colormaps[colormap_idx];
            needs_redraw = true;
        }

        // Advance playback
        let prev_frame = pb.current_frame;
        pb.tick(dt);
        if pb.current_frame != prev_frame {
            needs_redraw = true;
        }

        let (cur_w, cur_h) = window.get_size();

        if needs_redraw {
            let snap = pb.snapshot();
            let gauss = pb.gauss_nodes();

            match projection {
                Projection::Equirectangular => {
                    let cfg = SphericalRenderConfig::equirectangular(cur_w, cur_h);
                    render_equirectangular(&mut rgba_buf, &snap, gauss, &cfg, colormap);

                    // Status bar
                    let status = format!(
                        "frame {}/{} t={:.3} [{}] x{:.1} | {}",
                        pb.current_frame,
                        pb.frame_count(),
                        snap.time,
                        snap.field_name,
                        pb.speed,
                        if pb.playing { ">" } else { "||" },
                    );
                    renderer::render_status(&mut rgba_buf, &renderer::RenderConfig {
                        display_width: cfg.display_width,
                        display_height: cfg.display_height,
                        frame_width: cfg.frame_width,
                        frame_height: cfg.frame_height,
                        tiles: 1,
                        sim_nx: snap.im,
                        particle_radius: 0,
                        display_x_offset: 0,
                    }, &status);

                    let w = cfg.frame_width;
                    let h = cfg.frame_height;
                    framebuf.resize(w * h, 0);
                    rgba_to_argb(&rgba_buf, &mut framebuf);
                    window.update_with_buffer(&framebuf, w, h).unwrap();
                }
                Projection::Orthographic => {
                    let cfg = SphericalRenderConfig::orthographic(cur_w, cur_h);
                    render_orthographic(
                        &mut rgba_buf, &snap, gauss, &cfg, colormap, cam_lat, cam_lon,
                    );

                    let status = format!(
                        "frame {}/{} t={:.3} [{}] x{:.1} | {} | lat={:.0} lon={:.0}",
                        pb.current_frame,
                        pb.frame_count(),
                        snap.time,
                        snap.field_name,
                        pb.speed,
                        if pb.playing { ">" } else { "||" },
                        cam_lat.to_degrees(),
                        cam_lon.to_degrees(),
                    );
                    renderer::render_status(&mut rgba_buf, &renderer::RenderConfig {
                        display_width: cfg.display_width,
                        display_height: cfg.display_height,
                        frame_width: cfg.frame_width,
                        frame_height: cfg.frame_height,
                        tiles: 1,
                        sim_nx: snap.im,
                        particle_radius: 0,
                        display_x_offset: 0,
                    }, &status);

                    let w = cfg.frame_width;
                    let h = cfg.frame_height;
                    framebuf.resize(w * h, 0);
                    rgba_to_argb(&rgba_buf, &mut framebuf);
                    window.update_with_buffer(&framebuf, w, h).unwrap();
                }
            }

            needs_redraw = false;
        } else {
            // Still need to call update to process events
            let (_w, _h) = (framebuf.len().max(1), 1);
            // Use current framebuf size
            let total = cur_w * cur_h;
            if framebuf.len() == total {
                window.update_with_buffer(&framebuf, cur_w, cur_h).unwrap();
            } else {
                window.update();
            }
        }

        frame_count += 1;
        if now.duration_since(last_fps_time) >= Duration::from_secs(1) {
            display_fps = frame_count;
            frame_count = 0;
            last_fps_time = now;
            let proj_label = projection.label();
            window.set_title(&format!(
                "fludarium \u{2223} {} \u{00b7} {} \u{00b7} {} fps",
                pb.model_name, proj_label, display_fps
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_no_panic() {
        let mut sim = state::SimState::new(400, 0.15, state::N);
        let params = solver::SolverParams::default();
        let cfg = renderer::RenderConfig::default_config();

        for _ in 0..3 {
            solver::fluid_step(&mut sim, &params);
            let snap = sim.snapshot();
            let rgba = renderer::render(&snap, &cfg, renderer::VizMode::Field, renderer::ColorMap::TokyoNight);
            assert_eq!(rgba.len(), cfg.frame_width * cfg.frame_height * 4);
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
            let rgba = renderer::render(&snap, &cfg, renderer::VizMode::Field, renderer::ColorMap::TokyoNight);
            assert_eq!(rgba.len(), cfg.frame_width * cfg.frame_height * 4);
        }
    }

    #[test]
    fn test_drain_latest_gets_newest() {
        let (tx, rx) = std::sync::mpsc::sync_channel::<i32>(10);
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
        let rgba = renderer::render(&snap, &cfg, renderer::VizMode::Field, renderer::ColorMap::TokyoNight);

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
