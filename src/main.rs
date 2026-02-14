mod iterm2;
mod overlay;
mod renderer;
mod sixel;
mod solver;
mod state;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use minifb::{Key, KeyRepeat, Window, WindowOptions};
use state::FrameSnapshot;

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

fn format_status(params: &solver::SolverParams, tiles: usize, num_particles: usize, panel_visible: bool, model: state::FluidModel, show_vorticity: bool) -> String {
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
                let viz = if show_vorticity { "vorticity" } else { "dye" };
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

fn main() {
    if is_headless() {
        run_headless();
    } else {
        run_gui();
    }
}

fn run_gui() {
    let mut model = state::FluidModel::KarmanVortex;
    let win_width = 1280;
    let win_height = 640;
    let target_fps = 60;
    let steps_per_frame = 1;
    let num_particles = 400;
    let rb_tiles = 3;

    let mut tiles = 1; // Karman uses tiles=1

    // Per-model parameter storage
    let mut rb_params = solver::SolverParams::default();
    let mut karman_params = solver::SolverParams::default_karman();
    let mut current_params = karman_params.clone();
    let mut show_vorticity = false;
    let mut status_text = format_status(&current_params, tiles, num_particles, false, model, show_vorticity);

    let sim_nx = compute_sim_nx(win_width, win_height, model);
    let mut render_cfg = renderer::RenderConfig::fit(win_width, win_height, tiles, sim_nx);
    let mut w = render_cfg.frame_width;
    let mut h = render_cfg.frame_height;

    let mut window = Window::new(
        "fluvarium",
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
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    // Parameter channel: main → physics thread
    let (param_tx, param_rx) = mpsc::channel::<solver::SolverParams>();

    // Reset channel: main → physics thread (model switch)
    let (reset_tx, reset_rx) = mpsc::channel::<(state::FluidModel, state::SimState)>();

    // Physics thread → FrameSnapshot
    let (snap_tx, snap_rx) = mpsc::sync_channel::<FrameSnapshot>(1);
    let physics_running = running.clone();
    let init_params = current_params.clone();
    let physics_thread = std::thread::spawn(move || {
        let mut cur_model = model;
        let mut sim = match cur_model {
            state::FluidModel::KarmanVortex => state::SimState::new_karman(
                num_particles,
                init_params.inflow_vel,
                init_params.cylinder_x,
                init_params.cylinder_y,
                init_params.cylinder_radius,
                sim_nx,
            ),
            _ => state::SimState::new(num_particles, init_params.bottom_base, sim_nx),
        };
        let mut params = init_params;

        while physics_running.load(Ordering::SeqCst) {
            // Check for model reset
            while let Ok((new_model, new_sim)) = reset_rx.try_recv() {
                cur_model = new_model;
                sim = new_sim;
            }

            // Drain parameter updates (take latest)
            while let Ok(new_params) = param_rx.try_recv() {
                params = new_params;
            }

            for _ in 0..steps_per_frame {
                match cur_model {
                    state::FluidModel::KarmanVortex => solver::fluid_step_karman(&mut sim, &params),
                    _ => solver::fluid_step(&mut sim, &params),
                }
            }
            if snap_tx.send(sim.snapshot()).is_err() {
                break;
            }
        }
    });

    // Overlay state
    let mut overlay_state = overlay::OverlayState::new();

    // Main thread: render + display
    let mut framebuf = vec![0u32; w * h];
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
                status_text = format_status(&current_params, tiles, num_particles, false, model, show_vorticity);
                needs_redraw = true;
            } else {
                break;
            }
        }

        // Space: toggle overlay
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            overlay_state.toggle();
            status_text = format_status(&current_params, tiles, num_particles, overlay_state.visible, model, show_vorticity);
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
                    status_text = format_status(&current_params, tiles, num_particles, true, model, show_vorticity);
                }
                needs_redraw = true;
            }
            if window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
                if overlay::adjust_param(&mut current_params, overlay_state.selected, 1, false, model) {
                    let _ = param_tx.send(current_params.clone());
                    status_text = format_status(&current_params, tiles, num_particles, true, model, show_vorticity);
                }
                needs_redraw = true;
            }

            // Comma/Period: fine adjust
            if window.is_key_pressed(Key::Comma, KeyRepeat::Yes) {
                if overlay::adjust_param(&mut current_params, overlay_state.selected, -1, true, model) {
                    let _ = param_tx.send(current_params.clone());
                    status_text = format_status(&current_params, tiles, num_particles, true, model, show_vorticity);
                }
                needs_redraw = true;
            }
            if window.is_key_pressed(Key::Period, KeyRepeat::Yes) {
                if overlay::adjust_param(&mut current_params, overlay_state.selected, 1, true, model) {
                    let _ = param_tx.send(current_params.clone());
                    status_text = format_status(&current_params, tiles, num_particles, true, model, show_vorticity);
                }
                needs_redraw = true;
            }

            // R: reset selected parameter to default
            if window.is_key_pressed(Key::R, KeyRepeat::No) {
                overlay::reset_param(&mut current_params, overlay_state.selected, model);
                let _ = param_tx.send(current_params.clone());
                status_text = format_status(&current_params, tiles, num_particles, true, model, show_vorticity);
                needs_redraw = true;
            }
        }

        // M: switch fluid model
        if window.is_key_pressed(Key::M, KeyRepeat::No) {
            // Save current params for old model
            match model {
                state::FluidModel::RayleighBenard => rb_params = current_params.clone(),
                state::FluidModel::KarmanVortex => karman_params = current_params.clone(),
            }
            model = match model {
                state::FluidModel::RayleighBenard => state::FluidModel::KarmanVortex,
                state::FluidModel::KarmanVortex => state::FluidModel::RayleighBenard,
            };
            // Restore saved params for new model
            current_params = match model {
                state::FluidModel::KarmanVortex => karman_params.clone(),
                _ => rb_params.clone(),
            };
            tiles = match model {
                state::FluidModel::KarmanVortex => 1,
                _ => rb_tiles,
            };
            // Compute NX from current window size
            let (cur_w, cur_h) = window.get_size();
            let new_nx = compute_sim_nx(cur_w, cur_h, model);
            let new_sim = match model {
                state::FluidModel::KarmanVortex => state::SimState::new_karman(
                    num_particles,
                    current_params.inflow_vel,
                    current_params.cylinder_x,
                    current_params.cylinder_y,
                    current_params.cylinder_radius,
                    new_nx,
                ),
                _ => state::SimState::new(num_particles, current_params.bottom_base, new_nx),
            };
            let _ = reset_tx.send((model, new_sim));
            let _ = param_tx.send(current_params.clone());
            overlay_state.selected = 0;
            // Recompute layout for new tile count and NX
            render_cfg = renderer::RenderConfig::fit(cur_w, cur_h, tiles, new_nx);
            w = render_cfg.frame_width;
            h = render_cfg.frame_height;
            framebuf = vec![0u32; w * h];
            status_text = format_status(&current_params, tiles, num_particles, overlay_state.visible, model, show_vorticity);
            last_snap = None;
            needs_redraw = true;
        }

        // V: toggle vorticity visualization
        if window.is_key_pressed(Key::V, KeyRepeat::No) {
            show_vorticity = !show_vorticity;
            status_text = format_status(&current_params, tiles, num_particles, overlay_state.visible, model, show_vorticity);
            needs_redraw = true;
        }

        // --- Check for window resize ---
        let (new_w, new_h) = window.get_size();
        if new_w != w || new_h != h {
            // Stretch absorbs resize; NX stays the same
            render_cfg = renderer::RenderConfig::fit(new_w, new_h, tiles, render_cfg.sim_nx);
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
            let mut rgba = renderer::render(&s, &render_cfg, show_vorticity);
            renderer::render_status(&mut rgba, &render_cfg, &status_text);
            overlay::render_overlay(
                &mut rgba,
                render_cfg.frame_width,
                render_cfg.display_width,
                render_cfg.display_height,
                &overlay_state,
                &current_params,
                model,
            );
            rgba_to_argb(&rgba, &mut framebuf);
            last_snap = Some(s);
            needs_redraw = false;
        } else if needs_redraw {
            if let Some(ref s) = last_snap {
                let mut rgba = renderer::render(s, &render_cfg, show_vorticity);
                renderer::render_status(&mut rgba, &render_cfg, &status_text);
                overlay::render_overlay(
                    &mut rgba,
                    render_cfg.frame_width,
                    render_cfg.display_width,
                    render_cfg.display_height,
                    &overlay_state,
                    &current_params,
                    model,
                );
                rgba_to_argb(&rgba, &mut framebuf);
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
            window.set_title(&format!("fluvarium — {display_fps} fps"));
        }
    }

    // Shutdown
    running.store(false, Ordering::SeqCst);
    drop(snap_rx);
    let _ = physics_thread.join();
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

fn run_headless() {
    use std::io::Write;

    let model = state::FluidModel::KarmanVortex;
    let (win_width, win_height) = query_terminal_pixel_size().unwrap_or((640, 320));
    let steps_per_frame = 1;
    let num_particles = 400;
    let tiles = match model {
        state::FluidModel::KarmanVortex => 1,
        _ => 3,
    };
    let frame_interval = Duration::from_millis(33); // ~30fps

    let current_params = solver::SolverParams::default_karman();
    let show_vorticity = false;
    let status_text = format_status(&current_params, tiles, num_particles, false, model, show_vorticity);

    let sim_nx = compute_sim_nx(win_width, win_height, model);
    let render_cfg = renderer::RenderConfig::fit(win_width, win_height, tiles, sim_nx);

    // Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    // Physics thread → FrameSnapshot
    let (snap_tx, snap_rx) = mpsc::sync_channel::<FrameSnapshot>(1);
    let physics_running = running.clone();
    let init_params = current_params.clone();
    let physics_thread = std::thread::spawn(move || {
        let mut sim = match model {
            state::FluidModel::KarmanVortex => state::SimState::new_karman(
                num_particles,
                init_params.inflow_vel,
                init_params.cylinder_x,
                init_params.cylinder_y,
                init_params.cylinder_radius,
                sim_nx,
            ),
            _ => state::SimState::new(num_particles, init_params.bottom_base, sim_nx),
        };
        let params = init_params;

        while physics_running.load(Ordering::SeqCst) {
            for _ in 0..steps_per_frame {
                match model {
                    state::FluidModel::KarmanVortex => solver::fluid_step_karman(&mut sim, &params),
                    _ => solver::fluid_step(&mut sim, &params),
                }
            }
            if snap_tx.send(sim.snapshot()).is_err() {
                break;
            }
        }
    });

    // Terminal setup
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::with_capacity(4 * 1024 * 1024, stdout.lock());
    let _ = write!(out, "\x1b[?1049h"); // alternate screen
    let _ = write!(out, "\x1b[?25l"); // hide cursor
    let _ = write!(out, "\x1b[2J"); // clear screen
    let _ = out.flush();

    let mut encoder = iterm2::Iterm2Encoder::new();

    while running.load(Ordering::SeqCst) {
        let frame_start = Instant::now();

        // Blocking receive — waits for next physics frame
        let snap = match snap_rx.recv() {
            Ok(s) => s,
            Err(_) => break,
        };

        let mut rgba = renderer::render(&snap, &render_cfg, show_vorticity);
        renderer::render_status(&mut rgba, &render_cfg, &status_text);

        let seq = encoder.encode(&rgba, render_cfg.frame_width, render_cfg.frame_height);
        let _ = write!(out, "\x1b[H"); // cursor home
        let _ = out.write_all(seq);
        let _ = out.flush();

        // Rate limit to ~30fps
        let elapsed = frame_start.elapsed();
        if elapsed < frame_interval {
            std::thread::sleep(frame_interval - elapsed);
        }
    }

    // Terminal restore
    let _ = write!(out, "\x1b[?25h"); // show cursor
    let _ = write!(out, "\x1b[?1049l"); // restore main screen
    let _ = out.flush();

    // Shutdown
    running.store(false, Ordering::SeqCst);
    drop(snap_rx);
    let _ = physics_thread.join();
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
            let rgba = renderer::render(&snap, &cfg, false);
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
            let rgba = renderer::render(&snap, &cfg, false);
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
        let rgba = renderer::render(&snap, &cfg, false);

        let mut encoder = iterm2::Iterm2Encoder::new();
        let seq = encoder.encode(&rgba, cfg.frame_width, cfg.frame_height);

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
}
