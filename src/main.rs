mod config;
mod renderer;
mod sixel;
mod solver;
mod state;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use minifb::{Key, Window, WindowOptions};
use state::FrameSnapshot;

/// Convert RGBA &[u8] buffer to 0RGB &[u32] buffer for minifb.
fn rgba_to_argb(rgba: &[u8], out: &mut [u32]) {
    for (i, pixel) in rgba.chunks_exact(4).enumerate() {
        out[i] = (pixel[0] as u32) << 16 | (pixel[1] as u32) << 8 | pixel[2] as u32;
    }
}

fn main() {
    let cfg = config::load();
    let status_text = format!(
        "visc={:.3} diff={:.3} dt={:.3} buoy={:.1} src={:.1} cool={:.1} base={:.2} | tiles={} p={}",
        cfg.physics.visc, cfg.physics.diff, cfg.physics.dt,
        cfg.physics.heat_buoyancy, cfg.physics.source_strength, cfg.physics.cool_rate,
        cfg.physics.bottom_base, cfg.display.tiles, cfg.particles,
    );
    let win_width = cfg.display.width;
    let win_height = cfg.display.height;
    let tiles = cfg.display.tiles;
    let target_fps = cfg.display.target_fps;
    let steps_per_frame = cfg.display.steps_per_frame;
    let num_particles = cfg.particles;
    let bottom_base = cfg.physics.bottom_base;
    let params = solver::SolverParams::from(&cfg.physics);

    let mut render_cfg = renderer::RenderConfig::fit(win_width, win_height, tiles);
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

    // Physics thread → FrameSnapshot
    let (snap_tx, snap_rx) = mpsc::sync_channel::<FrameSnapshot>(1);
    let physics_running = running.clone();
    let physics_thread = std::thread::spawn(move || {
        let mut sim = state::SimState::new(num_particles, bottom_base);

        while physics_running.load(Ordering::SeqCst) {
            for _ in 0..steps_per_frame {
                solver::fluid_step(&mut sim, &params);
            }
            if snap_tx.send(sim.snapshot()).is_err() {
                break;
            }
        }
    });

    // Main thread: render + display
    let mut framebuf = vec![0u32; w * h];
    let mut frame_count = 0u32;
    let mut last_fps_time = Instant::now();
    let mut display_fps: u32;

    let mut last_snap: Option<FrameSnapshot> = None;

    while window.is_open() && !window.is_key_down(Key::Escape) && running.load(Ordering::SeqCst) {
        // Check for window resize
        let (new_w, new_h) = window.get_size();
        if new_w != w || new_h != h {
            render_cfg = renderer::RenderConfig::fit(new_w, new_h, tiles);
            w = render_cfg.frame_width;
            h = render_cfg.frame_height;
            framebuf = vec![0u32; w * h];
            // Re-render last snapshot at new size
            if let Some(ref s) = last_snap {
                let mut rgba = renderer::render(s, &render_cfg);
                renderer::render_status(&mut rgba, &render_cfg, &status_text);
                rgba_to_argb(&rgba, &mut framebuf);
            }
        }

        // Non-blocking: grab latest snapshot if available
        let mut snap = None;
        while let Ok(s) = snap_rx.try_recv() {
            snap = Some(s);
        }

        if let Some(s) = snap {
            let mut rgba = renderer::render(&s, &render_cfg);
            renderer::render_status(&mut rgba, &render_cfg, &status_text);
            rgba_to_argb(&rgba, &mut framebuf);
            last_snap = Some(s);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_no_panic() {
        let mut sim = state::SimState::new(400, 0.15);
        let params = solver::SolverParams::default();
        let cfg = renderer::RenderConfig::default_config();

        for _ in 0..3 {
            solver::fluid_step(&mut sim, &params);
            let snap = sim.snapshot();
            let rgba = renderer::render(&snap, &cfg);
            let result = sixel::encode_sixel(&rgba, cfg.frame_width, cfg.frame_height);
            assert!(result.is_ok(), "Sixel encoding should succeed");
            let data = result.unwrap();
            assert!(!data.is_empty(), "Sixel output should not be empty");
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
}
