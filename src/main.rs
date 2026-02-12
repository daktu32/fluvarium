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

const WIN_WIDTH: usize = 640;
const WIN_HEIGHT: usize = 320;

/// Convert RGBA &[u8] buffer to 0RGB &[u32] buffer for minifb.
fn rgba_to_argb(rgba: &[u8], out: &mut [u32]) {
    for (i, pixel) in rgba.chunks_exact(4).enumerate() {
        out[i] = (pixel[0] as u32) << 16 | (pixel[1] as u32) << 8 | pixel[2] as u32;
    }
}

fn main() {
    let render_cfg = renderer::RenderConfig::fit(WIN_WIDTH, WIN_HEIGHT);
    let w = render_cfg.frame_width;
    let h = render_cfg.frame_height;

    let mut window = Window::new(
        "fluvarium",
        w,
        h,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");

    // Cap display at ~60fps; physics runs faster via steps_per_frame
    window.set_target_fps(60);

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
        let mut sim = state::SimState::new();
        let params = solver::SolverParams::default();
        let steps_per_frame = 1;

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

    while window.is_open() && !window.is_key_down(Key::Escape) && running.load(Ordering::SeqCst) {
        // Non-blocking: grab latest snapshot if available
        let mut snap = None;
        while let Ok(s) = snap_rx.try_recv() {
            snap = Some(s);
        }

        if let Some(s) = snap {
            let rgba = renderer::render(&s, &render_cfg);
            rgba_to_argb(&rgba, &mut framebuf);
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
        let mut sim = state::SimState::new();
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
