mod renderer;
mod sixel;
mod solver;
mod state;

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use state::FrameSnapshot;

/// Shared FPS counters for pipeline stages.
struct FpsCounters {
    physics: AtomicU32,
    physics_ms: AtomicU32,
    render: AtomicU32,
    render_ms: AtomicU32,
    encode_ms: AtomicU32,
}

/// Drain the channel and return the most recent item.
/// Blocks on the first recv, then drains any additional buffered items.
#[cfg(test)]
fn drain_latest<T>(rx: &mpsc::Receiver<T>) -> Option<T> {
    let mut latest = rx.recv().ok()?;
    while let Ok(newer) = rx.try_recv() {
        latest = newer;
    }
    Some(latest)
}

fn main() {
    // Determine render layout from terminal pixel size
    let (render_cfg, top_pad) = match renderer::terminal_pixel_size() {
        Some((w, h)) => {
            let cfg = renderer::RenderConfig::fit(w, 256);
            let pad = h.saturating_sub(cfg.frame_height) / 2;
            (cfg, pad)
        }
        None => (renderer::RenderConfig::fit(542, 256), 0),
    };

    // Enter alternate screen and hide cursor
    let stdout = io::stdout();
    {
        let mut handle = stdout.lock();
        let _ = handle.write_all(b"\x1b[?1049h"); // alternate screen
        let _ = handle.write_all(b"\x1b[?25l"); // hide cursor
        let _ = handle.write_all(b"\x1b[?80h"); // disable Sixel scrolling
        let _ = handle.write_all(b"\x1b[2J"); // clear screen
        let _ = handle.flush();
    }

    // Set up Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    let fps = Arc::new(FpsCounters {
        physics: AtomicU32::new(0),
        physics_ms: AtomicU32::new(0),
        render: AtomicU32::new(0),
        render_ms: AtomicU32::new(0),
        encode_ms: AtomicU32::new(0),
    });

    // Stage 1: Physics thread → FrameSnapshot
    // Buffer of 1: physics can run 1 step ahead, then backpressures to render pace
    let (snap_tx, snap_rx) = mpsc::sync_channel::<FrameSnapshot>(1);
    let physics_running = running.clone();
    let physics_fps = Arc::clone(&fps);
    let physics_thread = std::thread::spawn(move || {
        let mut sim = state::SimState::new();
        let params = solver::SolverParams::default();
        let steps_per_frame = 1;
        let mut count = 0u32;
        let mut last_report = Instant::now();

        while physics_running.load(Ordering::SeqCst) {
            let pt0 = Instant::now();
            for _ in 0..steps_per_frame {
                solver::fluid_step(&mut sim, &params);
            }
            physics_fps.physics_ms.store(pt0.elapsed().as_millis() as u32, Ordering::Relaxed);
            if snap_tx.send(sim.snapshot()).is_err() {
                break;
            }
            count += 1;
            let now = Instant::now();
            if now.duration_since(last_report) >= Duration::from_secs(1) {
                physics_fps.physics.store(count, Ordering::Relaxed);
                count = 0;
                last_report = now;
            }
        }
    });

    // Stage 2: Render thread — FrameSnapshot → encoded sixel bytes
    // Uses recv() (NOT drain_latest) to process every snapshot in order,
    // so particle positions advance smoothly without skipping frames.
    let (sixel_tx, sixel_rx) = mpsc::sync_channel::<Vec<u8>>(2);
    let render_running = running.clone();
    let render_fps = Arc::clone(&fps);
    let render_thread = std::thread::spawn(move || {
        let mut count = 0u32;
        let mut last_report = Instant::now();

        let encoder = sixel::SixelEncoder::new();

        while render_running.load(Ordering::SeqCst) {
            let Ok(snap) = snap_rx.recv() else {
                break;
            };
            let t0 = Instant::now();
            let rgba = renderer::render(&snap, &render_cfg);
            let t1 = Instant::now();
            let data = encoder.encode(&rgba, render_cfg.frame_width, render_cfg.frame_height, top_pad);
            let t2 = Instant::now();
            render_fps.render_ms.store(t1.duration_since(t0).as_millis() as u32, Ordering::Relaxed);
            render_fps.encode_ms.store(t2.duration_since(t1).as_millis() as u32, Ordering::Relaxed);
            if sixel_tx.send(data).is_err() {
                break;
            }
            count += 1;
            let now = Instant::now();
            if now.duration_since(last_report) >= Duration::from_secs(1) {
                render_fps.render.store(count, Ordering::Relaxed);
                count = 0;
                last_report = now;
            }
        }
    });

    // Stage 3: Main thread — output encoded frames at steady pace
    let frame_duration = Duration::from_millis(16);
    let mut out_count = 0u32;
    let mut out_last = Instant::now();
    let mut out_fps: u32;
    while running.load(Ordering::SeqCst) {
        let frame_start = Instant::now();

        let Ok(sixel_data) = sixel_rx.recv() else {
            break;
        };

        let sixel_kb = sixel_data.len() / 1024;
        let out_t0 = Instant::now();
        if sixel::output_frame(&sixel_data).is_err() {
            break;
        }
        let out_ms = out_t0.elapsed().as_millis() as u32;

        out_count += 1;
        let now = Instant::now();
        if now.duration_since(out_last) >= Duration::from_secs(1) {
            out_fps = out_count;
            out_count = 0;
            out_last = now;

            // Show FPS in terminal title
            let p = fps.physics.load(Ordering::Relaxed);
            let r = fps.render.load(Ordering::Relaxed);
            let rm = fps.render_ms.load(Ordering::Relaxed);
            let em = fps.encode_ms.load(Ordering::Relaxed);
            let pm = fps.physics_ms.load(Ordering::Relaxed);
            let title = format!(
                "\x1b]0;fluvarium  phys:{p} render:{r} out:{out_fps} | sim:{pm}ms draw:{rm}ms enc:{em}ms write:{out_ms}ms {sixel_kb}KB\x07"
            );
            let _ = io::stdout().write_all(title.as_bytes());
        }

        let elapsed = frame_start.elapsed();
        if elapsed < frame_duration {
            std::thread::sleep(frame_duration - elapsed);
        }
    }

    // Signal all threads to stop.
    // Drop receivers first to unblock senders (prevents deadlock where
    // render blocks on sixel_tx.send() and physics blocks on snap_tx.send()).
    running.store(false, Ordering::SeqCst);
    drop(sixel_rx);
    let _ = render_thread.join();
    let _ = physics_thread.join();

    // Restore terminal
    {
        let mut handle = stdout.lock();
        let _ = handle.write_all(b"\x1b[?80l"); // re-enable Sixel scrolling
        let _ = handle.write_all(b"\x1b[?25h"); // show cursor
        let _ = handle.write_all(b"\x1b[?1049l"); // exit alternate screen
        let _ = handle.flush();
    }
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
        let latest = drain_latest(&rx).unwrap();
        assert_eq!(latest, 2, "Should get the last item sent");
    }
}
