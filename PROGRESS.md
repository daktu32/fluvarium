# fluvarium Progress

## Current Status: minifb Native Window — 60fps Smooth Animation

49 tests passing. 2-thread pipeline (physics + render/display). N=128 grid at 60fps with minifb.

## Completed

### Sub-issue 1: Project init
- `cargo init` with `icy_sixel`, `ctrlc`, `minifb` dependencies
- Module skeleton: `src/{solver,state,renderer,sixel}.rs`

### Sub-issue 2: SimState
- `SimState` struct with all grid fields + Xor128 PRNG
- `idx(x, y)` with mod wrap-around
- `FrameSnapshot` for decoupling simulation state from rendering
- Initial conditions: Gaussian hot spot at bottom center, cold top
- 11 tests

### Sub-issue 3: CFD primitives
- `set_bnd`: field_type 0/1 (Neumann), 2 (no-penetration wall), 3 (temperature Dirichlet with Gaussian profile)
- `lin_solve`: Gauss-Seidel iteration
- `diffuse`: implicit diffusion

### Sub-issue 4: Advection & projection
- `advect`: Semi-Lagrangian reverse trace + bilinear interpolation
- `project`: divergence → pressure solve → gradient subtraction

### Sub-issue 5: fluid_step & buoyancy
- `SolverParams`: visc=0.008, diff=0.002, dt=0.003, buoyancy=8.0
- Buoyancy: `vy += dt * buoyancy * (T - T_ambient)` where T_ambient=BOTTOM_BASE=0.15
- Localized Gaussian heat source (`inject_heat_source`) with volumetric heating + Newtonian cooling
- Particle advection with bilinear velocity interpolation, periodic X wrap, ping-pong Y reflection
- Temperature clamped to [0, 1]
- Step order: diffuse vel → project → advect vel → buoyancy → **project** → diffuse temp → advect temp → inject heat → advect particles

### Sub-issue 6: Renderer
- `temperature_to_rgba`: Blackbody palette (black→dark red→crimson→orange→amber)
- Color bar with tick marks on the right side
- `render`: accepts `&FrameSnapshot`, temperature field + particles + color bar → RGBA buffer with y-axis flip
- Adaptive contrast particles (bright on dark, dark on bright backgrounds)
- Dynamic `RenderConfig::fit()` for arbitrary pixel dimensions
- 10 tests

### Sub-issue 7: Sixel output (test-only)
- `encode_sixel`: icy_sixel (test-only)
- `SixelEncoder`: custom encoder with fixed 64-color palette + 32KB RGB→palette LUT (test-only)
- 8 tests

### Sub-issue 8: Main loop — minifb native window
- **Physics thread**: `fluid_step` (1 step/frame) → `FrameSnapshot` via `sync_channel(1)`
- **Main thread**: `render()` → `rgba_to_argb()` → `window.update_with_buffer()` at 60fps
- Ctrl+C handler with clean shutdown (`drop(rx)` before `join()`)
- 640×320 window, non-resizable
- FPS counter in title bar
- 2 tests

## Performance Evolution

| Milestone | fps | Bottleneck |
|-----------|-----|------------|
| MVP single-thread (Sixel) | ~7 | icy_sixel encoding (130ms) |
| Custom Sixel encoder | ~7 | Physics N=256 (148ms) |
| N=256, dt=0.002 | ~14 | Physics (62ms) |
| N=128, half-size (Sixel) | ~55 | None (all stages <16ms) |
| N=128, full-width × half-height (Sixel) | ~34 | write (23ms) |
| **minifb native window** | **~60** | **None (vsync-limited)** |

## Key Debugging History

1. **Buoyancy sign error**: `vy -=` → `vy +=` (hot fluid was sinking)
2. **Flickering**: Added synchronized output (DEC 2026), single-buffer writes
3. **Temperature going all blue**: set_bnd Neumann BCs inside Gauss-Seidel iterations were erasing temperature boundaries → added field_type 3 (Dirichlet)
4. **Sharp red/blue interface**: Temperature advected with non-divergence-free velocity → moved projection before temperature advection
5. **No convection cells**: Uniform-in-x buoyancy killed by projection → switched from velocity noise to large-scale temperature perturbation at convection wavelengths
6. **Particle warping**: `drain_latest()` skipped intermediate frames → switched to `recv()` everywhere
7. **Ctrl+C deadlock**: 3-thread pipeline deadlocked on shutdown (blocked senders) → `drop(sixel_rx)` before joining threads
8. **Turbulent plume**: High buoyancy (60-200) caused Ra≈60,000+ with dominant numerical viscosity → physicist-tuned params (visc=0.008, diff=0.002, buoyancy=8.0) for clean convection
9. **Sixel bottleneck**: Sixel encode+write limited to ~34fps → switched to minifb native window for 60fps

## Dependency Cleanup
- Removed `libc` (terminal ioctl no longer needed with minifb)
- Removed `image` (unused)
- Removed `terminal_pixel_size` / `query_pixel_size_xterm` from renderer.rs
- Sixel encoder gated behind `#[cfg(test)]`

## Test Summary
- **49 tests, all passing** (1 ignored: diagnostic)
- `cargo test` and `cargo build --release` both succeed with 0 warnings

## Next Steps
- Fine-tune convection dynamics (plume count, speed, visual appeal)
- Issue #9: Parameter display / TUI overlay
- Issue #10: Simulation presets
