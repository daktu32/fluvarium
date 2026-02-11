# fluvarium Progress

## Current Status: Performance Optimized — 34fps Smooth Animation

49 tests passing. 3-thread pipeline with custom Sixel encoder. N=128 grid at 34fps.

## Completed

### Sub-issue 1: Project init
- `cargo init` with `icy_sixel`, `image`, `ctrlc`, `libc` dependencies
- Module skeleton: `src/{solver,state,renderer,sixel}.rs`

### Sub-issue 2: SimState
- `SimState` struct with all grid fields + Xor128 PRNG
- `idx(x, y)` with mod wrap-around
- `FrameSnapshot` for decoupling simulation state from rendering
- Initial conditions: linear gradient + large-scale sinusoidal perturbation to seed convection
- 11 tests

### Sub-issue 3: CFD primitives
- `set_bnd`: field_type 0/1 (Neumann), 2 (no-penetration wall), 3 (temperature Dirichlet)
- `lin_solve`: Gauss-Seidel iteration
- `diffuse`: implicit diffusion

### Sub-issue 4: Advection & projection
- `advect`: Semi-Lagrangian reverse trace + bilinear interpolation
- `project`: divergence → pressure solve → gradient subtraction

### Sub-issue 5: fluid_step & buoyancy
- `SolverParams`: visc=0.0007, diff=0.0001, dt=0.002, buoyancy=14.0
- Buoyancy: `vy += dt * buoyancy * (T - T_ambient)` where T_ambient=0.5
- Large-scale thermal perturbation injection (modes 1-4) each step
- Particle advection with bilinear velocity interpolation, periodic X wrap, ping-pong Y reflection
- Step order: diffuse vel → project → advect vel → buoyancy → **project** → diffuse temp → advect temp → perturbation → advect particles

### Sub-issue 6: Renderer
- `temperature_to_rgba`: Tokyo Night palette (dark navy→blue→cyan→orange→soft pink)
- Color bar with tick marks on the right side
- `render`: accepts `&FrameSnapshot`, temperature field + particles + color bar → RGBA buffer with y-axis flip
- Adaptive contrast particles (bright on dark, dark on bright backgrounds)
- Dynamic `RenderConfig::fit()` for terminal pixel size detection (xterm CSI 14t + TIOCGWINSZ fallback)
- 10 tests

### Sub-issue 7: Sixel output
- `encode_sixel`: icy_sixel fallback (test-only)
- `SixelEncoder`: custom encoder with fixed 64-color palette + 32KB RGB→palette LUT for O(1) color mapping
- `output_frame`: synchronized output (DEC 2026 BSU/ESU) + single-buffer write
- 8 tests

### Sub-issue 8: Main loop — 3-thread pipeline
- **Physics thread**: `fluid_step` → `FrameSnapshot` via `sync_channel(1)`
- **Render thread**: `render()` → `SixelEncoder::encode()` → sixel bytes via `sync_channel(2)`
- **Main thread**: `output_frame()` → stdout, FPS/timing diagnostics in terminal title bar
- All channels use `recv()` (no frame skipping) for smooth particle trajectories
- Ctrl+C handler with deadlock-free shutdown (`drop(rx)` before `join()`)
- Alternate screen + cursor hide + Sixel scrolling disable
- Display: full terminal width × half height (widescreen layout, simulation tiled 2×)
- 2 tests

## Performance Evolution

| Milestone | fps | Bottleneck |
|-----------|-----|------------|
| MVP single-thread | ~7 | icy_sixel encoding (130ms) |
| Custom Sixel encoder | ~7 | Physics N=256 (148ms) |
| N=256, dt=0.002 | ~14 | Physics (62ms) |
| N=128, half-size | ~55 | None (all stages <16ms) |
| N=128, full-width × half-height | ~34 | write (23ms) |

## Key Debugging History

1. **Buoyancy sign error**: `vy -=` → `vy +=` (hot fluid was sinking)
2. **Flickering**: Added synchronized output (DEC 2026), single-buffer writes, Atkinson dithering
3. **Temperature going all blue**: set_bnd Neumann BCs inside Gauss-Seidel iterations were erasing temperature boundaries → added field_type 3 (Dirichlet)
4. **Sharp red/blue interface**: Temperature advected with non-divergence-free velocity → moved projection before temperature advection
5. **No convection cells**: Uniform-in-x buoyancy killed by projection → switched from velocity noise to large-scale temperature perturbation at convection wavelengths
6. **Particle warping**: `drain_latest()` skipped intermediate frames → switched to `recv()` everywhere
7. **Ctrl+C deadlock**: 3-thread pipeline deadlocked on shutdown (blocked senders) → `drop(sixel_rx)` before joining threads

## Test Summary
- **49 tests, all passing**
- `cargo test` and `cargo build --release` both succeed

## Next Steps
- Fine-tune convection dynamics (plume count, speed, visual appeal)
- Explore further optimization opportunities
