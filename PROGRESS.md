# fludarium Progress

## Current Status: Modularized Dual-Model Fluid Simulator

140 tests passing. Codebase modularized into directory-based modules: `solver/` (7 files), `renderer/` (3 files), plus extracted `input.rs` and `physics.rs`. 2-thread pipeline (physics + render/display). Dual simulation models (Rayleigh-Benard convection + Kármán vortex street). N=80 grid with aspect-scaled NX for Kármán. Real-time parameter tuning via overlay panel in both GUI and headless modes. Headless terminal rendering via iTerm2 Graphics Protocol with full keyboard controls, adaptive render resolution, and dynamic terminal resize support. No external config files — all defaults in code.

## Completed

### Sub-issue 1: Project init
- `cargo init` with `icy_sixel`, `ctrlc`, `minifb` dependencies
- Module skeleton: `src/{solver,state,renderer,sixel}.rs`

### Sub-issue 2: SimState
- `SimState` struct with all grid fields + Xor128 PRNG
- `idx(x, y, nx)` with mod wrap-around, variable grid width
- `FrameSnapshot` for decoupling simulation state from rendering (includes velocity, trails, cylinder geometry)
- `FluidModel` enum: `RayleighBenard` / `KarmanVortex`
- Initial conditions: Gaussian hot spot at bottom center (RB), uniform inflow + cylinder obstacle (Kármán)
- Particle trail ring buffer (`TRAIL_LEN=8`) for trajectory visualization
- Fractional cylinder mask with smooth anti-aliased edges

### Sub-issue 3: CFD primitives
- `BoundaryConfig` enum dispatches per-model boundary conditions
- `set_bnd_rb`: periodic X, field_type 0/1 (Neumann), 2 (no-penetration wall), 3 (temperature Dirichlet with Gaussian profile)
- `set_bnd_karman`: Left Dirichlet inflow, right zero-gradient outflow, top/bottom no-slip walls
- `lin_solve`: Gauss-Seidel iteration
- `diffuse`: implicit diffusion

### Sub-issue 4: Advection & projection
- `advect`: Semi-Lagrangian reverse trace + bilinear interpolation, X clamping for Kármán
- `project`: divergence → pressure solve → gradient subtraction

### Sub-issue 5: fluid_step & buoyancy
- `SolverParams` with `default()` (RB) and `default_karman()` — all params in code
- Buoyancy: `vy += dt * buoyancy * (T - T_ambient)` where T_ambient=bottom_base
- Localized Gaussian heat source (`inject_heat_source`) with volumetric heating + Newtonian cooling
- Particle advection with bilinear velocity interpolation, periodic X wrap, ping-pong Y reflection
- `fluid_step_karman`: inflow injection, cylinder mask damping, dye tracer, vorticity confinement, wake perturbation
- Geometry-based particle respawn (distance check instead of mask threshold) to prevent surface sticking

### Sub-issue 6: Renderer
- Tokyo Night colormap (navy→blue→purple→pink→orange)
- Color bar with tick marks on the right side
- Adaptive contrast particles (bright on dark, dark on bright backgrounds)
- Dynamic `RenderConfig::fit()` for arbitrary pixel dimensions and configurable tile count
- Kármán: stretched to window, smooth anti-aliased cylinder rendering, optional vorticity visualization, particle trails
- **Status bar**: 5x7 bitmap font rendering current parameters
- **Font system**: nearest-neighbor resize for overlay text

### Sub-issue 7: Sixel output (test-only)
- `encode_sixel`: icy_sixel (test-only)
- `SixelEncoder`: custom encoder with fixed 64-color palette + 32KB RGB→palette LUT (test-only)

### Sub-issue 8: Main loop — minifb native window
- **Physics thread**: `fluid_step` / `fluid_step_karman` (configurable steps/frame) → `FrameSnapshot` via `sync_channel(1)`
- **Main thread**: `render()` → `rgba_to_argb()` → `window.update_with_buffer()` at 60fps target
- Ctrl+C handler with clean shutdown
- **Resizable window** with dynamic re-render on size change
- FPS counter in title bar
- **M key**: model switch with per-model parameter preservation (`rb_params` / `karman_params`)
- **V key**: toggle vorticity visualization (Kármán mode)

### Sub-issue 9: Interactive Overlay Parameter Panel
- **overlay.rs**: btop-style semi-transparent panel with darken background
  - `OverlayState` (visible, selected), `ParamDef` with get/set function pointers
  - RB: 7 params (visc, diff, dt, buoyancy, source_strength, cool_rate, bottom_base)
  - Kármán: 5 params (visc, diff, dt, inflow_vel, confinement)
- **Keyboard controls**: Space=toggle, Up/Down=navigate, Left/Right=adjust, Comma/Period=fine, R=reset, Escape=close/quit
- **Real-time tuning**: `mpsc::channel` sends updated `SolverParams` to physics thread

### Sub-issue 10: Headless Terminal Mode (iTerm2 Graphics Protocol)
- **iterm2.rs**: `Iterm2Encoder` struct with reusable buffers (PNG, base64, escape sequence)
  - RGBA → RGB strip → PNG encode (fast compression) → base64 → iTerm2 `\x1b]1337;File=...` escape sequence
  - Zero per-frame allocation via buffer reuse
- **`--headless` CLI flag**: renders to terminal instead of opening minifb window
  - 640×320 resolution at ~30fps
  - Alternate screen buffer + hidden cursor for clean terminal output
  - Ctrl+C restores terminal state (cursor visible, main screen)
  - Same physics thread and render pipeline as GUI mode
- **Dependencies**: `png = "0.17"`, `base64 = "0.22"`
- **main.rs refactor**: `run_gui()` (existing) + `run_headless()` (new) dispatched from `main()`

### Sub-issue 11: Headless Keyboard Controls
- **TermKey enum + parse_key()**: Terminal escape sequence parser for raw stdin bytes
  - ESC[A/B/C/D → arrow keys, bare ESC → Escape, Space/Comma/Period/q/r/m/v
  - Incremental buffer parsing with `(Option<TermKey>, bytes_consumed)` return
- **raw_term module**: macOS termios FFI for terminal raw mode
  - RAII `RawTerminal` guard (disable ICANON/ECHO, VMIN=0/VTIME=0, keep ISIG for Ctrl+C)
  - Non-blocking `read_stdin()` via fcntl O_NONBLOCK
  - Non-macOS stubs
- **run_headless() refactor**: Full keyboard support matching GUI mode
  - `param_tx`/`param_rx` + `reset_tx`/`reset_rx` channels to physics thread
  - Overlay panel: Space=toggle, Up/Down=nav, Left/Right=adjust, Comma/Period=fine, R=reset
  - Model switch (M) with per-model param storage (`rb_params`/`karman_params`)
  - Vorticity toggle (V), quit (q/Escape)
  - Non-blocking `try_recv()` + keyboard polling loop at ~30fps
- **20 new tests**: 18 parse_key + 2 raw_term smoke tests

### Headless Performance & Resize
- **Render resolution capping**: Headless render dimensions capped at ~200K pixels (`HEADLESS_MAX_RENDER_PIXELS = 640×320`), aspect-ratio preserving scale-down. Terminal upscales via iTerm2's `width/height` pixel parameters.
- **Separate display dimensions**: `Iterm2Encoder::encode()` accepts image dimensions (for PNG) and display dimensions (for escape sequence) independently, enabling low-res render + full-res display.
- **Adaptive particle size**: `RenderConfig::particle_radius` field (0=single pixel, 1=3×3 diamond). Headless at reduced resolution uses single-pixel dots so particles don't appear oversized after terminal upscaling.
- **Dynamic terminal resize**: Each frame polls `TIOCGWINSZ` ioctl to detect terminal size changes. On resize: recomputes render dimensions, regenerates `RenderConfig`, triggers immediate redraw. Simulation grid (sim_nx) preserved (stretch-to-fit, matching GUI behavior).
- **`headless_render_dims()` helper**: DRY extraction of render scale computation, shared between init and resize paths.

### Config simplification
- Removed YAML config system (`config.rs`, `serde`/`serde_yaml` dependencies)
- All defaults managed in code: `SolverParams::default()`, `default_karman()`, `PARAM_DEFS_*`
- Window size (1280×640), tiles, particles hardcoded in `main.rs`
- Startup model: Kármán vortex (visc=0.015, diff=0.003, dt=0.06, confinement=3.0, cylinder at vertical center)

## Performance Evolution

| Milestone | fps | Bottleneck |
|-----------|-----|------------|
| MVP single-thread (Sixel) | ~7 | icy_sixel encoding (130ms) |
| Custom Sixel encoder | ~7 | Physics N=256 (148ms) |
| N=256, dt=0.002 | ~14 | Physics (62ms) |
| N=128, half-size (Sixel) | ~55 | None (all stages <16ms) |
| N=128, full-width × half-height (Sixel) | ~34 | write (23ms) |
| **minifb native window** | **~60** | **None (vsync-limited)** |
| **Headless (pre-opt, 1280×640)** | ~20-25 | PNG encode + stdout I/O (>33ms/frame) |
| **Headless (post-opt, 640×320)** | **~30** | **None (within 33ms budget)** |

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
10. **Particle sticking on cylinder**: mask threshold (>0.5) missed smooth edge zone → switched to geometry-based distance check with 1-cell margin
11. **Config/default mismatch**: YAML config values diverged from `default_karman()` → removed config system, single source of truth in code

## Dependency Cleanup
- Removed `libc` (terminal ioctl no longer needed with minifb)
- Removed `image` (unused)
- Removed `serde` + `serde_yaml` (YAML config removed)
- Sixel encoder gated behind `#[cfg(test)]`

### Kent Beck-Style Refactoring (Phase 1 + 2)
- **Step 1.1**: Extracted `create_sim_state()` — eliminated 5 duplicated `match model { KarmanVortex => new_karman(...), _ => new(...) }` patterns in main.rs
- **Step 1.2**: Extracted `spawn_physics_thread()` + `PhysicsChannels` struct — eliminated identical 47-line physics thread closure duplicated between run_gui and run_headless (-29 lines)
- **Step 1.3**: Extracted `ModelParams` struct with `save_and_switch()` — eliminated duplicated save/toggle/restore model-switch logic between GUI (M key) and headless (m key), with dedicated test
- **Step 2.2**: Extracted `interpolate_velocity()` in solver.rs — shared bilinear velocity interpolation between `advect_particles` and `advect_particles_karman`, with 2 precision tests
- **Phase 3 (YAGNI)**: Deferred trait-based abstractions until 3rd model per Kent Beck's Rule of Three

### Solver/Physics Refactoring (Phase 2)
- **Phase 1**: Updated `set_bnd_rb` doc comment from numeric `field_type 0/1/2/3` to enum variant names (`FieldType::Scalar/Vx/Vy/Temperature`)
- **Phase 2**: Added `BoundaryConfig::periodic_x()` and `x_range(nx)` helper methods, eliminated 5 duplicated `matches!`/`match` patterns in `core.rs`
- **Phase 3**: Added `apply_mask_fields(vx, vy, mask)` to directly damp arbitrary velocity fields, replaced 4-line swap hack in `fluid_step_karman`
- **Phase 4**: Extracted `inject_wake_perturbation()` and `damp_dye_in_cylinder()` from inline code in `fluid_step_karman`
- **Phase 5**: Renamed scratch buffers for clarity: `work`→`scratch_a`, `work2`→`scratch_b`, `work3`→`vorticity`, `work4`→`vorticity_abs` with doc comments

### Legacy Cleanup
- Removed `src/sixel.rs` and `icy_sixel` dependency (test-only legacy, replaced by iTerm2 protocol)
- Removed `REVIEW-fluid-simulation.md` (outdated review document)

### Module Structure Refactoring
- **`solver.rs` (1621行) → `src/solver/` ディレクトリモジュール (7ファイル)**:
  - `mod.rs` (446): fluid_step(), fluid_step_karman(), pub re-exports, integration tests
  - `boundary.rs` (268): FieldType, BoundaryConfig, set_bnd(), set_bnd_rb(), set_bnd_karman()
  - `core.rs` (314): lin_solve(), diffuse(), advect(), project()
  - `params.rs` (85): SolverParams struct + Default + default_karman()
  - `thermal.rs` (106): inject_thermal_perturbation(), inject_heat_source(), apply_buoyancy()
  - `particle.rs` (224): interpolate_velocity(), advect_particles(), advect_particles_karman()
  - `karman.rs` (240): apply_mask(), inject_inflow(), inject_dye(), vorticity_confinement()
- **`renderer.rs` (1230行) → `src/renderer/` ディレクトリモジュール (3ファイル)**:
  - `mod.rs` (926): VizMode, RenderConfig, render_into(), render(), compute_vorticity(), particle rendering
  - `font.rs` (236): 5×7 bitmap font, glyph(), draw_text(), draw_text_sized(), render_status()
  - `color.rs` (92): Tokyo Night colormap (COLOR_STOPS, temperature_to_rgba()), color bar constants
- **`main.rs` (1283行) → 分割 (3ファイル)**:
  - `input.rs` (190): TermKey enum + parse_key() + 18 tests
  - `physics.rs` (190): PhysicsChannels, spawn_physics_thread(), ModelParams, create_sim_state()
  - `main.rs` (925): run_gui(), run_headless(), raw_term, integration tests
- Public API preserved — no changes needed in consuming code
- Multi-agent parallel refactoring (3 rust-architect agents working simultaneously)

## Test Summary
- **140 tests, all passing** (1 ignored: diagnostic)
- Includes 18 parse_key tests + 2 raw_term smoke tests + iTerm2 display dimension test
- ModelParams save_and_switch test + 2 interpolate_velocity precision tests
- `cargo test` succeeds with 0 failures

## Next Steps
- Fine-tune convection dynamics (plume count, speed, visual appeal)
- Issue #10: Simulation presets
