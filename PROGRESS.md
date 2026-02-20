# fludarium Progress

## Current Status: Quad-Model Fluid Simulator

238 tests passing. Codebase modularized into directory-based modules: `solver/` (10 files), `renderer/` (4 files), plus extracted `input.rs` and `physics.rs`. 2-thread pipeline (physics + render/display). Four simulation models (Rayleigh-Bénard convection + Kármán vortex street + Kelvin-Helmholtz instability + Lid-Driven Cavity) + RB benchmark mode (`--ra`/`--pr` CLI). N=80 grid with aspect-scaled NX for Kármán. Real-time parameter tuning via overlay panel in both GUI and headless modes. Headless terminal rendering via iTerm2 Graphics Protocol with full keyboard controls, adaptive render resolution, and dynamic terminal resize support. Per-model colormaps: TokyoNight (RB), SolarWind (Kármán), OceanLava (KH), ArcticIce (Cavity). Playback supports three grid types: spherical (equirectangular/orthographic), 1D periodic (line plot), and 2D channel (heatmap). No external config files — all defaults in code.

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
| **Headless (stored PNG + threaded I/O)** | **~50-60** | **None (within 16ms budget)** |

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

### Sub-issue 12: Kelvin-Helmholtz Instability Model
- **`FluidModel::KelvinHelmholtz`** — third simulation model: shear-driven interface instability
- **Physics**: tanh shear profile (`vx = U·tanh((y-N/2)/δ)`), sinusoidal vy perturbation (k=2,4), passive dye tracer
- **Boundary conditions**: X-periodic + Y free-slip (Neumann for vx, reflection for vy)
- **`solver/kh.rs` (new)**: `reinject_shear()` (wall-only relaxation toward target shear profile, 6 rows from each wall with fade)
- **`fluid_step_kh()`**: reinject shear → diffuse → project → advect → project → vorticity confinement → diffuse/advect dye → clamp → advect particles
- **Vorticity confinement**: reuses Kármán's `vorticity_confinement()` to counteract Semi-Lagrangian numerical diffusion — essential for cat's-eye vortex formation
- **Discovery-oriented defaults**: `confinement=0.0`, `shear_velocity=0.08` — conservative start, user tunes params to reveal dramatic KH vortex structures
- **`SolverParams::default_kh()`**: visc=0.001, diff=0.0005, dt=0.1, shear_velocity=0.08, confinement=0.0, shear_relax=1.0, shear_thickness=3.0
- **3 new SolverParams fields**: `shear_velocity`, `shear_relax`, `shear_thickness`
- **Overlay**: 7 adjustable parameters (visc, diff, dt, shear vel, confinement, relaxation, thickness)
- **3-model cycle**: M key cycles RB → Kármán → KH → RB (save/restore per-model params)
- **Renderer**: no changes needed (cylinder=None, tiles=1, existing colormap works for dye tracer)
- **No per-frame dye injection**: periodic boundaries mean initial dye field advects freely, enabling natural vortex roll-up visualization
- **9 QA tests + 3 unit tests**: initial conditions, shear profile, boundary conditions, dye interface, perturbation, fluid step stability, param consistency
- Multi-agent team development: physics-theorist (model design) + rust-architect (code architecture) + qa-agent (quality assurance)

### Per-Model Colormaps & Vorticity Visualization
- **`ColorMap` enum**: `TokyoNight` (RB), `OceanLava` (KH), `SolarWind` (Kármán), `ArcticIce` (Cavity) — model-specific palettes
- **Ocean & Lava**: deep blue → medium blue → white interface → orange → deep red — designed for KH 2-fluid mixing
- **Solar Wind**: deep space → indigo → violet → magenta → plasma gold — cosmic theme for Kármán dye wake
- **`map_to_rgba(t, colormap)`**: replaces `temperature_to_rgba()` (kept as `#[cfg(test)]` alias)
- **Signed vorticity visualization**: diverging blue(−)/red(+) with independent per-sign normalization
- **Color bar**: model-aware gradient, diverging for vorticity mode
- 11 new colormap tests (endpoints + gradient continuity for each palette, including ArcticIce)

### BGM Audio-Visual Sync
- **mpv IPC socket**: `--input-ipc-server` for programmatic control
- **Pre-warm pipeline**: spawn mpv paused+muted → unpause to trigger yt-dlp + audio pipeline init → wait for `time-pos` valid → re-pause + seek back + restore volume
- **Instant unpause**: on first rendered frame, IPC unpause is near-instant since stream is fully loaded
- **`mpv_ipc()` helper**: send JSON IPC command, read response with timeout
- **Cleanup**: `BgmGuard::drop` removes IPC socket file

### Window Title Bar
- **Model-aware title**: "fludarium ∣ Kármán Vortex · 45 fps" with proper Unicode (en-dash, accented letters, middle dot)
- **`model_label()` + `format_title()`** helper functions

### Sub-issue 13: Lid-Driven Cavity Model
- **`FluidModel::LidDrivenCavity`** — fourth simulation model: classic CFD benchmark
- **Physics**: square box with moving top wall (`lid_velocity=1.0`), all other walls no-slip, generates primary vortex + corner secondary vortices
- **Boundary conditions**: `BoundaryConfig::LidDrivenCavity { lid_velocity }` — top wall Dirichlet (vx=lid_velocity, vy negated), bottom/left/right no-slip (all velocity negated), scalars Neumann
- **`solver/cavity.rs` (new)**: `compute_velocity_dye()` — velocity magnitude `sqrt(vx²+vy²)` normalized to [0,1] mapped to temperature field for visualization
- **`fluid_step_cavity()`**: diffuse → project → advect → project → velocity dye → advect particles
- **`SolverParams::default_cavity()`**: visc=0.01, diff=0.001, dt=0.05, lid_velocity=1.0, project_iter=30
- **`solver/core.rs` fix**: advect X-clamping generalized to all non-periodic models (was Karman-only)
- **Overlay**: 4 adjustable parameters (visc, diff, dt, lid_velocity) with Re display
- **4-model cycle**: M key cycles RB → Kármán → KH → Cavity → RB (save/restore per-model params)
- **ArcticIce colormap**: deep void → dark teal → bright cyan → near white → bright mint — velocity magnitude visualization
- **15 new tests**: state (3), params (1), boundary (6), cavity (3), solver integration (2)
- Multi-agent team development: solver-agent (rust-architect) for core physics + team lead for UI/colormap

### Headless Performance Overhaul
- **Custom uncompressed PNG encoder**: Replaced `png` crate's `Compression::Fast` (zlib level 1) with hand-written stored deflate encoder — zero hash-table lookups, zero LZ77 matching. CRC-32 table generated at compile time, Adler-32 with NMAX chunking.
- **RGBA passthrough**: Switched from RGB (color_type=2) to RGBA (color_type=6), eliminating per-pixel RGBA→RGB strip loop (204,800 iterations → 320 per-row bulk memcpy).
- **Background encoder thread**: Offloaded PNG encode → base64 → stdout pipeline to dedicated thread via `sync_channel(1)`. Main loop only handles rendering (~2ms), I/O blocking no longer affects frame pacing.
- **Zero-copy buffer recycling**: Replaced `rgba_buf.clone()` (800KB/frame) with `std::mem::take` + return channel. Steady-state: 2 buffers circulating, zero allocations per frame.
- **60fps target**: `HEADLESS_FRAME_INTERVAL_MS` reduced from 33ms (30fps) to 16ms (60fps), matching GUI mode.
- **Frame skipping**: `try_send` with `TrySendError::Full` recovery provides natural backpressure — encoder-busy frames are dropped, buffer recovered immediately.
- 2 new PNG helper tests (CRC-32 + Adler-32 known values), 1 new roundtrip decode test via `png` crate.

### Lid-Driven Cavity: Aspect Ratio & Particle Fix
- **Pillarboxing**: `RenderConfig::fit_square()` preserves aspect ratio for square bounded domains (Cavity). Black bars on left/right sides center the simulation area within the window.
- **`display_x_offset` field**: All rendering paths (field, color bar, trails, particle heads, status bar, overlay panel) offset by `x_off` for correct pillarbox alignment.
- **`advect_particles_cavity()`**: X-axis ping-pong reflection (matching Y) replaces periodic wrap. Particles now circulate with the cavity flow instead of piling up on the right edge.
- **`make_render_cfg()` helper**: Dispatches `fit_square()` for Cavity, `fit()` for others — used in all 6 RenderConfig creation sites (GUI init, model switch, resize; headless init, model switch, resize).

## Test Summary
- **181 tests, all passing** (1 ignored: diagnostic)
- Includes 18 parse_key tests + 2 raw_term smoke tests + iTerm2 display dimension test
- ModelParams save_and_switch test (4-model cycle) + 2 interpolate_velocity precision tests
- 3 KH unit tests + 9 KH QA tests (initial conditions, shear maintenance, param defaults)
- 11 colormap tests (TokyoNight + OceanLava + SolarWind + ArcticIce endpoints and gradient continuity)
- 15 Lid-Driven Cavity tests (state, params, boundary, cavity dye, solver integration)
- CRC-32 + Adler-32 known values + PNG stored-deflate roundtrip decode test
- `cargo test` succeeds with 0 failures

### Distribution Setup
- Added `Cargo.toml` metadata: authors, description, license, repository, keywords, categories
- Created MIT LICENSE file
- Updated README.md with Install section (`cargo install` and `brew tap`/`brew install`)
- Verified `cargo install --path .` produces working `fludarium` binary
- Removed obsolete `PRD.md`

### spmodel-rs 連携: 球面データ再生モード
- **`--playback dir/`**: spmodel-rs の `.spg` 出力ディレクトリを指定して球面データ再生
- **SpgReader**: `.spg` バイナリ + `manifest.json` リーダー (standalone)
- **SphericalSnapshot**: 球面データ構造 + ガウス格子補間
- **PlaybackState**: 再生制御 (play/pause, seek, speed, field選択)
- **等距円筒図法 + 正射影**: P キーで切替
- **BlueWhiteRed**: 発散型カラーマップ (渦度等)
- **グラティキュール**: 緯度経度線 + ラベル (両投影対応)
- **グローバルカラーレンジ**: 全フレーム横断 min/max 事前計算、チカチカ防止
- **HUD フィールドバッジ**: 半透明ダーク背景の表示フィールド名 + インデックス
- **レンジパディング**: 微小変動フィールド (geopotential 等) の表示改善
- **粒子移流**: u_cos/v_cos フィールドがあれば自動で粒子トレーサーを表示
- **1D line plot**: KdV等の周期1Dデータを折れ線グラフで再生

### NetCDF (.nc) 再生対応 via gtool-rs
- **`--playback file.nc`**: gtool-rs の GtoolReader 経由で .nc ファイルを直接再生
- **ガウスノード**: NetCDF の `mu` 座標変数から直接読み込み (計算不要)
- **全機能対応**: global_ranges、粒子移流、HUD、グラティキュール全て .nc でも動作
- **2D/1D 自動判別**: Gaussian grid は球面再生、Periodic は 1D line plot

### 2D チャンネルヒートマップ再生
- **`renderer/heatmap.rs` (新規)**: 2D ヒートマップレンダラー
  - バイリニア補間でシミュレーション格子を画面解像度に拡大
  - 等値線 (16本) を modular arithmetic で1パスオーバーレイ
  - 軸フレーム + Z軸/X軸ラベル (物理座標値5目盛)
  - カラーバー + ティック値 + フィールドバッジ
  - 流星スタイル粒子トレイル描画 (Bresenham line + 5×5 soft glow)
- **`ChannelParticles`**: 流線関数 ψ ベースの粒子移流
  - `u = ∂ψ/∂z`, `w = -∂ψ/∂x` の中心差分で速度復元
  - 4サブステップ移流 + 32フレームリングバッファトレイル
  - X 周期境界、Z 壁反射
- **`PlaybackState` 拡張**: gtool-rs `GridType::Channel` 自動検出、domain (lx, lz) 読み込み、psi フィールド検索
- **GUI 統合**: ヒートマップ描画パス、チャンネル用キー制御 (正射影/P キー無効化)、Channel タイトルバー
- **3グリッドタイプ対応**: 球面 (Gaussian) → 等距円筒/正射影、1D (Periodic) → 折れ線、2D (Channel) → ヒートマップ
- 3 新テスト (heatmap config, buffer size, colored pixels)

### RB ベンチマーク比較モード
- **`--ra <Ra> --pr <Pr>`**: Rayleigh/Prandtl 数を指定してベンチマーク RB モードで起動
  - `SolverParams::from_ra_pr(ra, pr)`: 無次元化パラメータ自動計算 (diff=1/√(Ra·Pr), visc=Pr·diff, buoyancy=Ra·visc·diff)
  - `benchmark_mode: bool` フィールドで通常 RB とベンチマークを区別
- **`BoundaryConfig::RayleighBenardBenchmark`**: 均一 Dirichlet BC (底面 T=1.0, 上面 T=0.0), no-slip 壁, X 周期
- **`apply_buoyancy_perturbation()`**: 伝導プロファイル T_cond(y)=1-y/(N-1) からの偏差 θ=T-T_cond で浮力計算 → 射影での浮力消失を防止
- **`solver::diagnostics` (新規)**: compute_nusselt() (Nu=1+<vy·θ>/diff), compute_kinetic_energy(), compute_theta()
- **`fluid_step_benchmark()`**: inject_heat_source 省略 + 摂動浮力版の専用ステップ関数
- **`SimState::new_benchmark()`**: 伝導プロファイル + 正弦波摂動で初期化
- **`--export <path.nc>`**: GtoolWriter で theta/zeta/psi を Channel 形式で NetCDF 出力 (100ステップ間隔)
- **stderr 診断**: Nu 数と運動エネルギー KE をリアルタイム出力
- **`compute_vorticity`/`compute_stream_function`**: `pub(crate)` に昇格 (エクスポートで使用)
- 21 新テスト: params (3), boundary (3), thermal (2), diagnostics (6), solver integration (3), state (4)

### 渦中心軌跡トラッキング (Gyre Center Trajectory)
- **gyre center 検出**: 符号付き渦度最大値 (max vort) + パラボリック補間でサブグリッド精度の (lon, lat) 取得
- **NC プリコンピュート**: gtool-rs に `define_scalar`/`write_scalar`/`read_scalar` を追加、spmodel-rs の beta_gyre.rs で gyre_lon/gyre_lat をスカラータイムシリーズとして NC に埋め込み
- **フォールバック**: NC にスカラーがなければ fludarium 側で vort フィールドから自前計算
- **デフォルト表示制御**: NC に gyre_lon/gyre_lat がある場合のみデフォルト ON、vort のみの場合はデフォルト OFF (G キーでトグル可)
- **スクリーンブレンド描画**: 時間グラデーション (シアン→ゴールド) + ソフトグロー (3px 幅) + 放射状マーカー。暗背景でも明背景でも映える screen blend 方式
- **正射影 projection fix**: 順変換の経度回転符号を修正 (`R_y(-cam_lon)`)、粒子と gyre track の両方で視点回転時のずれを解消

### 球面描画 FPS 最適化
- **等距円筒図法**: `find_gauss_neighbors` を行単位にホイスト (ピクセル単位 → 行単位)、経度テーブル事前計算
- **正射影**: mu ルックアップテーブル (1024 エントリ) で二分探索を排除、`mu = y1` で asin+sin ラウンドトリップ回避
- **結果**: 正射影で ~10fps → ~50fps に改善

## Test Summary
- **238 tests, all passing** (1 ignored: diagnostic)
- 球面関連: graticule 4 tests, spherical interpolation/particle 7 tests, lineplot 3 tests
- heatmap: config/buffer/render 3 tests
- playback: gauss_nodes 4 tests
- font: glyph/draw_text/status 5 tests
- ベンチマーク: params 3, boundary 3, thermal 2, diagnostics 6, solver 3, state 4

## Next Steps
- Tag v0.1.0 and push for GitHub Release
- Create `daktu32/homebrew-fludarium` tap repository with formula
- Fine-tune convection dynamics (plume count, speed, visual appeal)
- Issue #10: Simulation presets
