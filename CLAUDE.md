# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

fluvarium is a fluid dynamics visualizer. It runs a Rayleigh-Benard convection simulation (localized hot spot at bottom center, cold top) and renders it via minifb native window. Written in Rust (edition 2024), single crate, no external C dependencies.

## Build & Test Commands

```bash
cargo build              # Debug build
cargo build --release    # Release build (recommended for running)
cargo test               # Run all 49 tests
cargo test solver        # Run only solver module tests
cargo test state         # Run only state module tests
cargo test renderer      # Run only renderer module tests
cargo test sixel         # Run only sixel module tests
cargo test test_name     # Run a single test by name
cargo run --release      # Run the simulation
```

## Architecture

2-thread pipeline:

```
Physics thread: SimState → fluid_step() → FrameSnapshot → sync_channel(1)
Main thread:    FrameSnapshot → render() → rgba_to_argb() → minifb window
```

### Module Responsibilities

- **`state.rs`** — `SimState` struct (velocity fields `vx/vy/vx0/vy0`, `temperature`, scratch buffers `work/work2`, `Xor128` PRNG, particles). Grid is `N=128`, flat `Vec<f64>` of `SIZE = N*N`. The `idx(x, y)` function handles 2D→1D with mod-N wrapping for horizontal periodicity.
- **`solver.rs`** — Jos Stam "Stable Fluids" CFD: `diffuse`, `advect` (Semi-Lagrangian), `project` (pressure projection), `fluid_step` (orchestrates one timestep). `SolverParams` holds all tuning constants (visc=0.008, diff=0.002, dt=0.003, buoyancy=8.0). Boundary conditions via `set_bnd(field_type, ...)` where field_type 0=scalar, 1=vx, 2=vy(negate at walls), 3=temperature(Dirichlet with Gaussian profile). `inject_heat_source` provides volumetric heating at bottom center + Newtonian cooling at top.
- **`renderer.rs`** — Maps temperature field to RGBA via 5-stop blackbody colormap (black→ember→crimson→orange→amber). `RenderConfig::fit()` computes layout for given pixel dimensions. Renders simulation tiled 2× horizontally with color bar + tick marks + adaptive contrast particles.
- **`sixel.rs`** — Custom `SixelEncoder` with fixed palette + LUT, and `encode_sixel` via icy_sixel. Both gated behind `#[cfg(test)]`.
- **`main.rs`** — minifb window (640×320), physics thread + main render/display thread, `ctrlc` handler, FPS counter in title bar.

### Coordinate System

- `y=0` is **bottom** (hot), `y=N-1` is **top** (cold) in simulation space
- Renderer flips Y for screen output (screen top = cold, screen bottom = hot)
- X-axis wraps (periodic boundary), Y-axis has wall boundaries (top/bottom)

### Key Constants

| Constant | Location | Value | Meaning |
|----------|----------|-------|---------|
| `N` | state.rs | 128 | Grid dimension |
| `SIZE` | state.rs | 16384 | N*N total cells |
| `WIN_WIDTH` | main.rs | 640 | Window width in pixels |
| `WIN_HEIGHT` | main.rs | 320 | Window height in pixels |
| `BOTTOM_BASE` | solver.rs | 0.15 | Base temperature at bottom boundary |

## Dependencies

- `icy_sixel` — Pure Rust Sixel encoder (test-only)
- `ctrlc` — Ctrl+C signal handler
- `minifb` — Lightweight native window for pixel buffer display

## Reference

CFD algorithm ported from [msakuta/cfd-wasm](https://github.com/msakuta/cfd-wasm). See `PRD.md` for full product requirements and `PROGRESS.md` for implementation status.
