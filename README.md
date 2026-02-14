# fludarium

**A fluid dynamics aquarium for your desktop.**

fludarium is a real-time fluid dynamics simulator designed purely for visual enjoyment. No tropical fish or coral — just thermal convection plumes and Kármán vortex streets flowing quietly on your screen.

> *fluvius* (Latin: flow) + *-arium* (a place for) — a place to watch the flow.

<!-- TODO: Add screenshot/GIF here -->
<!-- ![fludarium](docs/screenshot.png) -->

## Design Concept

fludarium is a **desktop gadget for watching computational fluid dynamics (CFD)**.

Just as an aquarium frames a living ecosystem in glass, fludarium captures the beauty of fluid behavior — governed by real physics — in a small window. It's not a productivity tool. It's not a research simulator. It's a quiet little indulgence: watching beautiful flow.

- **Viewing first** — Enjoyable without touching anything. Even more fun if you tweak the parameters.
- **Physics-based** — Built on Jos Stam's "Stable Fluids" algorithm. Not just eye candy — a physically meaningful simulation.
- **Tokyo Night palette** — A 5-stop gradient from deep navy to sunset orange. Easy on the eyes, day or night.
- **Pure Rust** — No external C dependencies. Just `cargo build` and you're done.

## Simulation Models

### Rayleigh-Bénard Convection

Convection cells rising from a heat source at the bottom. Watch the regular pattern of thermal plumes emerge from the temperature differential.

### Kármán Vortex Street

A uniform flow past a cylinder produces a beautiful alternating trail of vortices — one of the most iconic phenomena in fluid dynamics. Supports particle trails and vorticity visualization.

## Install

### cargo install (Rust toolchain required)

```bash
cargo install --git https://github.com/daktu32/fludarium
```

### Homebrew (macOS)

```bash
brew tap daktu32/fludarium https://github.com/daktu32/fludarium
brew install fludarium
```

### Run

```bash
# GUI mode (native window)
fludarium

# Headless terminal mode
fludarium --headless
```

## Getting Started

### Requirements

- Rust toolchain (edition 2024) — if building from source
- macOS / Linux
- GUI mode: display server (X11 / Wayland / macOS)
- Headless mode: iTerm2 or WezTerm (iTerm2 Graphics Protocol support)

### Build from source

```bash
# Build
cargo build --release

# GUI mode (native window)
cargo run --release

# Headless terminal mode
cargo run --release -- --headless
```

## Controls

| Key | Action |
|-----|--------|
| `Space` | Toggle parameter panel |
| `M` | Switch simulation model |
| `V` | Toggle vorticity view (Kármán) |
| `↑` `↓` | Select parameter |
| `←` `→` | Adjust parameter |
| `,` `.` | Fine adjust |
| `R` | Reset to defaults |
| `Escape` / `Q` | Quit |

## Architecture

A 2-thread pipeline separates physics from rendering for smooth 60fps output.

```
Physics thread                              Main thread
┌──────────────┐                           ┌──────────────────────┐
│  SimState    │  FrameSnapshot            │  render()            │
│  fluid_step()│ ──── sync_channel(1) ───→ │  overlay             │
│              │                           │  window / terminal   │
└──────┬───────┘                           └──────────────────────┘
       ↑                                          │
       └──── mpsc (SolverParams) ─────────────────┘
                real-time parameter updates
```

## Technical Details

- **CFD Algorithm**: Jos Stam "Stable Fluids" — Semi-Lagrangian advection, Gauss-Seidel pressure projection, implicit diffusion
- **Grid**: 80×80 (Rayleigh-Bénard) / 80×aspect-scaled (Kármán)
- **Rendering**: 5-stop Tokyo Night colormap, adaptive contrast particles, bitmap font overlay
- **GUI**: minifb native window (60fps vsync)
- **Headless**: iTerm2 Graphics Protocol — RGBA → PNG → base64, adaptive render resolution with terminal resize support

## Credits

- CFD algorithm ported from [msakuta/cfd-wasm](https://github.com/msakuta/cfd-wasm)
- Jos Stam, ["Real-Time Fluid Dynamics for Games"](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf) (GDC 2003)

## License

MIT
