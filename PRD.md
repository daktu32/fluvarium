# fluvarium — Product Requirements Document

> Version: 0.1.0 (MVP)
> Date: 2026-02-11

## 1. 概要

**fluvarium** は、ターミナル上で流体力学シミュレーションをリアルタイム描画し、眺めて楽しむ鑑賞アプリケーションである。
名前は fluvius（ラテン語: 流れ）+ -arium（場所）に由来する。流体のアクアリウム。

## 2. MVP スコープ

### 2.1 やること

- 熱対流（Rayleigh-Benard convection）シミュレーションの実行
- 256x256 グリッドでの演算
- Sixel プロトコルによるターミナルへのピクセル描画
- 固定初期条件（下辺高温・上辺低温）
- best effort でのフレーム更新（FPS 固定なし）
- Ctrl+C による終了

### 2.2 やらないこと（MVP外）

- マウス/キーボードによるインタラクション
- 自然言語によるパラメータ指定（LLM 連携）
- 複数シミュレーションモードの切り替え
- 設定ファイル / CLI オプション
- WezTerm 以外のターミナル対応
- パーティクル描画・等高線表示

## 3. 技術仕様

### 3.1 技術スタック

| 項目 | 選定 |
|------|------|
| 言語 | Rust (edition 2021) |
| 構成 | 単一クレート (`cargo new fluvarium`) |
| CFD アルゴリズム | Jos Stam "Stable Fluids" ベース（[cfd-wasm](https://github.com/msakuta/cfd-wasm) から移植） |
| Sixel エンコード | [icy_sixel](https://github.com/mkrueger/icy_sixel)（Pure Rust、外部 C ライブラリ依存なし） |
| ピクセルバッファ | `image` クレート |
| ターミナル | WezTerm 前提（Sixel 対応） |

### 3.2 アーキテクチャ

```
┌──────────────────────────────────────────────┐
│  main loop                                   │
│                                              │
│  ┌────────────┐  ┌───────────┐  ┌─────────┐ │
│  │ CFD Engine │→│ Colormap  │→│  Sixel  │→ stdout
│  │ (solver)   │  │ (render)  │  │(encode) │ │
│  └────────────┘  └───────────┘  └─────────┘ │
│        ↑                                     │
│   State (vx, vy, temperature, density)       │
└──────────────────────────────────────────────┘
```

### 3.3 モジュール構成

```
src/
├── main.rs          # エントリポイント、メインループ
├── solver.rs        # CFD ソルバー（diffuse, advect, project, buoyancy）
├── state.rs         # シミュレーション状態（グリッド、速度場、温度場）
├── renderer.rs      # 状態 → ピクセルバッファ変換（カラーマップ適用）
└── sixel.rs         # Sixel エンコード・ターミナル出力
```

### 3.4 CFD アルゴリズム詳細

cfd-wasm から以下のコア部分を移植する。

#### 移植対象

| 関数 | 役割 |
|------|------|
| `lin_solve` | Gauss-Seidel 反復法による線形方程式の求解 |
| `diffuse` | 粘性拡散（陰的解法） |
| `advect` | Semi-Lagrangian 移流（バイリニア補間） |
| `project` | 圧力投影（非圧縮性の強制） |
| `fluid_step` | 1 タイムステップの統合処理 |
| 浮力項 | 温度勾配から垂直速度への寄与 |

#### 移植しない

- パーティクルシステム (`particles.rs`)
- 等高線描画 (`contour_lines.rs`, `marching_squares.rs`)
- WebGL レンダリング
- WASM バインディング

#### デフォルトパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| grid_size | 256 x 256 | シミュレーション格子サイズ |
| visc | 0.01 | 粘性係数 |
| diff | 0.0 | 拡散係数 |
| delta_time | 1.0 | タイムステップ |
| diffuse_iter | 4 | 拡散反復回数 |
| project_iter | 20 | 圧力投影反復回数 |
| heat_buoyancy | 0.05 | 浮力係数 |
| heat_exchange_rate | 0.2 | 熱交換率 |
| decay | 0.01 | 減衰率 |

### 3.5 描画仕様

- **解像度**: 256x256 ピクセル
- **カラーマップ**: 温度場に基づく。低温を青、高温を赤とするグラデーション
- **Sixel 出力**: 毎フレーム、カーソルをホームポジションに戻してから Sixel データを出力
- **フレームレート**: best effort（演算 + エンコード完了次第、次フレームへ）

### 3.6 初期条件（MVP）

- 下辺境界: 高温（固定温度源）
- 上辺境界: 低温（固定温度源）
- 左右境界: 周期境界（ラップアラウンド）
- 初期速度場: ゼロ + 微小な乱数擾乱
- 初期温度場: 線形グラデーション（下から上へ）

## 4. データフロー

```
1. State を初期条件で初期化
2. loop {
     a. fluid_step(&mut state)     // CFD 1ステップ演算
     b. render(&state) → ImageBuffer  // 温度場 → ピクセル
     c. encode_sixel(&image) → bytes  // Sixel エンコード
     d. print!("\x1b[H")             // カーソルをホームへ
     e. stdout.write_all(&bytes)     // Sixel 出力
     f. stdout.flush()
   }
3. Ctrl+C で終了（ctrlc クレートまたは signal handler）
```

## 5. 受け入れ基準

1. `cargo run` で WezTerm 上に熱対流のアニメーションが表示される
2. 下辺から上昇する対流セルが視認できる
3. フレームが連続的に更新され、ちらつきなくアニメーションする
4. Ctrl+C で正常終了する
5. 外部 C ライブラリへの依存がない（Pure Rust）

## 6. 将来構想（MVP 後）

- CLI オプションによるパラメータ指定（`clap`）
- 複数シミュレーションモード（カルマン渦、自由流れ等）
- マウスによる力の印加
- 自然言語 → パラメータ変換（LLM 連携）
- パーティクル / 等高線の可視化オプション
- ターミナルサイズに応じた解像度自動調整

## 7. 参考資料

- [Jos Stam "Stable Fluids" (GDC 2003)](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf)
- [Mike Ash's Fluid Simulation for Dummies](https://mikeash.com/pyblog/fluid-simulation-for-dummies.html)
- [msakuta/cfd-wasm](https://github.com/msakuta/cfd-wasm) — Rust 製 CFD の参考実装
- [icy_sixel](https://github.com/mkrueger/icy_sixel) — Pure Rust Sixel エンコーダ
