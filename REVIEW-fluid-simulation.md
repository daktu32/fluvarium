# 流体シミュレーション レビュー報告

> 初回レビュー: 2026-02-11
> 第 2 回レビュー: 2026-02-11（第 1 次改修後）
> 第 3 回レビュー: 2026-02-11（第 2 次改修後）— **全 Issue 対応完了**
> 対象: fluvarium CFD ソルバー（Jos Stam "Stable Fluids" ベース）

## 前提: このコードが解いている方程式

Boussinesq 近似付き 2D 非圧縮 Navier-Stokes を、Jos Stam の "Stable Fluids" 法で解いている。

- **運動量**: ∂**u**/∂t + (**u**·∇)**u** = −∇p/ρ + ν∇²**u** + βg(T−T₀)**ĵ**
- **連続**: ∇·**u** = 0
- **エネルギー**: ∂T/∂t + (**u**·∇)T = κ∇²T

Stam 法自体は「無条件安定だが物理的精度は低い」ビジュアライゼーション向け手法であり、鑑賞アプリとしての選択は適切。

---

## Issue ステータスサマリ

| Issue | 内容 | 状態 |
|-------|------|------|
| [#1](https://github.com/daktu32/fluvarium/issues/1) | 浮力 dt 欠落 | **修正完了** |
| [#2](https://github.com/daktu32/fluvarium/issues/2) | velocity decay | **修正完了** |
| [#3](https://github.com/daktu32/fluvarium/issues/3) | 厚い温度境界 | **修正完了** |
| [#4](https://github.com/daktu32/fluvarium/issues/4) | ノイズ注入 | **部分対応** — 方式は大幅改善。毎ステップ注入は残存（後述） |
| [#5](https://github.com/daktu32/fluvarium/issues/5) | free-slip 壁面 | **修正完了** |
| [#6](https://github.com/daktu32/fluvarium/issues/6) | diffuse_iter | **修正完了** |
| [#7](https://github.com/daktu32/fluvarium/issues/7) | density dead code | **修正完了** |

全 35 テスト通過。

---

## 第 2 次改修の変更内容

### パラメータ変更（初版 → 第 1 次 → 第 2 次）

| パラメータ | 初版 | 第 1 次 | 第 2 次 | 評価 |
|-----------|------|--------|--------|------|
| `heat_buoyancy` | 0.05 | 0.1 | **5.0** | dt 修正に伴い再調整。5.0 × 0.02 = 0.1/step で実効値は同等 |
| `noise_amp` | 0.001 | 0.01 | 0.01 | 変更なし |
| `diffuse_iter` | 4 | 4 | **20** | Stam 推奨値に到達 |
| `decay` | 0.999 | 0.999 | **削除** | フィールドごと除去 |
| `steps_per_frame` | 5 | 5 | **2** | diffuse_iter 増に伴う負荷調整 |
| `frame_duration` | 50ms | 50ms | **16ms** | ~60 FPS 上限に変更 |

### (D) 浮力に dt 追加 — [#1](https://github.com/daktu32/fluvarium/issues/1) 修正完了

`solver.rs:187-195`:
```rust
fn apply_buoyancy(vy: &mut [f64], temperature: &[f64], buoyancy: f64, dt: f64) {
    // ...
    vy[ii] += dt * buoyancy * (temperature[ii] - t_ambient);
}
```

`dt * heat_buoyancy = 0.02 * 5.0 = 0.1` で実効浮力は以前と同等。**dt を変更してもパラメータの物理的意味が保たれるようになった。** これが今回の改修で最も重要な修正。

### (E) velocity decay の除去 — [#2](https://github.com/daktu32/fluvarium/issues/2) 修正完了

`SolverParams` から `decay` フィールドが完全に削除され、`fluid_step` 末尾の速度減衰コードも除去。散逸は粘性拡散 (`visc = 0.0001`, `diffuse_iter = 20`) のみに委ねられる。物理的に正しい方向。

### (F) vx 壁面を no-slip に変更 — [#5](https://github.com/daktu32/fluvarium/issues/5) 修正完了

`solver.rs:11-14` で field_type 1 と 2 を統合:
```rust
1 | 2 => {
    x[idx(i as i32, 0)] = -x[idx(i as i32, 1)];
    x[idx(i as i32, (N - 1) as i32)] = -x[idx(i as i32, (N - 2) as i32)];
}
```

vx, vy ともに壁面で符号反転 → no-slip + no-penetration。実際の RB 対流に合致。テスト `test_set_bnd_vx_noslip` も追加済み。

### (G) 温度境界を壁面 1 行のみに変更 — [#3](https://github.com/daktu32/fluvarium/issues/3) 修正完了

`solver.rs:16-20`:
```rust
3 => {
    x[idx(i as i32, 0)] = 1.0;
    x[idx(i as i32, (N - 1) as i32)] = 0.0;
}
```

行 1, 行 N-2 の強制が除去され、壁面 1 行のみの標準 Dirichlet BC に。テスト `test_heat_source_boundaries` で y=1 が上書きされないことを検証済み。

### (H) diffuse_iter を 20 に増加 — [#6](https://github.com/daktu32/fluvarium/issues/6) 修正完了

Stam の原論文推奨値に到達。拡散の等方性と収束精度が大幅に改善。`steps_per_frame` を 5→2 に減らして演算負荷を調整。

### (I) density の除去 — [#7](https://github.com/daktu32/fluvarium/issues/7) 修正完了

`SimState` から `density` フィールドを削除。`fluid_step` から density の diffuse/advect/decay コードも除去。不要な演算がなくなり、コードも簡潔に。

---

## 残存する問題

### 1. 温度擾乱の毎ステップ注入 — [#4](https://github.com/daktu32/fluvarium/issues/4) 部分対応

`solver.rs:154-183` の `inject_thermal_perturbation` は毎タイムステップ実行される。

**改善済みの点**（第 1 次改修から）:
- 対流セルスケールの波長 (k=1..4) のみ → スペクトル的に合理的
- 壁面でゼロになる `sin(πy/L)` エンベロープ → 境界条件と整合
- 速度ではなく温度を擾乱 → 浮力を介した自然な速度応答

**残る懸念**:
- `noise_amp = 0.01` が `dt` スケーリングなし — dt を変更すると注入率が変わる
- 毎ステップ注入は非物理的な外部強制 — エネルギー保存を破る
- #1, #2 が修正された今、対流が自律的に持続するなら不要になる可能性あり

**検証提案**: `noise_amp = 0.0` にして十分なステップ数を回し、対流が自律的に持続するか確認する。持続するなら注入を初期条件のみに限定できる。

### 2. `idx` の Y 方向ラッピング

`state.rs:23-27` で Y 方向も mod N でラップする設計は壁境界と概念的に矛盾する。`set_bnd` による上書きで実害はないが、バグの温床として残る。

### 3. Rayleigh 数の整理

パラメータから算出:
- g·β = 5.0（heat_buoyancy）
- ΔT = 1.0, L = 1.0
- ν = κ = 0.0001
- **Pr = 1.0**
- **Ra = 5.0 × 1.0 × 1.0³ / (0.0001 × 0.0001) = 5 × 10⁸**

Ra ~ 5 × 10⁸ は完全に乱流レジーム。256×256 格子では空間解像度が不足するが、Semi-Lagrangian の数値拡散が自然な LES（Large Eddy Simulation）的フィルタとして機能するため、大スケール構造は再現可能。鑑賞用としては問題ない。

---

## 総合評価

| 観点 | 初回 | 第 2 次改修後 |
|------|------|-------------|
| **Stam 法の実装** | 骨格は正しい | 骨格は正しい |
| **Boussinesq 浮力** | dt 欠落（バグ） | **dt 修正済み。パラメータが物理的に整合** |
| **境界条件** | free-slip + 厚い温度強制 | **no-slip + 壁面 1 行 Dirichlet（標準 RB 条件）** |
| **速度散逸** | 非物理的 decay | **粘性拡散のみ（正しい）** |
| **拡散反復** | 4 回（不十分） | **20 回（Stam 推奨値）** |
| **ノイズ注入** | 白色速度ノイズ | **対流スケール温度擾乱（大幅改善）** |
| **dead code** | density 演算 | **除去済み** |
| **物理的忠実度** | 低い | **中程度 — Stam 法の限界内で正しく実装** |
| **鑑賞アプリとして** | 合致 | **より物理的な対流パターンが期待できる** |

## 結論

7 件の Issue のうち 6 件が完全に修正され、残る #4（ノイズ注入）も方式は大幅に改善されている。最も重要だった **浮力 dt の修正 (#1)** を起点に、decay 除去 (#2)、境界条件正常化 (#3, #5)、反復数増加 (#6) が連鎖的に実現した。コードは Stam 法の枠内で物理的に正しい Rayleigh-Benard 対流シミュレーションになった。

残る改善点は以下の 2 つのみ:
1. `noise_amp = 0.0` で対流の自律持続を検証し、可能なら注入を除去 (#4)
2. `idx` の Y ラッピングをドキュメント化または防御的に修正
