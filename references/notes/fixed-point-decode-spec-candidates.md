# 固定小数点デコード仕様候補メモ

## Source

Related Issue: [#77](https://github.com/nana-nun/sidf-lab/issues/77)

Primary references:

- [Deterministic PRNG と Bit-perfect 再現性](deterministic-prng-bit-perfect.md)
- [Rust core PRNG test vector spike](rust-core-prng-test-vector.md)
- [The Rust Reference: Operator expressions / Overflow](https://doc.rust-lang.org/reference/expressions/operator-expr.html#overflow)
- [Rust `u32` primitive integer methods](https://doc.rust-lang.org/std/primitive.u32.html)
- [fixed crate documentation](https://docs.rs/fixed/)
- [SIDF v0.2.1 draft](../../specs/sidf-v0.2.1.md)
- [SIDF v0.3.0 draft](../../specs/sidf-v0.3.0-draft.md)

参照日: 2026-05-30

## Summary

このメモは、SIDF Rust core の Model C 最小アニーリングループへ進む前に、固定小数点デコードで決めるべき値表現、丸め、overflow、acceptance 判定、state 更新の候補を整理する。

結論として、現時点では正式仕様を確定しない。まずは「Python float版Model Cと近い挙動を観察する移植」と「実装非依存のbit-perfect decoder」を分け、後者では PRNG、値スケール、丸め、overflow、update trace をすべてtest vector化する必要がある。

## Current SIDF Context

現行 Python の Model C は、`float64` の state と guide を使い、局所 energy を `lambda_data * (v_i - s_i)^2 + lambda_smooth * weight * (v_i - v_j)^2` として計算している。`src/sidf_lab/anneal.py` では、seed付き `np.random.default_rng`、pixel permutation、Gaussian proposal、`np.clip`、`math.exp(-delta / temp)` による Metropolis 判定を使っている。

この実装は研究プロトタイプとして有用だが、次の理由でそのまま実装非依存の仕様にはしにくい。

- `float64` の演算順序、丸め、`exp` 実装、proposal sampling が仕様として固定されていない。
- NumPy RNG の乱数消費順に state 更新が依存する。
- `np.clip`、境界処理、neighbor order、acceptance の等号条件を明示しないと、Rust側で同じtraceを作れない。
- Rust の通常整数演算は overflow check の有無により挙動確認が必要になるため、decoder仕様では `wrapping`、`saturating`、`checked` などを明示する必要がある。

## Fixed and Candidate Items

### 仕様としてほぼ固定したい事項

- serialized guide は decode 前に整数gridとして扱う。PNG出力用の丸めと decoder 内部値の丸めは分ける。
- pixel traversal は最初のRust spikeでは row-major に固定する。counter-based PRNG を使う場合も、counter layout は test vector に保存する。
- PRNG は #50 の spike に合わせ、最初は `Philox4x32-10` test vector を候補として使う。ただし正式採用は未決定。
- acceptance 判定では `delta_energy <= 0` を常にacceptするか、`delta_energy < 0` のみ無条件acceptにするかを仕様で決める。Python現行実装は `delta < 0.0` を使っている。
- every accepted update は state に即時反映する。同期更新に変える場合は別モデルとして扱う。

### 未確定の設計候補

| Area | Candidate | Benefit | Risk |
| --- | --- | --- | --- |
| pixel value | `u16` Q0.16 | 0から1未満の正規化値を小さく扱える | 1.0を表す端点、PNG丸め、差分二乗の精度を決める必要がある |
| pixel value | `u32` Q0.24 または Q0.32 | proposalやenergy差の細かさを残しやすい | accumulator幅と演算コストが増える |
| energy accumulator | signed `i64` / `i128` integer accumulator | deltaの符号比較を明示しやすい | scale設計を誤ると簡単にoverflowまたは過剰精度になる |
| pairwise weight | guide差から固定lookup table | `exp(-gamma * diff^2)` をdecode時にfloat計算しない | table生成規則と補間なしの量子化誤差を決める必要がある |
| temperature | sweepごとの固定table | acceptance比較が安定しやすい | parameterを自由に変える実験には不便 |
| proposal | integer delta table | Gaussian float samplingを避けられる | Python float版と同じ挙動にはならない |
| acceptance probability | threshold table for `exp(-delta/temp)` | decoder内のtranscendental functionを避けられる | table resolutionでaccept率が変わる |
| overflow | saturating arithmetic | pixel値やenergyの破綻を抑えやすい | saturationがenergy地形を歪める可能性がある |
| overflow | checked arithmetic in debug/test vector | 設計ミスを早く検出できる | release decoderの仕様には直接ならない |
| overflow | wrapping arithmetic | PRNGやhashでは自然 | energyやpixel値では意図しない急変を起こしやすい |

## Rounding Rules To Decide

固定小数点化では、単にbit幅を選ぶだけでは足りない。少なくとも次を明示する。

1. guide読み込み時: 8-bit / 16-bit入力を内部scaleへ写す丸め。
2. value proposal: integer deltaを足した後の下限上限処理。
3. subtraction: `v_i - s_i` と `v_i - v_j` を signed value として扱うか、abs差分として扱うか。
4. square: 差分二乗後に何bit右shiftするか、丸めは floor / nearest / ties-to-even のどれか。
5. weight multiplication: `lambda` と `weight` の積をいつscale調整するか。
6. neighbor sum: neighborごとに丸めるか、sumしてから丸めるか。
7. delta energy: old/new energy の差を signed accumulator で比較するか、別scaleへ変換するか。
8. acceptance threshold: PRNG出力を `[0, 1)` 固定小数点に写す丸め。
9. output: final stateをPNGやmetrics用floatへ変換する丸め。decoder互換性の対象は内部stateか、serialized outputかを分ける。

## Candidate Minimal Model C Fixed-point Trace

Issue #76 の Rust core 最小ループへ渡しやすい最小trace候補は次の形にする。

```text
input:
  width, height
  guide_u16_q0_16[row_major]
  initial_state_u16_q0_16[row_major] or initialization counter rule
  decoder_seed
  sweeps
  lambda_data_q
  lambda_smooth_q
  pair_weight_table_id
  proposal_table_id
  temperature_table_id

for sweep in 0..sweeps:
  for pixel_index in row_major:
    old_value = state[pixel_index]
    proposal_delta = table(prng(stage=anneal, sweep, pixel_index, purpose=proposal))
    new_value = clamp_value(old_value + proposal_delta)
    old_energy = local_energy_fixed(old_value)
    new_energy = local_energy_fixed(new_value)
    delta = new_energy - old_energy
    threshold = prng(stage=anneal, sweep, pixel_index, purpose=accept)
    accept if fixed_accept(delta, threshold, temperature[sweep])
    if accept:
      state[pixel_index] = new_value
```

このtraceは、Python float版との完全一致を狙うものではない。まずは、同じ整数入力、seed、table、sweep数から Rust とテストoracleが同じ state hash を返すことを目標にする。

## Rust Implementation Notes

- 通常の `+`、`*` に decoder互換性を委ねず、意図した演算を関数名または型で明示する。Rust公式ドキュメント上も `wrapping_*`、`saturating_*`、`overflowing_*` は別の意味を持つ。
- PRNG内の modular arithmetic は `wrapping_*` が自然だが、energy accumulatorでは原則として `checked_*` で設計余裕を確認し、必要なら広い型へ上げる。
- pixel value のclampは saturating arithmetic だけに任せず、`0..=MAX_VALUE` の意味を明示する。
- `fixed` crate は候補だが、SIDF仕様ではcrate APIそのものではなく、bit幅、scale、丸め、overflow規則を文章とtest vectorで定義する。
- `fixed` crate が提供しない `exp` などの解析関数に依存しない形へ acceptance rule と pairwise weight を寄せる。
- `no_std` を急がない。まずは `cargo test` で固定traceとoverflow検出が読める小さい実装を優先する。

## Relevance to SIDF

固定小数点化は、画質、圧縮率、super-resolution 性能を改善するための主張ではない。SIDFにとっての主な価値は、seed と保存されたパラメータから同じdecode過程を再現できるようにすることである。

Model C freeze 後の Rust core では、Pythonが担う実験・可視化・metricsと、Rustが担う bit-perfect decoder loop を分けるのが安全である。Python側で同じ固定小数点decoderを再実装する場合も、production pathではなくtest vector検証用の独立oracleとして扱う。

## Limitations

- このメモは仕様候補であり、固定小数点のbit幅やrounding modeを確定していない。
- ここで提案したtraceは、Python float版Model Cの画像と同一になる保証を持たない。
- acceptance probabilityをtable化する場合、画質や収束挙動は再評価が必要になる。
- `fixed` crate の利用可否、Rust version、dependency policyはまだ決めていない。
- Model D の confidence map、texture prior、multi-resolution pipeline には踏み込んでいない。

## Follow-up

- Issue #76 では、まず Model C の `local_energy_fixed` と1 sweepのupdate traceを小さいguideで検証する。
- 固定trace用に、`references/fixed-point-test-vectors/` または `rust/sidf-core/tests/fixtures/` に tiny guide、config、expected state hash を保存する。
- energy accumulator の scale 設計は、cross / diagonal / soft gradient の代表値で overflow余裕を見積もってから決める。
- acceptance table と proposal table は、最初は画質最適化ではなく再現性確認用の小さなtableとして追加する。
