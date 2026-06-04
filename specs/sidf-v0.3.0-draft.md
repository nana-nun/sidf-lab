# SIDF v0.3.0 Draft Specification

Status: Draft
Date: 2026-05-16

この文書は、SIDF v0.3.0 の確定仕様ではなく、Model D までの研究状態をもとにした draft 仕様案である。現段階の SIDF は実用圧縮形式ではなく、低解像度 STATIC guide、seed、物理パラメータ、決定論的な確率的緩和過程による画像再構成を検証する研究対象として扱う。

この draft は、特に次の主張をしない。

- PNG、JPEG、AVIF、JPEG XL、neural codec より高圧縮である。
- Ground Truth 比較なしに超解像性能がある。
- 自然画像の失われた高周波ディテールを復元できる。
- 環境非依存の bit-perfect 再現性が実装済みである。
- 大画像で実用的な decode time が得られている。

## 1. Scope

SIDF v0.3.0 draft の対象は、grayscale の confidence-aware multi-resolution reconstruction である。

対象に含めるもの:

- 低解像度 STATIC guide
- target output shape
- bilinear upscaled guide
- gradient-based confidence map
- edge-aware interaction
- seeded texture term
- deterministic stochastic relaxation pipeline
- baseline との差分評価

対象外、または未確定のもの:

- RGB / YCbCr / alpha channel
- 実用ファイルフォーマットとしての互換性保証
- entropy coding や bitstream 最適化
- neural decoder
- Rust 固定小数点 decoder の正式仕様
- 環境非依存の bit-perfect guarantee

## 2. Terms

STATIC guide:

低解像度または同解像度の grayscale guide。現段階では、復元対象の構造を弱く示す入力画像として扱う。

Upscaled guide:

STATIC guide を target output shape へ補間した画像。Model D の現行案では bilinear upscaling を使う。

Confidence map:

Upscaled guide の gradient magnitude などから作る画素ごとの拘束強度。エッジ付近では data fidelity を強くし、平坦部では相互作用や texture term の影響を相対的に許す候補として扱う。

Texture prior:

seed から決定論的に生成される texture-like variation。現状の white noise texture は粒状感に近く、意味的ディテールではない。

Decoder seed:

緩和過程、proposal、texture prior などを再現するための seed。現状は Python / NumPy 実装上の同一環境再現性を対象とし、Rust 移植後の bit-perfect 再現性は未確定である。

## 3. Draft Data Model

SIDF v0.3.0 draft は、少なくとも次の情報を持つ候補形式として整理する。

```text
sidf_version: "0.3.0-draft"
color_mode: "grayscale"
static_guide:
  width: integer
  height: integer
  values: grayscale array
output:
  width: integer
  height: integer
upscale:
  method: "bilinear"
confidence:
  method: "gradient"
  min_confidence: float
  max_confidence: float
energy:
  model: "model_d"
  lambda_data: float
  j_base: float
  gamma: float
  texture_strength: float
texture:
  method: "white_noise"
  seed: integer
anneal:
  decoder_seed: integer
  sweeps: integer
  temp_start: float
  temp_end: float
  proposal_sigma: float
```

この構造は保存形式の候補であり、binary layout、endianness、quantization、checksum、version negotiation は未定義である。

## 4. Reconstruction Pipeline

v0.3.0 draft の Model D pipeline は、次の順序で表現する。

```text
low-resolution STATIC guide
-> normalize grayscale values to [0, 1]
-> bilinear upscale to target output shape
-> compute gradient-based confidence map
-> generate deterministic texture prior from seed
-> run edge-aware stochastic relaxation
-> output grayscale reconstruction
```

この pipeline は、低解像度 guide から高解像度 output を生成する候補である。ただし、現段階では「超解像」と呼ぶより、confidence-aware multi-resolution reconstruction と呼ぶのが正確である。

## 5. Energy Model

Model D の候補 energy は次の形で記述する。

```text
E =
  lambda_data * sum_i c_i (v_i - s_i)^2
  + sum_(i,j) J_ij (v_i - v_j)^2
  + texture_strength * sum_i t_i v_i

J_ij = J_base * exp(-gamma * (s_i - s_j)^2)
```

記号:

- `v_i`: output pixel value
- `s_i`: upscaled guide value
- `c_i`: confidence value
- `t_i`: deterministic texture prior value
- `J_ij`: guide difference に応じた edge-aware interaction

この式は draft の候補であり、係数、符号、texture term の扱い、temperature schedule は今後の実験で変更される可能性がある。

### 確率モデルとしての扱い

Model C / D の energy は、MRF / Gibbs 型の画像復元と概念的に対応する `data fidelity term` と `edge-aware pairwise interaction term` を持つ。ただし、この draft では観測モデル、prior、posterior distribution を formal には定義していない。

そのため、この energy は現時点では確率モデルそのものではなく、seed つき緩和 decoder が低減しようとする decoder objective として扱う。`lambda_data * (v_i - s_i)^2` は通常 `data fidelity` と呼び、Gaussian observation model を別途仮定する場合に限って data likelihood と対応づける。`J_ij * (v_i - v_j)^2` は pairwise prior に近い役割を持つが、`J_ij` が guide `s` に依存するため、厳密な MRF prior と断定しない。

Model C / D を posterior energy や MAP 推定として記述するには、guide の観測モデル、latent image の prior、`J_ij` の確率モデル上の位置づけ、confidence map や texture field の意味、continuous value の扱い、annealing decoder の推定上の意味を別途定義する必要がある。

詳細な整理は `docs/model-c-energy-position.md` を参照する。

## 6. Current Results

### Model C

Model C は同解像度 guide に対して、data fidelity と edge-aware interaction を使う基礎モデルである。保存形式つきの cross baseline と freeze benchmark が残っている。

参照:

- `results/2026-05-16-issue-4-model-c-cross-baseline/notes.md`
- `results/2026-05-16-issue-5-model-c-freeze-benchmark/notes.md`

現在の結果:

- cross baseline では、Model C の `MAD = 0.011684`、`Background Mean = 0.007409`、`Edge Leakage = 0.008887` が保存された。
- freeze benchmark では cross、diagonal、circle、thin line、soft gradient の synthetic guides に対して `config.json`、`metrics.json`、`notes.md`、主要 PNG が保存された。
- hard edge shape では `Model C MAD <= 0.010742`、`Background Mean <= 0.006267`、`Edge Leakage <= 0.007102` が確認された。

解釈:

Model C は、少なくとも synthetic grayscale guide では背景漏れを抑えた安定化モデルとして有望である。ただし、これは実用圧縮性能や自然画像での一般性能を示すものではない。

### Model D

Model D は、16x16 guide から 64x64 output を生成する confidence-aware multi-resolution reconstruction の候補である。

観察:

- bilinear upscaled guide を初期的な構造拘束として使う。
- gradient-based confidence map はエッジ拘束として働く候補だが、保存済みの切り分け実験では uniform confidence より有利とは確認できていない。
- seeded texture term は再現可能な揺らぎを加える候補だが、現状の white-noise texture は意味的ディテールではなく粒状ノイズに近く、baseline 改善要因としては確認できていない。

保存済み結果:

- shape benchmark では、diagonal、circle、thin line、soft gradient で nearest / bilinear / bicubic / Model D candidate を保存した。
- cross comparison では、Model D の `MAD = 0.047106` が nearest `0.013794`、bilinear `0.033143`、bicubic `0.035119` より悪かった。
- cross comparison では、Model D の edge leakage `0.220617` が bilinear `0.219728` とほぼ同程度で、nearest `0.128409` より悪かった。
- natural patch GT evaluation では、Model D の `MAD = 0.055577` と global SSIM `0.948710` が nearest / bilinear / bicubic baseline を上回らなかった。
- natural patch では、明確な foreground/background 境界がないため edge leakage は使わず、gradient MAD と strong-edge MAD を代替指標として保存した。
- texture ablation では、`texture_strength=0.00, 0.10, 0.35, 0.70` を比較したが、非ゼロ white-noise texture_strength は synthetic cross の baseline 指標を改善しなかった。
- weight grid では、cross と natural patch の両方で `flat_conf_tex0` が Model D grid 内の最小MADだったが、nearest / bilinear / bicubic baseline は上回らなかった。
- term isolation では、cross と natural patch の両方で `data_pairwise_uniform` が term conditions 内の最小MADだったが、nearest / bilinear / bicubic baseline は上回らなかった。`pairwise_only` は data fidelity なしの復元条件として不十分だった。

解釈:

現行 Model D candidate は、confidence-aware multi-resolution reconstruction の候補としての観察価値はあるが、保存済みの synthetic shape、cross、1枚の自然画像 patch、および #37 / #56 / #61 の切り分けでは、nearest / bilinear / bicubic baseline に対する総合的な改善を示していない。特に現行の white-noise texture、gradient-based confidence map、data fidelity / pairwise interaction の組み合わせは、現在の設定では Ground Truth 差分や背景漏れを改善していない。

この結果は、confidence map や stochastic relaxation という研究方向全体を否定するものではない。一方で、現行 Model D の式をそのまま重み探索するだけでは baseline 改善に届きにくいため、confidence map の作り方や pairwise term の形を別候補として再設計する必要がある。

制限:

- 結果は少数の grayscale synthetic shape と1枚の自然画像 crop に限られる。
- 現行の white-noise texture term は意味的ディテールではない。
- confidence map、texture prior、data fidelity、pairwise interaction は、候補要素としては分離して記録されたが、現行設計で baseline 改善を示した要素はまだない。
- 現行実装の texture path は、この draft に書かれた線形 texture term そのものではなく、texture target 二乗項と初期状態混入を含むため、仕様候補として再整理が必要である。
- この結果は super-resolution や compression の成立を示さず、またそれらを否定する一般結果でもない。

関連 Issue:

- [#37 texture termの寄与をablationで検証する](https://github.com/nana-nun/sidf-lab/issues/37)
- [#56 Model Dのconfidence/data/texture重みを小規模gridで再評価する](https://github.com/nana-nun/sidf-lab/issues/56)
- [#61 Model Dのdata fidelity / pairwise / confidence項を分離比較する](https://github.com/nana-nun/sidf-lab/issues/61)
- [#67 Model Dのconfidence mapとpairwise termの再設計候補を比較する](https://github.com/nana-nun/sidf-lab/issues/67)

## 7. Baseline and Metrics Requirements

v0.3.0 draft を実験として評価するときは、少なくとも次の baseline を保存する。

- nearest upscaling
- bilinear upscaling
- bicubic upscaling
- SIDF rendered output

保存すべき artifacts:

- low-res guide PNG
- upscaled guide PNG
- confidence map PNG
- baseline PNGs
- rendered PNG
- difference map PNG
- `config.json`
- `metrics.json`
- `notes.md`

metrics 候補:

- MAD
- foreground / background mean
- foreground / background variance
- edge leakage
- edge width
- PSNR
- SSIM
- decode time

`notes.md` では、`Question`、`Hypothesis`、`Setup`、`Baseline`、`Result`、`Interpretation`、`Limitations`、`Next` を分ける。

## 8. Determinism Requirements

現段階の determinism target は、Python / NumPy の同一環境で同じ seed から同じ結果を再生成できることである。

Rust 移植前に固定すべき項目:

- PRNG algorithm
- seed expansion
- per-pixel update order
- proposal distribution
- floating-point or fixed-point arithmetic
- rounding rule
- boundary handling
- image normalization and quantization
- texture prior generation

Rust core decoder に移す前に、bit-perfect 再現性要件、test vector、Rust実装の照合手順を段階的に固定する。

関連 Issue:

- [#16 Rust移植に向けた deterministic PRNG と bit-perfect 再現性を調査する](https://github.com/nana-nun/sidf-lab/issues/16)
- [#50 Rust core向けcounter-based PRNG test vectorを追加する](https://github.com/nana-nun/sidf-lab/issues/50)
- [#55 Rust coreにPhilox4x32-10の最小実装を追加する](https://github.com/nana-nun/sidf-lab/issues/55)

## 9. Open Questions

未確定事項は、既存の Issue として分離している。完了済みの比較Issueから得た結果は、次の切り分けIssueへ接続する。

| Topic | Status | Issue |
| --- | --- | --- |
| Model D が bilinear / bicubic に対して何を改善するか | 保存済み比較では総合改善なし | [#6](https://github.com/nana-nun/sidf-lab/issues/6), [#30](https://github.com/nana-nun/sidf-lab/issues/30), [#36](https://github.com/nana-nun/sidf-lab/issues/36) |
| white-noise texture term の寄与 | #37 では baseline 改善なし | [#37](https://github.com/nana-nun/sidf-lab/issues/37) |
| confidence / data / texture 重みの切り分け | #56 の小規模gridでは baseline 改善なし | [#56](https://github.com/nana-nun/sidf-lab/issues/56) |
| data fidelity / pairwise / confidence の項分離 | #61 の term isolation では baseline 改善なし | [#61](https://github.com/nana-nun/sidf-lab/issues/61) |
| confidence map / pairwise term の再設計候補 | 次に比較する | [#67](https://github.com/nana-nun/sidf-lab/issues/67) |
| structured texture prior の実験利用 | helper追加済み、white noise と texture_strength=0 を対照に評価する | [#48](https://github.com/nana-nun/sidf-lab/issues/48), [#63](https://github.com/nana-nun/sidf-lab/issues/63) |
| Rust core の PRNG 実装 | test vector保存済み、Rust実装は未実施 | [#50](https://github.com/nana-nun/sidf-lab/issues/50), [#55](https://github.com/nana-nun/sidf-lab/issues/55) |
| Model C と Perona-Malik 型 diffusion の違い | 直接比較または比較不能な理由を整理する | [#40](https://github.com/nana-nun/sidf-lab/issues/40) |

## 10. Draft-to-Spec Criteria

この draft を正式な仕様候補に近づけるには、少なくとも次を満たす必要がある。

1. 現行 Model D が baseline を上回っていない結果を前提に、confidence map と pairwise term の再設計候補を比較する。
2. 現行 Model D の energy 式を正式化する前に、data fidelity、pairwise interaction、confidence weighting、texture prior の役割と符号を再定義する。
3. soft gradient や実画像 patch で confidence map が不自然な硬化を起こさないか、Ground Truth または代替指標とともに確認する。
4. structured texture prior を使う場合は、texture_strength=0 と white noise baseline との差分で評価し、意味的ディテール生成とは断定しない。
5. decoder seed、PRNG、丸め、更新順序を実装非依存に定義する。
6. binary layout と quantization を draft として別途定義する。

## 11. References

- `docs/sidf-research-notes.md`
- `docs/research-state.md`
- `docs/research-plan.md`
- `docs/repository-architecture.md`
- `docs/model-c-energy-position.md`
- `results/2026-05-16-issue-4-model-c-cross-baseline/notes.md`
- `results/2026-05-16-issue-5-model-c-freeze-benchmark/notes.md`
- `results/2026-05-16-issue-30-model-d-shape-benchmark/notes.md`
- `results/2026-05-17-issue-6-model-d-cross-comparison/notes.md`
- `results/2026-05-17-issue-36-model-d-natural-patch/notes.md`
- `results/2026-05-24-issue-37-texture-ablation/notes.md`
- `results/2026-05-24-issue-56-model-d-weight-grid/notes.md`
- `results/2026-05-24-issue-61-model-d-term-isolation/notes.md`
