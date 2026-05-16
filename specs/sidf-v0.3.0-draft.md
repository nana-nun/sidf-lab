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

## 6. Current Results

### Model C

Model C は同解像度 guide に対して、data fidelity と edge-aware interaction を使う基礎モデルである。保存形式つきの cross baseline と freeze benchmark が残っている。

参照:

- `results/2026-05-16-model-c-cross-baseline/notes.md`
- `results/2026-05-16-model-c-freeze-benchmark/notes.md`

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
- gradient-based confidence map がエッジ拘束として働く候補になっている。
- seeded texture term は再現可能な揺らぎを加えるが、現状では意味的ディテールではなく粒状ノイズに近い。

制限:

- Model D と nearest / bilinear / bicubic の保存形式つき metrics 比較は未完了である。
- Ground Truth 比較は未実施である。
- 斜線、曲線、soft gradient、実画像 patch での挙動は未確認である。

関連 Issue:

- [#6 Model D と bilinear/bicubic の比較指標を追加する](https://github.com/nana-nun/sidf-lab/issues/6)
- [#14 Model D を Guided Filter / guided upsampling と比較する](https://github.com/nana-nun/sidf-lab/issues/14)

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

Rust core decoder に移す前に、bit-perfect 再現性要件を Issue #16 で整理する。

関連 Issue:

- [#16 Rust移植に向けた deterministic PRNG と bit-perfect 再現性を調査する](https://github.com/nana-nun/sidf-lab/issues/16)

## 9. Open Questions

未確定事項は、既存の open Issue として分離している。

| Topic | Status | Issue |
| --- | --- | --- |
| Model D が bilinear / bicubic に対して何を改善するか | 未検証 | [#6](https://github.com/nana-nun/sidf-lab/issues/6) |
| Model D と guided filter / guided upsampling の関係 | 未整理 | [#14](https://github.com/nana-nun/sidf-lab/issues/14) |
| white noise 以外の structured texture prior | 未調査 | [#15](https://github.com/nana-nun/sidf-lab/issues/15) |
| MRF / Gibbs / stochastic relaxation との理論的整理 | 未整理 | [#12](https://github.com/nana-nun/sidf-lab/issues/12) |
| Rust 移植前の bit-perfect 再現性 | 未整理 | [#16](https://github.com/nana-nun/sidf-lab/issues/16) |

## 10. Draft-to-Spec Criteria

この draft を正式な仕様候補に近づけるには、少なくとも次を満たす必要がある。

1. Model D の baseline 比較実験を `results/` に保存する。
2. nearest / bilinear / bicubic との差分を metrics と画像で確認する。
3. soft gradient や実画像 patch で confidence map が不自然な硬化を起こさないか確認する。
4. texture prior を white noise baseline と structured noise 候補で比較する。
5. decoder seed、PRNG、丸め、更新順序を実装非依存に定義する。
6. binary layout と quantization を draft として別途定義する。

## 11. References

- `docs/sidf-research-notes.md`
- `docs/research-state.md`
- `docs/research-plan.md`
- `docs/repository-architecture.md`
- `results/2026-05-16-model-c-cross-baseline/notes.md`
- `results/2026-05-16-model-c-freeze-benchmark/notes.md`
