# SIDF v0.2.1 Draft Specification

Status: Draft
Date: 2026-05-17

この文書は SIDF v0.2.1 の確定仕様ではなく、Model C freeze benchmark の結果をもとに整理した draft 仕様案である。

現段階の SIDF は実用圧縮形式ではなく、低解像度または同解像度の grayscale guide、seed、物理パラメータ、決定論的な確率的緩和過程による画像再構成を検証する研究対象として扱う。

この draft は、特に次を主張しない。

- PNG、JPEG、AVIF、JPEG XL、neural codec より高圧縮である。
- Ground Truth 比較なしに超解像性能がある。
- 自然画像の一般的な復元性能が確認済みである。
- 環境非依存の bit-perfect 再現性が実装済みである。
- Python 実装の decode time が大画像で実用的である。

## 1. Scope

SIDF v0.2.1 draft の対象は、Model C による同解像度 grayscale reconstruction である。

対象に含めるもの:

- grayscale STATIC guide
- guide と同じ output shape
- data fidelity term
- edge-aware pairwise interaction term
- seed つき stochastic relaxation decoder
- synthetic guides での保存形式つき benchmark

対象外、または未確定のもの:

- RGB / YCbCr / alpha channel
- low-resolution guide から high-resolution output への Model D pipeline
- entropy coding や実用 bitstream
- binary layout、endianness、checksum、version negotiation
- Rust 固定小数点 decoder の正式仕様
- 実装非依存の bit-perfect guarantee

## 2. Terms

STATIC guide:

Model C では、output と同じ解像度を持つ grayscale guide。現行実験では synthetic guide に deterministic Gaussian noise を加えたものを主に使う。

Decoder state:

緩和過程で更新される output candidate `v`。各 pixel value は `[0, 1]` の連続値として扱う。

Data fidelity:

output candidate `v_i` を guide value `s_i` に近づける二乗誤差項。通常は `data likelihood` ではなく `data fidelity` と呼ぶ。Gaussian observation model を別途仮定する場合に限って、負の対数尤度と対応づけられる。

Edge-aware interaction:

guide 上で差が小さい近傍ほど output candidate を近づけ、差が大きい近傍では結合を弱める pairwise interaction。エッジをまたぐ平滑化を抑えるための項として扱う。

Decoder seed:

初期 state、pixel update order、proposal sampling を再現するための seed。現状は Python / NumPy の同一環境再現性を対象とし、実装非依存の bit-perfect 再現性は未確定である。

## 3. Draft Data Model

SIDF v0.2.1 draft は、少なくとも次の情報を持つ候補形式として整理する。

```text
sidf_version: "0.2.1-draft"
color_mode: "grayscale"
static_guide:
  width: integer
  height: integer
  values: grayscale array
output:
  width: same as static_guide.width
  height: same as static_guide.height
energy:
  model: "model_c"
  lambda_data: float
  j_base: float
  gamma: float
anneal:
  decoder_seed: integer
  sweeps: integer
  temp_start: float
  temp_end: float
  proposal_sigma: float
```

この構造は保存形式候補であり、binary encoding、quantization、圧縮方法、forward/backward compatibility は未定義である。

## 4. Reconstruction Pipeline

Model C pipeline は、次の順序で表現する。

```text
STATIC guide
-> normalize grayscale values to [0, 1]
-> initialize decoder state from decoder_seed
-> run stochastic relaxation with Model C local energy
-> output grayscale reconstruction
```

現行 Python 実装では、各 sweep で全 pixel を seed 由来の permutation order で更新し、proposal を Gaussian noise から生成する。accept/reject は Metropolis 型の判定を使う。

この手順は、低 energy state を探索する decoder procedure である。formal な posterior sampling または MAP 推定としてはまだ定義していない。

## 5. Energy Model

Model C の local energy は、pixel `i` の candidate value `v_i` に対して次の形で扱う。

```text
E_i(v_i) =
  lambda_data * (v_i - s_i)^2
  + sum_(j in N(i)) J_ij * (v_i - v_j)^2

J_ij = j_base * exp(-gamma * (s_i - s_j)^2)
```

記号:

- `v_i`: output candidate の pixel value
- `s_i`: STATIC guide の pixel value
- `N(i)`: 4-neighborhood
- `J_ij`: guide difference に応じた edge-aware interaction
- `lambda_data`: guide への拘束強度
- `j_base`: pairwise interaction の基準強度
- `gamma`: guide difference による結合減衰の強さ

境界条件:

- 現行 Python 実装は torus wrapping を使わない。
- 画像外の近傍は存在しないものとして扱う。

値域:

- proposal 後の candidate value は `[0, 1]` に clip する。
- 入力 guide も `[0, 1]` の grayscale value として扱う。

## 6. Decoder Parameters Used in Freeze Benchmark

Model C freeze benchmark では、次の設定を使った。

```text
model_c:
  j_base: 2.0
  lambda_data: 5.0
  gamma: 40.0
anneal:
  sweeps: 80
  temp_start: 0.5
  temp_end: 0.01
  proposal_sigma: 0.12
input_size: 32x32
static_noise_sigma: 0.03
```

参照:

- `results/2026-05-16-issue-5-model-c-freeze-benchmark/notes.md`
- `results/2026-05-16-issue-5-model-c-freeze-benchmark/summary_metrics.json`

## 7. Verified Results

この節は仕様上の定義ではなく、v0.2.1 draft の根拠になった測定結果である。

Model C freeze benchmark は、cross、diagonal、circle、thin line、soft gradient の synthetic guides で保存形式つきに実行された。各 shape には `config.json`、`metrics.json`、`notes.md`、主要 PNG が保存されている。

| Shape | Model C MAD | Background mean | Edge leakage | Decode time seconds |
| --- | ---: | ---: | ---: | ---: |
| cross | 0.010742 | 0.006267 | 0.006840 | 0.612222 |
| diagonal | 0.007401 | 0.005397 | 0.007102 | 0.662335 |
| circle | 0.009306 | 0.005878 | 0.005129 | 0.627884 |
| thin_line | 0.006110 | 0.005553 | 0.007054 | 0.656073 |
| soft_gradient | 0.021839 | 0.240061 | N/A | 0.701323 |

cross baseline の暫定基準:

```text
Background Mean <= 0.02
Edge Leakage    <= 0.02
MAD             <= 0.03
```

freeze benchmark の cross は、この暫定基準を満たした。

## 8. Interpretation

Model C は、synthetic grayscale guide に対して data fidelity と edge-aware interaction により背景漏れを抑える候補モデルとして有望である。

ただし、この解釈は次の範囲に限る。

- 同解像度 grayscale guide。
- 32x32 synthetic shapes。
- Python / NumPy の現行実装。
- 同一環境での再実行を前提にした再現性。

この結果は、実用圧縮性能、自然画像での一般性能、または環境非依存の正式仕様を示すものではない。

## 9. Probabilistic Interpretation

Model C energy は、MRF / Gibbs 型の画像復元と概念的に対応する `data fidelity term` と `edge-aware pairwise interaction term` を持つ。

ただし、この draft では観測モデル、prior、posterior distribution を formal には定義していない。そのため、Model C energy は確率モデルそのものではなく、seed つき緩和 decoder が低減しようとする deterministic decoder objective として扱う。

詳細な整理は `docs/model-c-energy-position.md` を参照する。

## 10. Limitations

- cross 以外の shape について、合格基準はまだ定義していない。
- soft gradient は edge leakage で評価しにくく、階調の自然さを別指標として扱う必要がある。
- edge width は Model C freeze benchmark では未計算である。
- 実画像 patch や自然画像 Ground Truth では未確認である。
- Python implementation は同一環境での再現性を対象としており、NumPy version や floating-point behavior をまたぐ bit-perfect guarantee はない。
- decode time は小画像では記録済みだが、大画像での品質と速度の trade-off は未確定である。

## 11. Rust Porting Open Items

Rust core decoder へ進む前に、少なくとも次を固定する必要がある。

- PRNG algorithm
- seed expansion
- initial state generation
- per-pixel update order
- proposal distribution
- temperature schedule
- accept/reject calculation
- floating-point or fixed-point arithmetic
- rounding and clipping rules
- boundary handling
- JSON or binary storage representation
- image normalization and quantization

関連 Issue:

- [#16 Rust移植に向けた deterministic PRNG と bit-perfect 再現性を調査する](https://github.com/nana-nun/sidf-lab/issues/16)

## 12. Related Work and Follow-up

Model C の位置づけ:

- `docs/model-c-energy-position.md`
- `references/notes/geman-geman-stochastic-relaxation.md`
- `references/notes/perona-malik-anisotropic-diffusion.md`

関連 Issue:

- [#5 Model C freeze benchmarkを作る](https://github.com/nana-nun/sidf-lab/issues/5)
- [#16 Rust移植に向けた deterministic PRNG と bit-perfect 再現性を調査する](https://github.com/nana-nun/sidf-lab/issues/16)
- [#39 soft gradient評価方針をfreeze criteriaに昇格する](https://github.com/nana-nun/sidf-lab/issues/39)
- [#40 Model CとPerona-Malik型diffusionの差を比較する](https://github.com/nana-nun/sidf-lab/issues/40)

## 13. Draft-to-Spec Criteria

この draft を正式な仕様候補に近づけるには、少なくとも次を満たす必要がある。

1. Rust 移植前の PRNG、固定小数点、丸め、更新順序を定義する。
2. 同じ seed と同じ serialized input から実装非依存に同じ output を得る条件を定義する。
3. soft gradient 用の評価方針を freeze criteria に入れる。
4. Model C と既存の edge-preserving smoothing / diffusion baseline を同条件で比較する。
5. storage format、versioning、metadata、checksum を draft として分離定義する。
6. 実験結果と仕様上の決定を引き続き分離して管理する。

## 14. References

- `docs/research-state.md`
- `docs/repository-architecture.md`
- `docs/model-c-energy-position.md`
- `results/2026-05-16-issue-4-model-c-cross-baseline/notes.md`
- `results/2026-05-16-issue-5-model-c-freeze-benchmark/notes.md`
- `results/2026-05-16-issue-5-model-c-freeze-benchmark/summary_metrics.json`
