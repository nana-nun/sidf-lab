# Model D/E 採否判断マップ

Status: Draft decision map
Date: 2026-06-27
Related Issue: [#106](https://github.com/nana-nun/sidf-lab/issues/106)

この文書は、Model D / Model E の保存済み実験から「現時点で採用しない候補」「再設計候補」「未評価候補」を追跡するための人間向けメモである。

SIDF は現段階では実用圧縮形式ではない。この文書は、負の結果を仕様へ混ぜず、次に何を検証すべきかを見失わないための整理である。

## Scope

対象:

- Model D の confidence-aware multi-resolution reconstruction 系列
- Model E の quantum-inspired implicit residual representation 系列
- 保存済み実験に基づく採用保留、不採用、再設計、未評価の区別

対象外:

- SIDF binary layout の確定
- 実用codecとの圧縮率比較
- RGB / 動画 / 大規模dataset評価
- 量子優位、一般的super-resolution、既存画像形式より高圧縮という主張

## Decision Table

| Model | Candidate | Current handling | Evidence | Interpretation |
| --- | --- | --- | --- | --- |
| Model D | 有限温度 Metropolis acceptance を使う現行更新手順 | 採用しない現行候補 | Issue #87 | uphill moveが多く、objectiveとreference差分を増やした。標準decoder手順としては扱わない。 |
| Model D | 現行quadratic data / pairwise objectiveをICMで最小化する手順 | 採用しない現行候補 | Issue #88 | objectiveをさらに下げてもMAD、PSNR、SSIM、gradient metricsは改善せず、bilinear / bicubic baselineを上回らなかった。 |
| Model D | gradient-based confidence map、edge-aware pairwise、deterministic texture field | 再設計候補 | Issues #56, #61, #67, #75 | 現行組み合わせではbaseline改善なし。ただし要素そのものは分離して再設計する余地がある。 |
| Model D | white-noise texture term | 採用しない現行候補 | Issues #37, #63, #75 | 粒状変化は出るが、意味的ディテールやbaseline改善としては確認できていない。 |
| Model E | fixed feature dictionary + linear readout の single-state / coupled-state 最小候補 | 採用しない現行候補 | Issue #98 | 同一side-bit比較でRFF / bicubicを上回らなかった。最小候補をSIDF draft仕様へ採用する根拠はない。 |
| Model E | 全parameter optimized Model E | 未評価 | Issues #103, #104 | #98では未実施。全parameter fitting基盤とsource分割datasetで再比較する。 |
| Model E | trainable SIREN / MLP baselineとの比較 | 未評価 | Issues #103, #104 | #98のSIRENはfixed sine feature + linear readoutであり、通常のtrainable SIREN比較ではない。 |

## Model D

### Result

Issue #87 では、現行相当の有限温度 Metropolis 条件が proposal の約26〜28%を uphill move として受理し、objective と reference差分を増やした。greedy acceptance はstochastic条件より改善したが、bilinear / bicubic baselineは上回らなかった。

Issue #88 では、Gaussian proposalに依存しない deterministic ICM を使い、現行quadratic objectiveをgreedy fixedより低い値へ下げた。一方、reference metricsは改善せず、cross / natural patchの両方でbilinearよりMADが悪かった。

### Interpretation

Model D の現行結果から言えることは、現行objectiveと現行更新手順の組み合わせを標準decoderへ採用する根拠がない、という範囲に限る。confidence map、edge-aware interaction、stochastic decoder一般を否定する結果ではない。

### Current Decision

- 有限温度 Metropolis acceptance を使う現行更新手順は、標準decoder候補として採用しない。
- 現行quadratic objectiveは、solverをICMへ変えてもreference品質改善に結びつかなかったため、標準objective候補として採用しない。
- confidence / pairwise / texture の各要素は、現行設計のまま採用せず、仮定を変える場合にだけ再設計候補として扱う。

### Limitations

- 主な判断はcrossと1枚のpublic-domain自然画像patchを中心にした小規模runに基づく。
- 別objective、別confidence設計、別texture prior、別datasetでの結果は未評価である。
- negative resultは、super-resolutionやcompressionの不可能性を示すものではない。

## Model E

### Result

Issue #98 では、nearest / bilinear / bicubic、Fourier、RFF、fixed sine feature版SIREN、Model E single-state、Model E coupled-stateを比較した。全parameterized候補は fixed feature dictionary + ridge least-squares readout に制限し、12-bit量子化後のparameter side bitsを保存した。

Evaluation splitでは `rff_mid` がparameterized候補内の最小MAD `0.034915`、mean serialized side bits `1312` だった。bicubic baselineのevaluation mean MADは `0.036953` だった。Model E候補のevaluation mean MADは次の範囲だった。

| Candidate | Mean serialized side bits | Evaluation mean MAD |
| --- | ---: | ---: |
| model_e_single_low | 532 | 0.039588 |
| model_e_single_mid | 892 | 0.039439 |
| model_e_coupled_low | 736 | 0.040270 |
| model_e_coupled_mid | 1276 | 0.039951 |

### Interpretation

このfixed-feature条件では、Model E候補がclassical INR baselineを一貫して上回るとは解釈しない。量子回路由来の構造そのものを採用理由にせず、同じserialized side bitsでclassical baselineを上回る測定結果が必要である。

### Current Decision

- #98 の最小Model E候補は、SIDF draft仕様へ採用しない。
- #98 はModel E全体の否定ではなく、fixed feature dictionary + linear readout 条件の負の結果として扱う。
- Model Eを継続する場合は、全parameter fitting、angle / frequency parameterization、source image単位のheld-out評価を別Issueで扱う。

### Limitations

- small SIREN baselineは通常のtrainable multi-layer SIRENではない。
- serialized bitsはparameter side informationの簡易見積もりであり、guide bits、entropy coding、container overheadを含まない。
- extrapolated outputはartifact診断であり、外挿解像度のGround Truth品質測定ではない。
- quantum advantage、compression、super-resolutionは未測定である。

## Unmeasured Claims

現時点で主張しないこと:

- SIDF が PNG / JPEG / AVIF / JPEG XL / neural codec より高圧縮である。
- Ground Truth比較なしに超解像性能がある。
- Model D の texture prior が意味的ディテールを生成する。
- Model E が量子優位を持つ。
- #98 のparameter side bitsだけで実用圧縮性能を評価できる。
- Python/NumPy結果が環境非依存のbit-perfect decoder仕様である。

## Next Issues

| Issue | Type | Purpose |
| --- | --- | --- |
| [#103](https://github.com/nana-nun/sidf-lab/issues/103) | `t:impl` | Model E / INR の全parameter fitting基盤を追加する。 |
| [#104](https://github.com/nana-nun/sidf-lab/issues/104) | `t:exp` | trainable INR baselineとsource分割datasetでModel Eを再比較する。 |
| [#107](https://github.com/nana-nun/sidf-lab/issues/107) | `t:ref` | INR圧縮のbit accountingとparameter量子化方針を整理する。 |
| [#108](https://github.com/nana-nun/sidf-lab/issues/108) | `t:impl` | source分割済みgrayscale patch fixtureを追加する。 |

## References

- `docs/research-state.md`
- `docs/research-plan.md`
- `docs/model-e-research-design.md`
- `specs/sidf-v0.3.0-draft.md`
- `results/2026-06-14-issue-87-model-d-update-procedure/notes.md`
- `results/2026-06-14-issue-88-model-d-deterministic-icm/notes.md`
- `results/2026-06-27-issue-98-model-e-bit-budget/notes.md`
