# Model D/E 採否判断マップ

Status: Draft decision map
Date: 2026-07-27
Related Issues: [#106](https://github.com/nana-nun/sidf-lab/issues/106), [#113](https://github.com/nana-nun/sidf-lab/issues/113), [#118](https://github.com/nana-nun/sidf-lab/issues/118), [#125](https://github.com/nana-nun/sidf-lab/issues/125), [#127](https://github.com/nana-nun/sidf-lab/issues/127), [#138](https://github.com/nana-nun/sidf-lab/issues/138)

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
| Model E | trainable / source-split の single-state / coupled-state 候補 | 採用しない現行候補 | Issue #104 | source分割評価で最良classical候補を上回らなかった。#104条件の現行候補を採用する根拠はない。 |
| Model E | compact parameterization redesign Candidate A/B/C | 採用しない現行候補、ただし一部改善あり | Issue #122 | Candidate A/Cは現行Model Eより少し良かったが、best classical INR、nearest、bicubicを上回らなかった。採用根拠には不足。 |
| Model E | gated interaction風coupling | 採用しない現行候補、ただし現行coupledを小さく改善 | Issue #118 | evaluation splitで現行coupledよりMADを改善したが、best classical INRに届かず、平均540 bitsのcoupling overheadを要した。 |
| Model E | trainable small MLP / RFF / SIREN / Fourier baseline | 比較基準として継続 | Issues #103, #104 | `mlp_small` が#104のevaluation splitで最良parameterized候補だった。Model E再設計時の比較対象として残す。 |

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

Issue #104 では、Issue #103 の最小trainable INR基盤と Issue #108 のsource分割fixtureを使い、trainable Fourier / RFF / SIREN / MLP baseline と trainable Model E single-state / coupled-state を比較した。結果は `results/2026-06-28-issue-104-trainable-inr-source-split/` に保存した。

Evaluation splitでは、best parameterized candidate は `mlp_small` で、mean serialized side bits `708`、mean quantized MAD `0.089009` だった。Best Model E candidate は `model_e_single` で、mean serialized side bits `576`、mean quantized MAD `0.090061` だった。

| Candidate | Family | Mean serialized side bits | Evaluation mean quantized MAD |
| --- | --- | ---: | ---: |
| bicubic | image baseline | n/a | 0.086314 |
| mlp_small | classical INR | 708 | 0.089009 |
| model_e_single | Model E | 576 | 0.090061 |
| model_e_coupled | Model E | 780 | 0.090095 |

Issue #122 では、Issue #116 で整理し Issue #121 で実装した Candidate A fixed ladder、Candidate B compact frequency table、Candidate C coordinate frequency + guide modulationを、#104と同じsource-split fixture、12-bit量子化、incremental side-bit accountingで比較した。結果は `results/2026-06-29-issue-122-model-e-parameterization-redesign/` に保存した。

Evaluation splitでは、best classical INR は `fourier_mid` で、mean serialized side bits `348`、mean quantized MAD `0.089244` だった。Best current Model E は `model_e_coupled` で、mean serialized side bits `780`、mean quantized MAD `0.090095` だった。Best new candidate は `candidate_a_ladder` で、mean serialized side bits `636`、mean quantized MAD `0.089839` だった。

| Candidate | Family | Mean serialized side bits | Evaluation mean quantized MAD |
| --- | --- | ---: | ---: |
| bicubic | image baseline | n/a | 0.086314 |
| nearest | image baseline | n/a | 0.089404 |
| fourier_mid | classical INR | 348 | 0.089244 |
| model_e_coupled | current Model E | 780 | 0.090095 |
| candidate_a_ladder | new Model E candidate | 636 | 0.089839 |
| candidate_c_modulated | new Model E candidate | 636 | 0.089848 |

Issue #118 では、#104と同じsource-split fixture、48-stepのdependency-free random-search fit、12-bit量子化、incremental side-bit accountingで、現行coupled-stateと2種類のcoupling variationを比較した。結果は `results/2026-07-27-issue-118-model-e-coupling-variants/` に保存した。

Evaluation splitでは、best classical INRは `mlp_small` で mean quantized MAD `0.089009`、最良Model E系候補は `model_e_gated_coupled` で `0.089183` だった。現行 `model_e_coupled_current` は `0.090095`、controlled-rotation風候補は `0.090181`、gated interaction風候補は `0.089183` だった。新coupling候補は現行coupled比で平均 `540` bits のcoupling overheadを要した。

| Candidate | Family | Mean serialized side bits | Evaluation mean quantized MAD |
| --- | --- | ---: | ---: |
| mlp_small | classical INR | 708 | 0.089009 |
| model_e_coupled_current | current Model E | 780 | 0.090095 |
| model_e_controlled_rotation | coupling variation | 1320 | 0.090181 |
| model_e_gated_coupled | coupling variation | 1320 | 0.089183 |

### Interpretation

このfixed-feature条件では、Model E候補がclassical INR baselineを一貫して上回るとは解釈しない。量子回路由来の構造そのものを採用理由にせず、同じserialized side bitsでclassical baselineを上回る測定結果が必要である。

#104 のtrainable / source-split条件でも、現行のsingle-state / coupled-state候補を採用する根拠は得られていない。ただし、これは #104 のparameterization、optimizer、source分割fixture、incremental side-bit見積もりに対する負の結果であり、Model E一般や別の量子回路由来座標関数の否定ではない。

#122 のcompact parameterization redesignでは、Candidate A/C が現行Model Eより少し改善した。しかし best classical INR、nearest、bicubic baselineを上回らなかったため、この設定のCandidate A/B/Cを採用候補へ戻す根拠は不足している。これはcandidate size、bit-depth耐性、autograd optimizer、別coupling設計の否定ではない。

#118 のgated interaction風couplingは現行coupled-stateを小さく改善したが、best classical INRには届かず、追加side bitsも増えた。したがって、今回の2案、source-split fixture、random-search fit条件では、coupling variationを採用候補へ戻す根拠として不足している。これは別のcoupling式やautograd optimizer条件を含むModel E一般の否定ではない。

### Current Decision

- #98 の fixed feature dictionary + linear readout 最小Model E候補は、SIDF draft仕様へ採用しない。
- #104 の trainable / source-split Model E single-state / coupled-state候補も、SIDF draft仕様へ採用しない。
- #122 のcompact parameterization redesign Candidate A/B/Cも、現時点ではSIDF draft仕様へ採用しない。
- #118 のcontrolled-rotation風 / gated interaction風coupling候補も、現時点ではSIDF draft仕様へ採用しない。
- #98、#104、#117、#118、#122 はModel E全体の否定ではなく、それぞれ評価した構造とprotocolの結果として扱う。
- #117 のoptimizer診断も、現行Model E候補を採用候補へ戻す根拠にはならなかった。
- #125 では autograd optimizer依存をdefault requirementsへ追加しない。続ける場合はPyTorch CPU optional backendの小さなspikeに分ける。
- Model Eを継続する場合は、candidate size、bit-depth耐性、optimizer依存、coupling設計を分ける。継続しない場合は、Model E系列を一時保留として扱い、残Issueを再優先度付けする。

### Continue / Pause Decision

Model Eを継続する価値があるのは、次の切り分けを小さく検証する場合に限る。

- #119: 量子化bit-depth比較の結果は別途保存済みであり、gated couplingの改善や540-bit overheadとは別軸である。判断文書への反映は [#136](https://github.com/nana-nun/sidf-lab/issues/136) で扱う。
- #125: default dependencyとしては導入せず、続ける場合はPyTorch CPU optional backendのspikeでModel Eとclassical INRへ同じoptimizer条件を適用できるかを測る。
- #118: gated interaction風候補は現行coupledを小さく改善したが、best classical INRを上回らず、平均540 bitsのoverheadを要した。この条件ではcoupling設計だけを理由に継続しない。
- #134: PyTorch導入済み環境で、Model Eとclassical INRに同じoptional autograd optimizerを適用した実測は未完了である。

一時保留の根拠は、#98 / #104 / #117 / #118 / #122 の全てで評価済みModel E候補が採用基準に届いていないことである。#122のCandidate A/Cと#118のgated interaction風候補には現行候補に対する小さな改善があったが、いずれもbest classical INRを上回らなかった。このため、Model Eを継続する場合でも「採用へ進む」ではなく、#134のoptimizer条件など未測定要因を切り分ける扱いにする。

### Limitations

- small SIREN baselineは通常のtrainable multi-layer SIRENではない。
- #104 のtrainable比較は小規模source-split fixtureと最小optimizerに基づく。
- #118 のcoupling比較は2案、64x64 crop各split 2件、48-step random-search fitに限られる。
- serialized bitsはparameter side informationの簡易見積もりであり、guide bits、entropy coding、container overheadを含まない。
- extrapolated outputはartifact診断であり、外挿解像度のGround Truth品質測定ではない。
- quantum advantage、compression、super-resolutionは未測定である。

## Unmeasured Claims

現時点で主張しないこと:

- SIDF が PNG / JPEG / AVIF / JPEG XL / neural codec より高圧縮である。
- Ground Truth比較なしに超解像性能がある。
- Model D の texture prior が意味的ディテールを生成する。
- Model E が量子優位を持つ。
- #98 / #104 のparameter side bitsだけで実用圧縮性能を評価できる。
- Python/NumPy結果が環境非依存のbit-perfect decoder仕様である。

## Related Issues

| Issue | Type | Status in this map | Purpose |
| --- | --- | --- | --- |
| [#103](https://github.com/nana-nun/sidf-lab/issues/103) | `t:impl` | reflected | Model E / INR の全parameter fitting基盤を追加した。 |
| [#104](https://github.com/nana-nun/sidf-lab/issues/104) | `t:exp` | reflected | trainable INR baselineとsource分割datasetでModel Eを再比較した。 |
| [#107](https://github.com/nana-nun/sidf-lab/issues/107) | `t:ref` | referenced | INR圧縮のbit accountingとparameter量子化方針を整理した。 |
| [#108](https://github.com/nana-nun/sidf-lab/issues/108) | `t:impl` | reflected | source分割済みgrayscale patch fixtureを追加した。 |
| [#113](https://github.com/nana-nun/sidf-lab/issues/113) | `t:docs` | current update | #104 の負の結果を採否判断へ反映する。 |
| [#117](https://github.com/nana-nun/sidf-lab/issues/117) | `t:exp` | reflected | 現行Model E fittingのoptimizer / initializationを診断した。 |
| [#118](https://github.com/nana-nun/sidf-lab/issues/118) | `t:exp` | reflected | coupling variationを比較し、gated interaction風候補の小さな改善とbit overheadを記録した。 |
| [#119](https://github.com/nana-nun/sidf-lab/issues/119) | `t:exp` | result saved | INR parameter量子化bit深度耐性を比較した。判断文書への反映は #136 で扱う。 |
| [#122](https://github.com/nana-nun/sidf-lab/issues/122) | `t:exp` | reflected | Candidate A/B/C parameterization redesignをsource-splitで比較した。 |
| [#125](https://github.com/nana-nun/sidf-lab/issues/125) | `t:impl` | current update | autograd optimizer基盤はdefault依存に追加せず、PyTorch CPU optional backendのspikeに分ける判断を文書化する。 |
| [#127](https://github.com/nana-nun/sidf-lab/issues/127) | `t:docs` | current update | Model E系列の継続/保留判断を整理する。 |
| [#134](https://github.com/nana-nun/sidf-lab/issues/134) | `t:exp` | unmeasured | PyTorch導入済み環境でoptional autograd optimizerを実測する。 |
| [#138](https://github.com/nana-nun/sidf-lab/issues/138) | `t:docs` | current update | #118のcoupling variation結果を採否判断へ反映する。 |

## References

- `docs/research-state.md`
- `docs/research-plan.md`
- `docs/model-e-research-design.md`
- `specs/sidf-v0.3.0-draft.md`
- `results/2026-06-14-issue-87-model-d-update-procedure/notes.md`
- `results/2026-06-14-issue-88-model-d-deterministic-icm/notes.md`
- `results/2026-06-27-issue-98-model-e-bit-budget/notes.md`
- `results/2026-06-28-issue-104-trainable-inr-source-split/notes.md`
- `results/2026-06-29-issue-122-model-e-parameterization-redesign/notes.md`
- `results/2026-07-27-issue-118-model-e-coupling-variants/notes.md`
- `docs/model-e-autograd-optimizer-decision.md`
