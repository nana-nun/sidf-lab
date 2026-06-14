# Model D Acceptance / Update Order Isolation

## Question

Model Dのreference差分増加は、有限温度Metropolis acceptanceによるuphill moveと、pixel更新順序のどちらに強く関係するか。

## Hypothesis

greedy acceptanceはuphill moveを除くため、bilinear初期状態からの確率的driftを抑え、現行stochastic条件よりreference差分を減らす可能性がある。fixed orderの影響はacceptance modeより小さいと予想する。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe experiments/exp_019_model_d_update_procedure.py`
- Date: 2026-06-14
- Experiment seed: 20260614
- Cross decoder seed: 8700
- Natural patch decoder seed: 8701
- Initial state: bilinear upscaled guide
- Confidence: uniform 1.0
- Texture: 0.0
- Pairwise: current quadratic interaction
- Conditions: `config.json` の `conditions`
- Model params: `{'j_base': 1.8, 'lambda_data': 6.0, 'gamma': 35.0, 'pairwise_cap': 0.08}`
- Python / dependency version: Python 3.12.13, NumPy 2.3.5

## Baseline

nearest、bilinear、bicubicを画像baselineとした。更新手順の対照は `stochastic_random` を現行相当とし、acceptanceだけをgreedyへ、更新順序だけをfixed row-majorへ切り替えた。

## Metrics

### Cross

| Output | MAD | PSNR | Global SSIM | Gradient magnitude MAD | Accept rate | Uphill rate | Final objective | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nearest | 0.013794 | 21.613 | 0.915546 | 0.013946 | N/A | N/A | N/A | 0.000174 |
| bilinear | 0.033143 | 21.495 | 0.901668 | 0.030511 | N/A | N/A | N/A | 0.000179 |
| bicubic | 0.035119 | 21.585 | 0.904375 | 0.030838 | N/A | N/A | N/A | 0.100430 |
| stochastic_random | 0.040447 | 21.178 | 0.890932 | 0.037808 | 0.682792 | 0.260219 | 27.776579 | 1.370695 |
| stochastic_fixed | 0.040212 | 21.235 | 0.891964 | 0.037880 | 0.681355 | 0.258943 | 27.811127 | 1.107381 |
| greedy_random | 0.034596 | 21.457 | 0.900102 | 0.030976 | 0.009368 | 0.000000 | 13.117264 | 1.032060 |
| greedy_fixed | 0.034611 | 21.455 | 0.900066 | 0.030997 | 0.009473 | 0.000000 | 13.105768 | 1.040300 |

### Natural Patch

| Output | MAD | PSNR | Global SSIM | Gradient magnitude MAD | Accept rate | Uphill rate | Final objective | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nearest | 0.045384 | 22.578 | 0.953421 | 0.033359 | N/A | N/A | N/A | 0.000197 |
| bilinear | 0.044369 | 23.276 | 0.959480 | 0.030565 | N/A | N/A | N/A | 0.000532 |
| bicubic | 0.042397 | 23.578 | 0.962820 | 0.029326 | N/A | N/A | N/A | 0.384258 |
| stochastic_random | 0.053094 | 22.406 | 0.950809 | 0.032389 | 0.567369 | 0.280375 | 185.871304 | 2.336570 |
| stochastic_fixed | 0.052589 | 22.506 | 0.951800 | 0.032166 | 0.567966 | 0.280697 | 182.190047 | 2.303896 |
| greedy_random | 0.044632 | 23.230 | 0.958994 | 0.030758 | 0.010457 | 0.000000 | 22.048265 | 2.232147 |
| greedy_fixed | 0.044626 | 23.233 | 0.959027 | 0.030742 | 0.009966 | 0.000000 | 22.095806 | 2.174582 |

## Saved Artifacts

- Config: `config.json`
- Metrics and diagnostics: `metrics.json`
- Notes: `notes.md`
- Cross artifacts: `cross/`
- Natural patch artifacts: `natural_patch/`
- 各caseにbaseline、各更新条件、reference/bilinear差分、`comparison.png`を保存した。

## Images

![Cross update procedure comparison](cross/comparison.png)

![Natural patch update procedure comparison](natural_patch/comparison.png)

## Result

更新条件内の最小MADは、crossでは `greedy_random` の `0.034596`、natural patchでは `greedy_fixed` の `0.044626` だった。

## Interpretation

stochastic条件はcrossでproposalの `0.260`、natural patchで `0.280` をuphill moveとして受理した。最終objectiveはcrossで初期 `13.637` から `27.777`、natural patchで初期 `22.779` から `185.871` へ増加した。

greedy条件はuphill moveを受理せず、最終objectiveをcross `13.117`、natural patch `22.048` まで低下させた。MADもstochastic条件から大きく改善し、自然画像ではbilinearに近い値へ戻った。一方、random / fixed order間のMAD差はcross・natural patchとも小さく、今回の設定では更新順序よりacceptance modeの影響が大きかった。

ただしgreedy条件もcrossではbilinear MAD `0.033143`、natural patchではbilinear MAD `0.044369` とbicubic MAD `0.042397` を上回らなかった。objective低下とreference metrics改善は一致せず、現行quadratic objective自体がreference品質を改善するとは確認できない。

## Limitations

- crossと1枚のpublic-domain自然画像patchだけの比較である。
- greedy条件もGaussian proposalを使うため、連続値objectiveの厳密な座標最適化ではない。
- random order条件とfixed order条件ではRNG消費順が異なり、pixelごとのproposal列は完全には一致しない。
- Global SSIMは依存なしの全画像SSIMであり、windowed SSIMではない。
- この結果はsuper-resolutionやcompressionの成立を示さない。

## Next

- Issue #88 で、Gaussian proposalに依存するgreedy更新と、quadratic objectiveの解析的な局所最小値を使うdeterministic ICM / coordinate descentを比較する。
- 目的はproposal samplingの非効率とobjective自体の限界を分けることであり、単純なtemperature調整は広げない。
