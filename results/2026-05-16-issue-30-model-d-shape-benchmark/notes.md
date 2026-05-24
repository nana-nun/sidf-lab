# Model D Shape Benchmark

## Question

Model Dがcross以外の低解像度guideでも境界と階調を妥当に扱えるかを、保存形式つきbenchmarkとして確認する。

## Hypothesis

diagonal line、circle、thin lineではconfidence mapとedge-aware interactionが境界付近の崩れを抑える可能性がある。soft gradientでは、confidence mapが階調を硬く分断しないことを確認する必要がある。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe experiments/exp_005_model_d_shape_benchmark.py`
- Date: 2026-05-16
- Experiment seed: 20260516
- Decoder seed base: 5300
- Low guide size: 16x16
- Output size: 64x64
- Shapes: diagonal, circle, thin_line, soft_gradient
- Model: Model D candidate
- Model config: `{'j_base': 1.8, 'lambda_data': 6.0, 'gamma': 35.0, 'texture_weight': 0.35}`

## Baseline

baselineはnearest、bilinear、bicubic upscaling。metricsのreferenceは同じsynthetic shapeを64x64で生成した比較用参照であり、実画像のGround Truthではない。

## Result

| Shape | Model D MAD | Bilinear MAD | Model D edge leakage | Model D edge width | Decode time seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| diagonal | 0.044796 | 0.030566 | 0.073768 | 3.540984 | 1.833956 |
| circle | 0.035057 | 0.018426 | 0.257899 | 4.215909 | 1.833668 |
| thin_line | 0.043634 | 0.030878 | 0.275593 | 3.984375 | 2.012119 |
| soft_gradient | 0.019143 | 0.000000 | N/A | N/A | 1.979287 |

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Notes: `notes.md`
- Per-shape artifacts: `<shape>/low_guide.png`, `<shape>/nearest.png`, `<shape>/bilinear.png`, `<shape>/bicubic.png`, `<shape>/confidence.png`, `<shape>/rendered_model_d.png`, `<shape>/diff_model_d_vs_bilinear.png`, `<shape>/comparison.png`, `<shape>/config.json`, `<shape>/metrics.json`, `<shape>/notes.md`

## Interpretation

このbenchmarkはModel D候補をshape coverageの観点で記録するためのもの。結果はsynthetic guideに限定され、実用圧縮性能、一般画像品質、または超解像性能を断定するものではない。

## Limitations

- 実画像パッチでは未検証。
- white-noise texture termのため、質感は意味的ディテールではない。
- Python/NumPy実装の結果であり、Rust固定小数点やbit-perfect再現性は未確認。
- Issue #6 の比較指標とは矛盾しない形で保存したが、crossを含む統一比較は別Issueで扱う。

## Next

- Issue #6 でModel Dとbilinear/bicubicの統一比較指標を整理する。
- Issue #14 でguided filter / guided upsamplingとの比較観点を調査する。
