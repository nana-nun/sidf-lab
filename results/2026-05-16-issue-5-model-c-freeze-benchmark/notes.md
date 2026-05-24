# Model C Freeze Benchmark Summary

## Question

Model CはRust移植前の基準実装候補として、cross以外の基本shapeでも保存形式つきで評価できるか。

## Hypothesis

crossでは既存の暫定目安を満たし、diagonal / circle / thin lineでも大きな背景漏れは抑えられる。soft gradientではエッジ指標ではなく階調の連続性を確認する必要がある。

## Setup

- Command: `$env:PYTHONPATH = "src"; python experiments/exp_004_shape_benchmark.py`
- Date: 2026-05-16
- Experiment seed: 20260516
- Decoder seed base: 4200
- Input size: 32x32
- Output size: 32x32
- Static noise sigma: 0.03
- Model: Model C

## Baseline

全shapeで、noiseを加えたstatic guideをそのまま表示する `baseline_direct.png` をbaselineとした。

## Metrics

| Shape | Model C MAD | Model C background mean | Model C edge leakage | Decode time seconds |
| --- | ---: | ---: | ---: | ---: |
| cross | 0.010742 | 0.006267 | 0.006840 | 0.612222 |
| diagonal | 0.007401 | 0.005397 | 0.007102 | 0.662335 |
| circle | 0.009306 | 0.005878 | 0.005129 | 0.627884 |
| thin_line | 0.006110 | 0.005553 | 0.007054 | 0.656073 |
| soft_gradient | 0.021839 | 0.240061 | N/A | 0.701323 |

## Saved Artifacts

- Summary metrics: `summary_metrics.json`
- Per-shape artifacts: `<shape>/config.json`, `<shape>/metrics.json`, `<shape>/notes.md`, `<shape>/*.png`

## Images

各shapeの `notes.md` に、Git管理される主要PNGへのMarkdown画像参照を保存した。

## Result

cross baseline criteria pass: `True`.

## Interpretation

このbenchmarkは、Model CをRust移植前の候補として評価するための保存形式を作った段階である。結果はshapeごとのsynthetic条件に限られ、一般画像品質や実用圧縮性能を示すものではない。

## Limitations

- cross以外の合格目安は未定義。
- soft gradientはedge leakageで評価しにくく、視覚的な階調確認を併記した。
- edge widthは今回未計算。
- Rust固定小数点やbit-perfect再現性はIssue #16で別途扱う。

## Next

- Issue #16 でRust移植前の再現性要件を整理する。
- cross以外のshapeに対する暫定合格目安を定義する。
