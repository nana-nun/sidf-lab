# Model C と Perona-Malik 型 diffusion の複数shape比較

## Question

Issue #40 の vertical edge 比較で確認した Model C と Perona-Malik 型 diffusion の類似点と相違点は、diagonal、circle、soft gradient でも同じように確認できるか。

## Hypothesis

Model C と Perona-Malik 型 diffusion は、どちらも大きな局所差をまたぐ混合を弱める点で似ている。ただし Model C は guide-derived fixed weight と data fidelity を持つ stochastic relaxation であり、Perona-Malik 型 diffusion は state-derived conductance による deterministic diffusion なので、形状を増やしても同等の方法とは扱わない。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe experiments/exp_013_model_c_perona_malik_shapes.py`
- Date: 2026-05-24
- Experiment seed base: 20260524
- Decoder seed base: 6464
- Input size: 48x48
- Shapes: diagonal, circle, soft_gradient
- Model C params: `{'j_base': 2.0, 'lambda_data': 5.0, 'gamma': 40.0}`
- Model C anneal config: `{'sweeps': 60, 'temp_start': 0.45, 'temp_end': 0.01, 'proposal_sigma': 0.09}`
- Perona-Malik config: `{'steps': 60, 'dt': 0.18, 'kappa': 0.11}`
- Python / dependency version: Python 3.12.13, NumPy 2.3.5

## Baseline

各shapeの baseline は noisy initial guide をそのまま表示する `initial_noisy.png` とした。Perona-Malik 型 diffusion は、この noisy initial から画像勾配ベースの conductance で明示的に反復更新する比較対象である。

## Result

| Shape | Model C MAD | Perona-Malik MAD | Model C edge leakage | Perona-Malik edge leakage | Model C seconds | Perona-Malik seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| diagonal | 0.028039 | 0.002126 | 0.071019 | 0.079275 | 1.057500 | 0.005050 |
| circle | 0.028700 | 0.002569 | 0.071592 | 0.080081 | 1.033684 | 0.005068 |
| soft_gradient | 0.027864 | 0.005826 | N/A | N/A | 1.033581 | 0.005065 |

soft gradient は明確な foreground/background 境界を持たないため、edge leakage は `N/A` とし、各shapeの `notes.md` に列平均の逆行数、slope error、二階差分を保存した。

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Per-shape artifacts: `<shape>/config.json`, `<shape>/metrics.json`, `<shape>/notes.md`, `<shape>/*.png`

## Images

各shapeの `notes.md` に主要PNGへのMarkdown画像参照を保存した。

## Interpretation

diagonal、circle、soft gradient へ広げても、Model C は guide から固定的に pair weight を決め、Perona-Malik 型 diffusion は現在画像状態から conductance を決めるという違いを図とmetricsで確認できる。両者は edge-aware な混合抑制という役割では似るが、係数決定元、更新過程、data fidelity の有無が異なるため、同じ方法または一般的な優劣として扱わない。

## Limitations

- synthetic 3形状の小規模比較であり、一般画像品質を示さない。
- Perona-Malik 実装は最小の明示的diffusion loopであり、元論文の数値解析や安定性検討を網羅しない。
- Model C は seed、sweeps、temperature schedule に依存する。
- soft gradient の代替指標は階調の基本確認であり、知覚品質を保証しない。
- この結果は compression、super-resolution、Model C の一般的優位性を示すものではない。

## Next

- shape別の合格目安が必要なら、Model C freeze criteria 側で別Issueとして整理する。
- Perona-Malik 型以外の guided filtering / anisotropic smoothing baseline と比較する場合は、低解像度guide条件と高解像度guidance条件を分ける。
