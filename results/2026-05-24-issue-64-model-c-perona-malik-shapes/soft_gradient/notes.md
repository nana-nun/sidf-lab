# Model C と Perona-Malik 型 diffusion の形状別比較: soft_gradient

## Question

Model C の guide-derived fixed weight と Perona-Malik 型 diffusion の state-derived conductance の違いは、`soft_gradient` でも確認できるか。

## Hypothesis

hard edge shape では、両者とも大きな局所差をまたぐ混合を弱めるが、Model C は noisy guide から固定weightを作り、Perona-Malik 型 diffusion は現在状態から conductance を更新するため、weight / conductance map の読み方は一致しない。soft gradient では明確な境界がないため、edge leakage ではなく階調の単調性と滑らかさで確認する。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe experiments/exp_013_model_c_perona_malik_shapes.py`
- Date: 2026-05-24
- Experiment seed: 20260526
- Decoder seed: 6466
- Input: 48x48 synthetic `soft_gradient` with deterministic Gaussian noise
- Model C params: `{'j_base': 2.0, 'lambda_data': 5.0, 'gamma': 40.0}`
- Model C anneal config: `{'sweeps': 60, 'temp_start': 0.45, 'temp_end': 0.01, 'proposal_sigma': 0.09}`
- Perona-Malik config: `{'steps': 60, 'dt': 0.18, 'kappa': 0.11}`
- Python / dependency version: Python 3.12.13, NumPy 2.3.5

## Baseline

Baseline は noisy initial guide をそのまま表示する `initial_noisy.png` とした。Perona-Malik 型 diffusion は、この noisy initial から画像勾配ベースの conductance で明示的に反復更新する比較対象である。

## Metrics

| Output | MAD vs clean | PSNR vs clean | Global SSIM vs clean | Gradient MAD vs clean | Edge leakage | Decode time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial_noisy | 0.035791 | 26.929 | 0.962428 | 0.066921 | N/A | N/A |
| model_c | 0.027864 | 29.067 | 0.976558 | 0.043009 | N/A | 1.033581 |
| perona_malik | 0.005826 | 39.651 | 0.997818 | 0.001723 | N/A | 0.005065 |

## Soft Gradient Alternative Metrics

soft gradient は明確な foreground/background 境界を持たないため、edge leakage は不適用とした。代替として、列平均の逆行数、clean gradient に対する slope error、二階差分の大きさを保存した。

| Output | Backward steps | Slope MAE vs clean | Mean abs second diff | Max abs second diff |
| --- | ---: | ---: | ---: | ---: |
| initial_noisy | 2 | 0.006708 | 0.011713 | 0.051513 |
| model_c | 0 | 0.005182 | 0.008241 | 0.027860 |
| perona_malik | 0 | 0.001605 | 0.000440 | 0.001941 |

## Pair Weight / Conductance Summary

hard edge shape では `boundary_pair_mean` を true mask をまたぐ4近傍pair、`flat_pair_mean` をそれ以外のpairとして計算した。soft gradient では明確な境界がないため、boundary 系は `N/A` とし、全pair平均だけを保存した。

| Map | Flat pair mean | Boundary pair mean | Boundary / flat |
| --- | ---: | ---: | ---: |
| model_c_guide_weight | 1.733735 | N/A | N/A |
| pm_initial_conductance | 0.770916 | N/A | N/A |
| pm_final_conductance | 0.995577 | N/A | N/A |

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Clean guide: `clean_guide.png`
- Initial noisy guide: `initial_noisy.png`
- Model C output: `model_c.png`
- Perona-Malik output: `perona_malik.png`
- Model C guide weight map: `model_c_weight_map.png`
- Perona-Malik initial conductance map: `pm_conductance_initial_map.png`
- Perona-Malik final conductance map: `pm_conductance_final_map.png`
- Difference maps: `diff_model_c_vs_clean.png`, `diff_pm_vs_clean.png`
- Comparison strip: `comparison.png`

## Images

![Comparison of clean, noisy initial, Model C, Perona-Malik, and difference maps](comparison.png)

![Model C guide-derived pair weights](model_c_weight_map.png)

![Perona-Malik initial conductance](pm_conductance_initial_map.png)

![Perona-Malik final conductance](pm_conductance_final_map.png)

## Result

`soft_gradient` について、Model C と Perona-Malik 型 diffusion の出力、差分、guide-derived weight、state-derived conductance を保存した。

## Interpretation

この形状でも、Model C のweightは入力guideから固定的に決まり、Perona-Malik 型のconductanceは初期状態とdiffusion後で変化するものとして確認できる。metricsの良し悪しは、この小規模synthetic条件での比較結果として扱い、一般的な優位性とは解釈しない。

## Limitations

- synthetic `soft_gradient` 1条件の小規模比較であり、一般画像品質を示さない。
- Perona-Malik 実装は最小の明示的diffusion loopであり、元論文の数値解析や安定性検討を網羅しない。
- Model C は stochastic relaxation なので、seed、sweeps、temperature schedule に依存する。
- この結果は compression、super-resolution、Model C の一般的優位性を示すものではない。

## Next

- Perona-Malik 型以外の guided filtering / anisotropic smoothing baseline と比較する場合は、低解像度guide条件と高解像度guidance条件を分ける。
- Model C freeze criteria として使う場合は、shape別の合格目安を別途定義する。
