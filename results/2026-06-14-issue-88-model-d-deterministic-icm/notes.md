# Model D Deterministic ICM Evaluation

## Question

Model Dのquadratic objectiveで、Gaussian proposal greedyの探索不足とobjective自体のreference品質上の限界を分けられるか。

## Hypothesis

解析的な局所最小値へ更新するdeterministic ICMはgreedy fixedよりobjectiveを大きく低下させる。一方、objective低下がMAD、PSNR、SSIM、gradient magnitude MADの改善と一致するとは限らない。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe experiments/exp_020_model_d_deterministic_icm.py`
- Date: 2026-06-14
- Experiment seed: 20260614
- Cross decoder seed: 8800
- Natural patch decoder seed: 8801
- Initial state: bilinear upscaled guide
- Confidence: uniform 1.0
- Texture: 0.0
- Pairwise: quadratic
- Model params: `{'j_base': 1.8, 'lambda_data': 6.0, 'gamma': 35.0}`
- Cross sweeps: 35
- Natural patch sweeps: 18
- Python / dependency version: Python 3.12.13, NumPy 2.3.5

### Analytic Coordinate Update

1画素 `v_i` に関係する局所objectiveは次である。

```text
lambda_data * c_i * (v_i - s_i)^2
+ sum_j J_ij * (v_i - v_j)^2
```

微分を0とした局所最小値は次になる。

```text
v_i* = (lambda_data * c_i * s_i + sum_j J_ij * v_j)
       / (lambda_data * c_i + sum_j J_ij)
```

`J_ij = j_base * exp(-gamma * (s_i - s_j)^2)` とし、fixed row-majorで直前までの更新値を使う。出力領域は `[0, 1]` なので更新値をclampする。今回の非負重みとguide/state範囲では加重平均自体が通常 `[0, 1]` に入るが、境界条件を明示するためclampを残した。分母が0の場合は現在値を保持する。

## Baseline

画像baselineはnearest、bilinear、bicubic。decoder比較はIssue #87相当の`greedy_fixed`と`deterministic_icm`で、同じbilinear初期状態、quadratic objective、sweep上限を使った。

## Metrics

### Cross

| Output | MAD | PSNR | Global SSIM | Gradient magnitude MAD | Final objective | Objective decrease | Updates | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nearest | 0.013794 | 21.613 | 0.915546 | 0.013946 | N/A | N/A | N/A | 0.000194 |
| bilinear | 0.033143 | 21.495 | 0.901668 | 0.030511 | N/A | N/A | N/A | 0.000261 |
| bicubic | 0.035119 | 21.585 | 0.904375 | 0.030838 | N/A | N/A | N/A | 0.106853 |
| greedy_fixed | 0.034656 | 21.455 | 0.900036 | 0.031011 | 13.096785 | 0.540559 | 1306 | 1.168005 |
| deterministic_icm | 0.035434 | 21.426 | 0.899088 | 0.031174 | 12.941375 | 0.695969 | 61652 | 0.527841 |

### Natural Patch

| Output | MAD | PSNR | Global SSIM | Gradient magnitude MAD | Final objective | Objective decrease | Updates | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nearest | 0.045384 | 22.578 | 0.953421 | 0.033359 | N/A | N/A | N/A | 0.000149 |
| bilinear | 0.044369 | 23.276 | 0.959480 | 0.030565 | N/A | N/A | N/A | 0.000343 |
| bicubic | 0.042397 | 23.578 | 0.962820 | 0.029326 | N/A | N/A | N/A | 0.429671 |
| greedy_fixed | 0.044619 | 23.228 | 0.958971 | 0.030756 | 22.079004 | 0.699860 | 2956 | 2.529227 |
| deterministic_icm | 0.045106 | 23.173 | 0.958354 | 0.031079 | 21.243696 | 1.535168 | 293753 | 1.569501 |

## Saved Artifacts

- Config: `config.json`
- Metrics and diagnostics: `metrics.json`
- Notes: `notes.md`
- Cross artifacts: `cross/`
- Natural patch artifacts: `natural_patch/`

## Images

![Cross deterministic ICM comparison](cross/comparison.png)

![Natural patch deterministic ICM comparison](natural_patch/comparison.png)

## Result

ICMはcrossでobjectiveを `13.637344` から `12.941375`、natural patchで `22.778864` から `21.243696` へ低下させた。greedy fixedの最終objectiveはcross `13.096785`、natural patch `22.079004` だった。

ICMのMADはcross `0.035434`、natural patch `0.045106` だった。

## Interpretation

ICMはgreedy fixedより低いobjectiveへ到達したため、Gaussian proposal greedyにはquadratic objectiveを十分に下げきらない探索不足があった。

一方、ICMはgreedy fixedよりobjectiveを強く低下させながら、cross / natural patchのMAD、PSNR、SSIM、gradient magnitude MADを改善しなかった。crossとnatural patchの両方でbilinearよりMADが悪く、natural patchではbicubicも上回らなかった。この結果は、proposal改善だけでは現行quadratic objectiveをreference品質の改善要因にできないというnegative evidenceである。objective最小化とGround Truth差分最小化は別の評価軸として扱う。

## Limitations

- crossと1枚のpublic-domain自然画像patchだけの比較である。
- fixed row-majorのGauss-Seidel型更新だけを評価し、Jacobi更新や線形方程式の直接解法とは比較していない。
- ICMの収束判定は最大画素変化 `1e-12` 以下であり、cross-environmentのbit-perfect再現性は未確認である。
- Global SSIMはwindowed SSIMではない。
- この結果はsuper-resolutionやcompressionの成立を示さない。

## Next

- Issue #92 で、有限温度Metropolisと現行quadratic objectiveを標準decoder候補として採用しない判断、およびdecoder procedure / objective designの未確定範囲をSIDF v0.3 draftへ反映する。
