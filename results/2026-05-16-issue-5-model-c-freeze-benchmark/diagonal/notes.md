# Model C Freeze Benchmark: diagonal

## Question

Model Cは `diagonal` guideで、noisy static guide direct displayに対して安定した再構成候補を出せるか。

## Hypothesis

hard edgeを持つshapeでは、data fidelityとedge-aware interactionにより背景漏れを抑えられる。soft gradientでは、edge leakageより階調の連続性と段差の有無を重視する。

## Setup

- Command: `$env:PYTHONPATH = "src"; python experiments/exp_004_shape_benchmark.py`
- Date: 2026-05-16
- Experiment seed: 20260517
- Decoder seed: 4201
- Input guide: synthetic `diagonal` with deterministic Gaussian noise
- Input size: 32x32
- Output size: 32x32
- Model: Model C
- Model config: `{'j_base': 2.0, 'lambda_data': 5.0, 'gamma': 40.0}`
- Anneal config: `{'sweeps': 80, 'temp_start': 0.5, 'temp_end': 0.01, 'proposal_sigma': 0.12}`
- Python / dependency version: Python 3.12.13, NumPy 2.3.5

## Baseline

baselineは、noiseを加えたstatic guideをそのまま表示する `baseline_direct.png` とした。

## Metrics

| Metric | Baseline direct | Model C |
| --- | ---: | ---: |
| MAD vs clean guide | 0.013058 | 0.007401 |
| Foreground mean | 0.493495 | 0.488908 |
| Background mean | 0.012189 | 0.005397 |
| Edge leakage | 0.011097 | 0.007102 |
| Foreground variance | 0.000737 | 0.000981 |
| Background variance | 0.000319 | 0.000151 |
| Decode time seconds | 0.000000 | 0.662335 |

Mask note: foregroundはclean guideで非ゼロの画素、backgroundはそれ以外。

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Notes: `notes.md`
- Clean guide image: `guide_clean.png`
- Static guide image: `static_guide.png`
- Baseline image: `baseline_direct.png`
- Rendered image: `rendered_model_c.png`
- Difference image: `diff_model_c_vs_baseline.png`
- Comparison image: `comparison.png`

## Images

### Comparison

![Comparison of clean guide, static guide, baseline direct, Model C, and absolute difference](comparison.png)

### Clean Guide

![Clean synthetic guide](guide_clean.png)

### Static Guide

![Noisy static guide](static_guide.png)

### Baseline Direct

![Baseline direct rendering of the static guide](baseline_direct.png)

### Model C Rendered

![Model C rendered output](rendered_model_c.png)

### Difference

![Absolute difference between Model C and baseline direct](diff_model_c_vs_baseline.png)

## Result

このshapeのmetricsとPNG成果物を保存した。crossについてはfreeze criteriaの暫定目安と比較する。

## Interpretation

hard edge shapeとして、背景平均とedge leakageを中心に見る。

## Limitations

- synthetic guideのみで、実画像パッチでは未確認。
- decode timeは環境依存。
- edge widthは今回の小さい32x32 synthetic guidesでは安定した定義を置けなかったため未計算。
- Rust固定小数点や環境非依存のbit-perfect再現性は未確認。関連Issue: #16。

## Next

- cross以外の合格目安を別途定義する。
- Rust移植前にPRNG、丸め、固定小数点の再現性要件をIssue #16で整理する。
