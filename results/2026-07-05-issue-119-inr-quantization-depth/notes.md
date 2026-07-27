# INR Parameter Quantization Depth Comparison

## Question

Model E single/coupled と classical INR baseline は、8-bit、12-bit、16-bit のparameter量子化でどちらが劣化しにくいか。

## Hypothesis

Model Eの回転角・状態更新に由来するparameterizationは、低bit量子化でも品質低下が小さい可能性がある。ただし #104 の結果では現行Model E候補は classical INR baselineを上回っていないため、この実験は採用判断ではなく量子化耐性の切り分けとして扱う。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe experiments/exp_026_inr_quantization_depth.py`
- Date: 2026-07-05
- Issue: #119
- Experiment seed: 20260705
- Output size: 64x64
- Low guide size: 16x16
- Low guide method: block average from 64x64 reference crop
- Fixture manifest: `experiments/assets/source_split_grayscale/manifest.json`
- Split policy: development/evaluation are separated by source image.
- Fit steps: 48
- Initial step scale: 0.035
- Parameter quantization depths: 8, 12, 16 bits in `[-1, 1]`
- Python / dependency version: Python 3.12.13, NumPy 2.3.5

## Baseline

画像baselineは nearest、bilinear、bicubic。Parameterized residual candidates は RFF、SIREN、small MLP、Model E single-state、Model E coupled-state。各candidateは一度float parameterをfitし、その同じparameterを8/12/16-bitに再量子化した。

## Metrics

- MAD、PSNR、global SSIM、gradient magnitude MAD、gradient magnitude correlation、Laplacian MAD
- incremental side bits と bits per output pixel
- float-to-quantized MAD delta
- fit time、decode time

## Result

| Split | Candidate | Family | Bit depth | Mean side bits | Mean float MAD | Mean quantized MAD | Float-to-quantized MAD delta | Mean PSNR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| development | mlp_small | mlp | 8 | 536 | 0.048495 | 0.048520 | +0.000026 | 22.835 |
| development | mlp_small | mlp | 12 | 708 | 0.048495 | 0.048545 | +0.000050 | 22.835 |
| development | mlp_small | mlp | 16 | 880 | 0.048495 | 0.048549 | +0.000054 | 22.834 |
| development | model_e_coupled | model_e_coupled | 8 | 584 | 0.048935 | 0.048935 | +0.000000 | 22.658 |
| development | model_e_coupled | model_e_coupled | 12 | 780 | 0.048935 | 0.048935 | +0.000000 | 22.658 |
| development | model_e_coupled | model_e_coupled | 16 | 976 | 0.048935 | 0.048935 | +0.000000 | 22.658 |
| development | model_e_single | model_e_single | 8 | 448 | 0.049300 | 0.049376 | +0.000076 | 22.679 |
| development | model_e_single | model_e_single | 12 | 576 | 0.049300 | 0.049301 | +0.000001 | 22.679 |
| development | model_e_single | model_e_single | 16 | 704 | 0.049300 | 0.049303 | +0.000003 | 22.679 |
| development | rff_small | rff | 8 | 448 | 0.048502 | 0.050164 | +0.001662 | 22.623 |
| development | rff_small | rff | 12 | 576 | 0.048502 | 0.050365 | +0.001863 | 22.608 |
| development | rff_small | rff | 16 | 704 | 0.048502 | 0.050361 | +0.001860 | 22.608 |
| development | siren_small | siren | 8 | 528 | 0.049232 | 0.049062 | -0.000170 | 22.663 |
| development | siren_small | siren | 12 | 696 | 0.049232 | 0.049104 | -0.000128 | 22.661 |
| development | siren_small | siren | 16 | 864 | 0.049232 | 0.049099 | -0.000133 | 22.661 |
| evaluation | mlp_small | mlp | 8 | 536 | 0.088811 | 0.089057 | +0.000246 | 17.863 |
| evaluation | mlp_small | mlp | 12 | 708 | 0.088811 | 0.089009 | +0.000199 | 17.864 |
| evaluation | mlp_small | mlp | 16 | 880 | 0.088811 | 0.089015 | +0.000204 | 17.863 |
| evaluation | model_e_coupled | model_e_coupled | 8 | 584 | 0.090095 | 0.090095 | +0.000000 | 17.756 |
| evaluation | model_e_coupled | model_e_coupled | 12 | 780 | 0.090095 | 0.090095 | +0.000000 | 17.756 |
| evaluation | model_e_coupled | model_e_coupled | 16 | 976 | 0.090095 | 0.090095 | +0.000000 | 17.756 |
| evaluation | model_e_single | model_e_single | 8 | 448 | 0.090070 | 0.090106 | +0.000036 | 17.767 |
| evaluation | model_e_single | model_e_single | 12 | 576 | 0.090070 | 0.090061 | -0.000009 | 17.767 |
| evaluation | model_e_single | model_e_single | 16 | 704 | 0.090070 | 0.090059 | -0.000011 | 17.767 |
| evaluation | rff_small | rff | 8 | 448 | 0.088828 | 0.093846 | +0.005018 | 17.709 |
| evaluation | rff_small | rff | 12 | 576 | 0.088828 | 0.093420 | +0.004592 | 17.716 |
| evaluation | rff_small | rff | 16 | 704 | 0.088828 | 0.093427 | +0.004599 | 17.716 |
| evaluation | siren_small | siren | 8 | 528 | 0.089741 | 0.090669 | +0.000928 | 17.737 |
| evaluation | siren_small | siren | 12 | 696 | 0.089741 | 0.090546 | +0.000805 | 17.738 |
| evaluation | siren_small | siren | 16 | 864 | 0.089741 | 0.090543 | +0.000802 | 17.738 |

Evaluation split summary:

- 8-bit: best classical `mlp_small` MAD 0.089057; best Model E `model_e_coupled` MAD 0.090095.
- 12-bit: best classical `mlp_small` MAD 0.089009; best Model E `model_e_single` MAD 0.090061.
- 16-bit: best classical `mlp_small` MAD 0.089015; best Model E `model_e_single` MAD 0.090059.

## Saved Artifacts

- `config.json`
- `metrics.json`
- `rate_distortion.csv`
- `notes.md`
- Per-case directories with `high_reference.png`, `low_guide.png`, nearest/bilinear/bicubic baselines, `*_8bit.png`, `*_12bit.png`, `*_16bit.png`, and `comparison_8bit.png`

## Images

![Development Hobbema TL 8-bit comparison](development_hobbema_landscape_tl/comparison_8bit.png)

![Development Hobbema BR 8-bit comparison](development_hobbema_landscape_br/comparison_8bit.png)

![Evaluation Hokusai TL 8-bit comparison](evaluation_hokusai_wave_tl/comparison_8bit.png)

![Evaluation Hokusai BR 8-bit comparison](evaluation_hokusai_wave_br/comparison_8bit.png)

## Interpretation

このrunでは、8/12/16-bitのいずれでも best Model E 候補が best classical INR 候補をevaluation splitのmean quantized MADで上回ったとは解釈しない。Model E single は8-bitでfloat-to-quantized deltaが小さいが、float時点のMADがclassical候補より高く、低bit耐性だけで採用根拠にはならない。

## Limitations

- Development source 1件、evaluation source 1件からの64x64 crop 2件ずつという小規模datasetである。
- `incremental_side_bits` はparameter side informationだけで、guide bits、container overhead、entropy codingを含まない。
- 同じrandom-search fit後の再量子化比較であり、bit depthごとに再fitしていない。
- 8-bit PNG保存画像は可視化artifactであり、metricsはNumPy配列上で計算した。
- 実用圧縮、super-resolution、量子優位は主張しない。

## Next

- Model E系列の継続/保留判断では、bit-depth耐性だけでなく #98 / #104 / #117 / #122 / #119 の負の結果をまとめて扱う。
- 追加実験をする場合は、bit-depthごとの再fitまたはoptional autograd backendでの同条件比較を別Issueに分ける。
