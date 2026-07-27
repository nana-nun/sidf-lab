# Model E Coupling Variants

## Question

Model E の coupled-state 更新を、controlled-rotation風またはgated interaction風に変えると、single-state、現行coupled、classical INR baselineに対して評価splitで改善するか。

## Hypothesis

現行coupled-stateは単純な隣接state相互作用だけを使うため、入力featureから制御されるcouplingを追加すると一部の残差構造を表現しやすくなる可能性がある。一方で、追加parameterによるside-bit overheadが増えるため、改善がなければ採用候補ではなく負の切り分け結果として扱う。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe experiments/exp_027_model_e_coupling_variants.py`
- Date: 2026-07-27
- Issue: #118
- Experiment seed: 20260727
- Output size: 64x64
- Low guide size: 16x16
- Low guide method: block average from 64x64 reference crop
- Fixture manifest: `experiments/assets/source_split_grayscale/manifest.json`
- Split policy: development/evaluation are separated by source image.
- Fit steps: 48
- Initial step scale: 0.035
- Parameter quantization: signed uniform 12-bit values in `[-1, 1]`
- Python / dependency version: Python 3.12.13, NumPy 2.3.5

## Baseline

画像baselineは nearest、bilinear、bicubic。Parameterized residual baselineは `rff_small` と `mlp_small`。Model E側は `model_e_single`、現行 `model_e_coupled_current`、新候補 `model_e_controlled_rotation`、`model_e_gated_coupled` を同じ `fit_inr` interface、同じstep数、同じ12-bit量子化で比較した。

## Result

| Split | Candidate | Role | Mean side bits | Coupling overhead bits | Mean float MAD | Mean quantized MAD | Mean gradient MAD | Mean decode seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| development | mlp_small | classical_baseline | 708 | N/A | 0.048495 | 0.048545 | 0.034111 | 0.00107 |
| development | model_e_controlled_rotation | coupling_variant | 1320 | 540 | 0.048274 | 0.048392 | 0.033992 | 0.00346 |
| development | model_e_coupled_current | current_coupled_baseline | 780 | N/A | 0.048935 | 0.048935 | 0.034390 | 0.00246 |
| development | model_e_gated_coupled | coupling_variant | 1320 | 540 | 0.048935 | 0.048935 | 0.034390 | 0.00314 |
| development | model_e_single | single_state_baseline | 576 | N/A | 0.049300 | 0.049301 | 0.034387 | 0.00133 |
| development | rff_small | classical_baseline | 576 | N/A | 0.048502 | 0.050365 | 0.034359 | 0.00141 |
| evaluation | mlp_small | classical_baseline | 708 | N/A | 0.088811 | 0.089009 | 0.065856 | 0.00143 |
| evaluation | model_e_controlled_rotation | coupling_variant | 1320 | 540 | 0.090194 | 0.090181 | 0.066379 | 0.00340 |
| evaluation | model_e_coupled_current | current_coupled_baseline | 780 | N/A | 0.090095 | 0.090095 | 0.066386 | 0.00255 |
| evaluation | model_e_gated_coupled | coupling_variant | 1320 | 540 | 0.089077 | 0.089183 | 0.065811 | 0.00373 |
| evaluation | model_e_single | single_state_baseline | 576 | N/A | 0.090070 | 0.090061 | 0.066337 | 0.00143 |
| evaluation | rff_small | classical_baseline | 576 | N/A | 0.088828 | 0.093420 | 0.066039 | 0.00150 |

Evaluation splitで最良classical baselineは `mlp_small` の mean quantized MAD `0.089009`、最良Model E系候補は `model_e_gated_coupled` の `0.089183` だった。

現行coupledと新coupling候補の比較では、現行coupled `model_e_coupled_current` が `0.090095`、controlled rotation が `0.090181`、gated coupled が `0.089183` だった。新候補のcoupling overheadは現行coupled比で平均 `540` bits だった。

## Saved Artifacts

- `config.json`
- `metrics.json`
- `notes.md`
- Per-case directories with `high_reference.png`, `low_guide.png`, nearest/bilinear/bicubic baselines, `*_quantized.png`, and `comparison.png`

## Images

![Development Hobbema TL comparison](development_hobbema_landscape_tl/comparison.png)

![Development Hobbema BR comparison](development_hobbema_landscape_br/comparison.png)

![Evaluation Hokusai TL comparison](evaluation_hokusai_wave_tl/comparison.png)

![Evaluation Hokusai BR comparison](evaluation_hokusai_wave_br/comparison.png)

## Interpretation

このrunでは、`model_e_gated_coupled` は現行coupled-stateよりevaluation splitのmean quantized MADを改善した。一方で、best classical INRの `mlp_small` には届かず、現行coupledに対して平均540 bitsのcoupling overheadも増えた。したがって、今回のfixtureとrandom-search fit条件では、gated interaction候補を採用候補へ戻す根拠としては不足している。`model_e_controlled_rotation` は現行coupledを改善しなかった。

## Limitations

- Development source 1件、evaluation source 1件からの64x64 crop 2件ずつという小規模datasetである。
- dependency-free random searchであり、各candidateの到達可能品質を保証しない。
- `incremental_side_bits` はparameter side informationだけで、guide bits、container overhead、entropy codingを含まない。
- 新coupling候補は2案だけであり、coupling設計一般の否定ではない。
- 実用圧縮、super-resolution、量子優位は主張しない。

## Next

- #118 の結果を Model E 系列の継続/保留判断へ反映する。
- Model Eをさらに続ける場合は、新しいcoupling式の追加より先に optional autograd optimizer の同条件実測、またはModel E系列の一時保留判断を分けて扱う。
