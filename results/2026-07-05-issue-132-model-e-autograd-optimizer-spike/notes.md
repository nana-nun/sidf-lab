# Model E PyTorch CPU Optional Optimizer Spike

## Question

PyTorch CPU optional backendをdefault dependencyにせず、Model E と classical INR baselineへ同じoptimizer条件を適用できるか。

## Hypothesis

PyTorchが導入済みの環境では、同じ `OptimizerSpec` から Fourier baseline、Model E single、Model E coupled をfitできる。PyTorch未導入の環境では、エラーで崩れず、skip理由と成果物を保存できる。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe experiments/exp_025_model_e_autograd_optimizer_spike.py`
- Date: 2026-07-05
- Issue: #132
- Experiment seed: 20260705
- Output size: 16x16
- Low guide size: 4x4
- Backend: `torch`
- Method: `adam`
- Steps: 8
- Learning rate: 0.03
- PyTorch available: False
- Default dependency added: False
- Python / dependency version: Python 3.12.13, NumPy 2.3.5

## Baseline

画像baselineは bilinear。PyTorch が導入済みの場合は、classical INR baselineとして Fourier order 1、Model E候補として single / coupled を同じ Adam 条件で比較する。

## Result

PyTorch がこの環境に未導入だったため、optional backend のfitは実行していない。 ただし、default dependencyを増やさずに未導入時の挙動を保存artifactとして確認した。

## Saved Artifacts

- `config.json`
- `metrics.json`
- `notes.md`
- `high_reference.png`
- `low_guide.png`
- `bilinear.png`

## Interpretation

このspikeは optimizer backend の接続確認であり、Model E の採用判断ではない。PyTorchが未導入の場合、default dependencyを増やさない方針が保たれていることと、未導入時に明示的にskipできることを確認する。

## Limitations

- 小さなsynthetic crossだけを使うため、一般的な画像品質は評価しない。
- PyTorch未導入時は autograd fit time、decode time、float/quantized MAD の候補比較は未測定である。
- compression、super-resolution、quantum advantageは主張しない。

## Next

- PyTorch導入済み環境で同じscriptを再実行し、optional backendのfit結果とloss curveを保存する。
- 結果が有用なら、source-split fixtureへ広げる別Issueを作る。
