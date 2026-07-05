"""Try the optional PyTorch optimizer backend on a tiny Model E fixture."""

from __future__ import annotations

import csv
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_005_model_d_shape_benchmark import save_comparison_png
from sidf_lab.guides import cross
from sidf_lab.inr_fit import INRSpec
from sidf_lab.inr_torch_fit import (
    OptimizerSpec,
    TorchBackendUnavailable,
    fit_inr_with_optimizer,
    torch_available,
)
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json
from sidf_lab.metrics import mad, psnr, ssim_global
from sidf_lab.model_e import QuantizationSpec, bilinear_resize


RESULT_DIR = Path("results/2026-07-05-issue-132-model-e-autograd-optimizer-spike")
DATE = "2026-07-05"
EXPERIMENT_SEED = 20260705
HIGH_SIZE = 16
LOW_SIZE = 4
QUANTIZATION = QuantizationSpec(bits_per_value=12, min_value=-1.0, max_value=1.0, header_bits=160)
OPTIMIZER = OptimizerSpec(backend="torch", method="adam", steps=8, learning_rate=0.03, seed=EXPERIMENT_SEED)


def image_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    return {
        "mad_vs_gt": mad(candidate, reference),
        "psnr_vs_gt": psnr(reference, candidate),
        "ssim_global_vs_gt": ssim_global(reference, candidate),
    }


def specs() -> list[tuple[str, INRSpec]]:
    return [
        ("fourier_order1", INRSpec("fourier", order=1, residual_limit=0.25)),
        ("model_e_single", INRSpec("model_e_single", depth=2, states=1, residual_limit=0.25)),
        ("model_e_coupled", INRSpec("model_e_coupled", depth=2, states=2, residual_limit=0.25)),
    ]


def save_trace_csv(path: Path, trace: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "gradient_norm"])
        writer.writeheader()
        writer.writerows(trace)


def main() -> None:
    ensure_dir(RESULT_DIR)
    reference = cross(HIGH_SIZE, width=3)
    low_guide = bilinear_resize(reference, (LOW_SIZE, LOW_SIZE))
    bilinear = bilinear_resize(low_guide, reference.shape)
    save_grayscale_png(RESULT_DIR / "high_reference.png", reference)
    save_grayscale_png(RESULT_DIR / "low_guide.png", low_guide)
    save_grayscale_png(RESULT_DIR / "bilinear.png", bilinear)

    config: dict[str, Any] = {
        "date": DATE,
        "issue": 132,
        "experiment_seed": EXPERIMENT_SEED,
        "high_size": HIGH_SIZE,
        "low_size": LOW_SIZE,
        "backend": OPTIMIZER.backend,
        "method": OPTIMIZER.method,
        "steps": OPTIMIZER.steps,
        "learning_rate": OPTIMIZER.learning_rate,
        "torch_available": torch_available(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "default_dependency_added": False,
    }
    metrics: dict[str, Any] = {
        "status": "pending",
        "baselines": {
            "bilinear": {
                "metrics": image_metrics(bilinear, reference),
            }
        },
        "outputs": {},
    }

    if not torch_available():
        metrics["status"] = "skipped"
        metrics["skip_reason"] = "PyTorch is not installed; optional backend is intentionally not a default dependency."
        save_json(RESULT_DIR / "config.json", config)
        save_json(RESULT_DIR / "metrics.json", metrics)
        (RESULT_DIR / "notes.md").write_text(build_notes(config, metrics), encoding="utf-8")
        print(metrics["skip_reason"])
        return

    try:
        images = [reference, bilinear]
        for index, (name, spec) in enumerate(specs()):
            run_optimizer = OptimizerSpec(
                backend=OPTIMIZER.backend,
                method=OPTIMIZER.method,
                steps=OPTIMIZER.steps,
                learning_rate=OPTIMIZER.learning_rate,
                seed=OPTIMIZER.seed + index,
            )
            result = fit_inr_with_optimizer(
                spec,
                low_guide,
                reference,
                optimizer=run_optimizer,
                quantization=QUANTIZATION,
            )
            trace = result.bits["optimizer"]["trace"]  # type: ignore[index]
            save_trace_csv(RESULT_DIR / f"{name}_loss_curve.csv", trace)  # type: ignore[arg-type]
            save_grayscale_png(RESULT_DIR / f"{name}_quantized.png", result.quantized_decoded)
            metrics["outputs"][name] = {
                "family": result.spec.family,
                "fit_seconds": result.fit_seconds,
                "decode_seconds": result.decode_seconds,
                "float_mse": result.float_mse,
                "quantized_mse": result.quantized_mse,
                "serialized_bits": result.bits["incremental_side_bits"],
                "metrics": image_metrics(result.quantized_decoded, reference),
                "optimizer": {
                    "backend": run_optimizer.backend,
                    "method": run_optimizer.method,
                    "steps": run_optimizer.steps,
                    "learning_rate": run_optimizer.learning_rate,
                    "final_loss": trace[-1]["loss"],  # type: ignore[index]
                },
            }
            images.append(result.quantized_decoded)
        metrics["status"] = "completed"
        save_comparison_png(RESULT_DIR / "comparison.png", images)
    except TorchBackendUnavailable as exc:
        metrics["status"] = "skipped"
        metrics["skip_reason"] = str(exc)

    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", metrics)
    (RESULT_DIR / "notes.md").write_text(build_notes(config, metrics), encoding="utf-8")
    print(metrics["status"])


def build_notes(config: dict[str, Any], metrics: dict[str, Any]) -> str:
    if metrics["status"] == "skipped":
        result = (
            "PyTorch がこの環境に未導入だったため、optional backend のfitは実行していない。"
            " ただし、default dependencyを増やさずに未導入時の挙動を保存artifactとして確認した。"
        )
        saved = "- `config.json`\n- `metrics.json`\n- `notes.md`\n- `high_reference.png`\n- `low_guide.png`\n- `bilinear.png`"
    else:
        result = (
            "PyTorch optional backendで、Fourier baseline、Model E single、Model E coupledを"
            "同じ Adam optimizer spec でfitした。"
        )
        saved = (
            "- `config.json`\n- `metrics.json`\n- `notes.md`\n- `comparison.png`\n"
            "- `*_quantized.png`\n- `*_loss_curve.csv`"
        )
    return f"""# Model E PyTorch CPU Optional Optimizer Spike

## Question

PyTorch CPU optional backendをdefault dependencyにせず、Model E と classical INR baselineへ同じoptimizer条件を適用できるか。

## Hypothesis

PyTorchが導入済みの環境では、同じ `OptimizerSpec` から Fourier baseline、Model E single、Model E coupled をfitできる。PyTorch未導入の環境では、エラーで崩れず、skip理由と成果物を保存できる。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_025_model_e_autograd_optimizer_spike.py`
- Date: {config["date"]}
- Issue: #{config["issue"]}
- Experiment seed: {config["experiment_seed"]}
- Output size: {config["high_size"]}x{config["high_size"]}
- Low guide size: {config["low_size"]}x{config["low_size"]}
- Backend: `{config["backend"]}`
- Method: `{config["method"]}`
- Steps: {config["steps"]}
- Learning rate: {config["learning_rate"]}
- PyTorch available: {config["torch_available"]}
- Default dependency added: {config["default_dependency_added"]}
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

画像baselineは bilinear。PyTorch が導入済みの場合は、classical INR baselineとして Fourier order 1、Model E候補として single / coupled を同じ Adam 条件で比較する。

## Result

{result}

## Saved Artifacts

{saved}

## Interpretation

このspikeは optimizer backend の接続確認であり、Model E の採用判断ではない。PyTorchが未導入の場合、default dependencyを増やさない方針が保たれていることと、未導入時に明示的にskipできることを確認する。

## Limitations

- 小さなsynthetic crossだけを使うため、一般的な画像品質は評価しない。
- PyTorch未導入時は autograd fit time、decode time、float/quantized MAD の候補比較は未測定である。
- compression、super-resolution、quantum advantageは主張しない。

## Next

- PyTorch導入済み環境で同じscriptを再実行し、optional backendのfit結果とloss curveを保存する。
- 結果が有用なら、source-split fixtureへ広げる別Issueを作る。
"""


if __name__ == "__main__":
    main()
