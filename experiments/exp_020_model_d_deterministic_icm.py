"""Compare Gaussian-proposal greedy updates with deterministic ICM."""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_005_model_d_shape_benchmark import save_comparison_png, upscale
from experiments.exp_008_model_d_natural_patch import downscale_block_average
from experiments.exp_018_model_d_redesign_candidates import cross_metrics, natural_metrics
from experiments.exp_019_model_d_update_procedure import decode
from sidf_lab.anneal import quadratic_coordinate_descent
from sidf_lab.guides import cross
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json


RESULT_DIR = Path("results/2026-06-14-issue-88-model-d-deterministic-icm")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-06-14"
EXPERIMENT_SEED = 20260614
CROSS_DECODER_SEED = 8800
NATURAL_DECODER_SEED = 8801
MODEL_PARAMS = {"j_base": 1.8, "lambda_data": 6.0, "gamma": 35.0}
CROSS_SWEEPS = 35
NATURAL_SWEEPS = 18
PROPOSAL_SIGMA = 0.08
ICM_TOLERANCE = 1e-12


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def run_case(
    case_name: str,
    reference: np.ndarray,
    low_guide: np.ndarray,
    *,
    decoder_seed: int,
    sweeps: int,
    foreground_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    case_dir = ensure_dir(RESULT_DIR / case_name)
    output_size = reference.shape[0]
    nearest, nearest_seconds = timed_call(upscale, low_guide, output_size, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, output_size, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, output_size, 3)
    confidence = np.ones_like(bilinear)

    (greedy, greedy_diagnostics), greedy_seconds = timed_call(
        decode,
        bilinear,
        acceptance_mode="greedy",
        update_order="fixed",
        decoder_seed=decoder_seed,
        sweeps=sweeps,
        temp_start=0.35,
        temp_end=0.01,
        proposal_sigma=PROPOSAL_SIGMA,
        pairwise_cap=0.08,
        **MODEL_PARAMS,
    )
    (icm, icm_diagnostics), icm_seconds = timed_call(
        quadratic_coordinate_descent,
        bilinear,
        bilinear,
        confidence,
        max_sweeps=sweeps,
        tolerance=ICM_TOLERANCE,
        **MODEL_PARAMS,
    )

    greedy_diagnostics["objective_decrease"] = (
        float(greedy_diagnostics["initial_objective"])
        - float(greedy_diagnostics["final_objective"])
    )
    greedy_diagnostics["updates"] = int(greedy_diagnostics["accepted"])
    greedy_diagnostics["sweeps_completed"] = sweeps
    greedy_diagnostics["converged"] = False

    outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
        "greedy_fixed": greedy,
        "deterministic_icm": icm,
    }
    timings = {
        "nearest": nearest_seconds,
        "bilinear": bilinear_seconds,
        "bicubic": bicubic_seconds,
        "greedy_fixed": greedy_seconds,
        "deterministic_icm": icm_seconds,
    }
    if foreground_mask is None:
        metrics = {name: natural_metrics(image, reference) for name, image in outputs.items()}
    else:
        metrics = {
            name: cross_metrics(image, reference, foreground_mask)
            for name, image in outputs.items()
        }

    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    for name, image in outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
        if name in {"greedy_fixed", "deterministic_icm"}:
            save_grayscale_png(
                case_dir / f"diff_{name}_vs_bilinear.png", np.abs(image - bilinear)
            )
            save_grayscale_png(
                case_dir / f"diff_{name}_vs_reference.png", np.abs(image - reference)
            )
    save_comparison_png(
        case_dir / "comparison.png",
        [
            reference,
            upscale(low_guide, output_size, 0),
            nearest,
            bilinear,
            bicubic,
            greedy,
            icm,
        ],
    )
    return {
        "metrics": metrics,
        "timings": timings,
        "diagnostics": {
            "greedy_fixed": greedy_diagnostics,
            "deterministic_icm": icm_diagnostics,
        },
    }


def metric_value(case_name: str, values: dict[str, float | None]) -> float:
    key = "mad_vs_reference" if case_name == "cross" else "mad_vs_gt"
    return float(values[key])


def format_rows(case_name: str, result: dict[str, Any]) -> str:
    rows = []
    for name in ["nearest", "bilinear", "bicubic", "greedy_fixed", "deterministic_icm"]:
        metrics = result["metrics"][name]
        diagnostics = result["diagnostics"].get(name)
        objective = "N/A"
        decrease = "N/A"
        updates = "N/A"
        if diagnostics is not None:
            objective = f"{float(diagnostics['final_objective']):.6f}"
            decrease = f"{float(diagnostics['objective_decrease']):.6f}"
            updates = str(diagnostics["updates"])
        rows.append(
            "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {gradient:.6f} | "
            "{objective} | {decrease} | {updates} | {seconds:.6f} |".format(
                name=name,
                mad=metric_value(case_name, metrics),
                psnr=float(
                    metrics[
                        "psnr_vs_reference" if case_name == "cross" else "psnr_vs_gt"
                    ]
                ),
                ssim=float(
                    metrics[
                        "ssim_global_vs_reference"
                        if case_name == "cross"
                        else "ssim_global_vs_gt"
                    ]
                ),
                gradient=float(metrics["gradient_magnitude_mad"]),
                objective=objective,
                decrease=decrease,
                updates=updates,
                seconds=result["timings"][name],
            )
        )
    return "\n".join(rows)


def build_notes(config: dict[str, Any], results: dict[str, Any]) -> str:
    cross_icm = results["cross"]["diagnostics"]["deterministic_icm"]
    natural_icm = results["natural_patch"]["diagnostics"]["deterministic_icm"]
    cross_greedy = results["cross"]["diagnostics"]["greedy_fixed"]
    natural_greedy = results["natural_patch"]["diagnostics"]["greedy_fixed"]
    cross_icm_mad = metric_value(
        "cross", results["cross"]["metrics"]["deterministic_icm"]
    )
    natural_icm_mad = metric_value(
        "natural_patch", results["natural_patch"]["metrics"]["deterministic_icm"]
    )
    return f"""# Model D Deterministic ICM Evaluation

## Question

Model Dのquadratic objectiveで、Gaussian proposal greedyの探索不足とobjective自体のreference品質上の限界を分けられるか。

## Hypothesis

解析的な局所最小値へ更新するdeterministic ICMはgreedy fixedよりobjectiveを大きく低下させる。一方、objective低下がMAD、PSNR、SSIM、gradient magnitude MADの改善と一致するとは限らない。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_020_model_d_deterministic_icm.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Cross decoder seed: {config["cross_decoder_seed"]}
- Natural patch decoder seed: {config["natural_decoder_seed"]}
- Initial state: bilinear upscaled guide
- Confidence: uniform 1.0
- Texture: 0.0
- Pairwise: quadratic
- Model params: `{config["model_params"]}`
- Cross sweeps: {config["cross_sweeps"]}
- Natural patch sweeps: {config["natural_sweeps"]}
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

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
{format_rows("cross", results["cross"])}

### Natural Patch

| Output | MAD | PSNR | Global SSIM | Gradient magnitude MAD | Final objective | Objective decrease | Updates | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{format_rows("natural_patch", results["natural_patch"])}

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

ICMはcrossでobjectiveを `{cross_icm["initial_objective"]:.6f}` から `{cross_icm["final_objective"]:.6f}`、natural patchで `{natural_icm["initial_objective"]:.6f}` から `{natural_icm["final_objective"]:.6f}` へ低下させた。greedy fixedの最終objectiveはcross `{cross_greedy["final_objective"]:.6f}`、natural patch `{natural_greedy["final_objective"]:.6f}` だった。

ICMのMADはcross `{cross_icm_mad:.6f}`、natural patch `{natural_icm_mad:.6f}` だった。

## Interpretation

ICMはgreedy fixedより低いobjectiveへ到達したため、Gaussian proposal greedyにはquadratic objectiveを十分に下げきらない探索不足があった。

一方、ICMはgreedy fixedよりobjectiveを強く低下させながら、cross / natural patchのMAD、PSNR、SSIM、gradient magnitude MADを改善しなかった。crossとnatural patchの両方でbilinearよりMADが悪く、natural patchではbicubicも上回らなかった。この結果は、proposal改善だけでは現行quadratic objectiveをreference品質の改善要因にできないというnegative evidenceである。objective最小化とGround Truth差分最小化は別の評価軸として扱う。

## Limitations

- crossと1枚のpublic-domain自然画像patchだけの比較である。
- fixed row-majorのGauss-Seidel型更新だけを評価し、Jacobi更新や線形方程式の直接解法とは比較していない。
- ICMの収束判定は最大画素変化 `{config["icm_tolerance"]}` 以下であり、cross-environmentのbit-perfect再現性は未確認である。
- Global SSIMはwindowed SSIMではない。
- この結果はsuper-resolutionやcompressionの成立を示さない。

## Next

- Issue #92 で、有限温度Metropolisと現行quadratic objectiveを標準decoder候補として採用しない判断、およびdecoder procedure / objective designの未確定範囲をSIDF v0.3 draftへ反映する。
"""


def main() -> None:
    ensure_dir(RESULT_DIR)
    cross_reference = cross(size=64, width=7, value=0.5)
    cross_guide = cross(size=16, width=2, value=0.5)
    natural_reference = np.load(SOURCE_ASSET)
    natural_guide = downscale_block_average(natural_reference, 32)
    results = {
        "cross": run_case(
            "cross",
            cross_reference,
            cross_guide,
            decoder_seed=CROSS_DECODER_SEED,
            sweeps=CROSS_SWEEPS,
            foreground_mask=cross_reference > 0.0,
        ),
        "natural_patch": run_case(
            "natural_patch",
            natural_reference,
            natural_guide,
            decoder_seed=NATURAL_DECODER_SEED,
            sweeps=NATURAL_SWEEPS,
        ),
    }
    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "cross_decoder_seed": CROSS_DECODER_SEED,
        "natural_decoder_seed": NATURAL_DECODER_SEED,
        "source_asset": SOURCE_ASSET.as_posix(),
        "model_params": MODEL_PARAMS,
        "cross_sweeps": CROSS_SWEEPS,
        "natural_sweeps": NATURAL_SWEEPS,
        "proposal_sigma": PROPOSAL_SIGMA,
        "icm_tolerance": ICM_TOLERANCE,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", results)
    (RESULT_DIR / "notes.md").write_text(build_notes(config, results), encoding="utf-8")

    for case_name, case_result in results.items():
        print(case_name)
        for name in ["bilinear", "greedy_fixed", "deterministic_icm"]:
            print(f"  {name}: MAD={metric_value(case_name, case_result['metrics'][name]):.6f}")


if __name__ == "__main__":
    main()
