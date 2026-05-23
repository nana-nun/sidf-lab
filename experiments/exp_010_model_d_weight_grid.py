"""Run a small Model D confidence/data/texture weight grid."""

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

from experiments.exp_005_model_d_shape_benchmark import (
    model_d_decode,
    save_comparison_png,
    seeded_texture,
    upscale,
)
from experiments.exp_008_model_d_natural_patch import (
    downscale_block_average,
    gradient_magnitude,
    natural_image_metrics,
)
from sidf_lab.confidence import gradient_confidence
from sidf_lab.guides import cross
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json
from sidf_lab.metrics import comparison_summary, mad


RESULT_DIR = Path("results/2026-05-24-model-d-weight-grid")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-05-24"
EXPERIMENT_SEED = 20260524
CROSS_DECODER_SEED = 6500
NATURAL_DECODER_SEED = 6501
CROSS_LOW_SIZE = 16
CROSS_HIGH_SIZE = 64
CROSS_LOW_WIDTH = 2
CROSS_HIGH_WIDTH = 7
NATURAL_LOW_SIZE = 32
NATURAL_HIGH_SIZE = 128

GRID_CONFIGS = [
    {
        "id": "current_tex0",
        "label": "current weights, texture off",
        "lambda_data": 6.0,
        "confidence_min": 0.2,
        "confidence_max": 1.0,
        "confidence_scale": 4.0,
        "texture_strength": 0.0,
    },
    {
        "id": "low_data_tex0",
        "label": "lower data fidelity, texture off",
        "lambda_data": 3.0,
        "confidence_min": 0.2,
        "confidence_max": 1.0,
        "confidence_scale": 4.0,
        "texture_strength": 0.0,
    },
    {
        "id": "high_data_tex0",
        "label": "higher data fidelity, texture off",
        "lambda_data": 12.0,
        "confidence_min": 0.2,
        "confidence_max": 1.0,
        "confidence_scale": 4.0,
        "texture_strength": 0.0,
    },
    {
        "id": "high_floor_tex0",
        "label": "higher confidence floor, texture off",
        "lambda_data": 6.0,
        "confidence_min": 0.5,
        "confidence_max": 1.0,
        "confidence_scale": 4.0,
        "texture_strength": 0.0,
    },
    {
        "id": "flat_conf_tex0",
        "label": "uniform confidence, texture off",
        "lambda_data": 6.0,
        "confidence_min": 1.0,
        "confidence_max": 1.0,
        "confidence_scale": 4.0,
        "texture_strength": 0.0,
    },
    {
        "id": "current_tex035",
        "label": "current weights, texture 0.35",
        "lambda_data": 6.0,
        "confidence_min": 0.2,
        "confidence_max": 1.0,
        "confidence_scale": 4.0,
        "texture_strength": 0.35,
    },
]

COMMON_MODEL_PARAMS = {
    "j_base": 1.8,
    "gamma": 35.0,
}
CROSS_DECODE_CONFIG = {
    "sweeps": 35,
    "temp_start": 0.35,
    "temp_end": 0.01,
    "proposal_sigma": 0.08,
}
NATURAL_DECODE_CONFIG = {
    "sweeps": 18,
    "temp_start": 0.35,
    "temp_end": 0.01,
    "proposal_sigma": 0.08,
}


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run a callable and return its result plus elapsed seconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def texture_strength_to_field(base_texture: np.ndarray, strength: float) -> np.ndarray:
    """Scale the existing Model D white-noise texture path for a grid point."""
    if strength == 0.0:
        return np.zeros_like(base_texture)
    return base_texture * (strength / 0.35)


def make_confidence(guide: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Build the confidence map for a grid point."""
    return gradient_confidence(
        guide,
        min_confidence=float(cfg["confidence_min"]),
        max_confidence=float(cfg["confidence_max"]),
        scale=float(cfg["confidence_scale"]),
    )


def decode_grid_point(
    guide: np.ndarray,
    base_texture: np.ndarray,
    cfg: dict[str, Any],
    *,
    decoder_seed: int,
    decode_config: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decode one grid point and return output, confidence, and seconds."""
    confidence = make_confidence(guide, cfg)
    texture_strength = float(cfg["texture_strength"])
    texture = texture_strength_to_field(base_texture, texture_strength)
    rendered, seconds = timed_call(
        model_d_decode,
        guide,
        confidence,
        texture,
        decoder_seed=decoder_seed,
        lambda_data=float(cfg["lambda_data"]),
        texture_weight=texture_strength,
        **COMMON_MODEL_PARAMS,
        **decode_config,
    )
    return rendered, confidence, seconds


def metric_with_gradient(candidate: np.ndarray, reference: np.ndarray, foreground_mask: np.ndarray) -> dict[str, float | None]:
    """Return synthetic-shape metrics with a simple gradient MAD."""
    values = comparison_summary(candidate, reference=reference, foreground_mask=foreground_mask)
    values["gradient_mad_vs_reference"] = mad(gradient_magnitude(candidate), gradient_magnitude(reference))
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    values["mean_error_vs_reference"] = float(diff.mean())
    return values


def run_cross_case() -> dict[str, Any]:
    """Run the grid on a synthetic cross."""
    case_dir = ensure_dir(RESULT_DIR / "cross")
    high_reference = cross(size=CROSS_HIGH_SIZE, width=CROSS_HIGH_WIDTH, value=0.5)
    low_guide = cross(size=CROSS_LOW_SIZE, width=CROSS_LOW_WIDTH, value=0.5)
    foreground_mask = high_reference > 0.0
    nearest, nearest_seconds = timed_call(upscale, low_guide, CROSS_HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, CROSS_HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, CROSS_HIGH_SIZE, 3)
    base_texture = seeded_texture(bilinear.shape, seed=EXPERIMENT_SEED)

    outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
    }
    timings = {
        "nearest": nearest_seconds,
        "bilinear": bilinear_seconds,
        "bicubic": bicubic_seconds,
    }
    confidences: dict[str, np.ndarray] = {}

    for cfg in GRID_CONFIGS:
        rendered, confidence, seconds = decode_grid_point(
            bilinear,
            base_texture,
            cfg,
            decoder_seed=CROSS_DECODER_SEED,
            decode_config=CROSS_DECODE_CONFIG,
        )
        outputs[str(cfg["id"])] = rendered
        confidences[str(cfg["id"])] = confidence
        timings[str(cfg["id"])] = seconds

    metrics = {
        name: metric_with_gradient(image, high_reference, foreground_mask)
        for name, image in outputs.items()
    }
    save_case_artifacts(
        case_dir,
        outputs,
        reference=high_reference,
        low_guide=low_guide,
        baseline_for_diff=bilinear,
        confidences=confidences,
        comparison_names=["nearest", "bilinear", "bicubic", "current_tex0", "high_data_tex0", "flat_conf_tex0", "current_tex035"],
    )
    return {"metrics": metrics, "timings": timings}


def run_natural_case() -> dict[str, Any]:
    """Run the grid on the saved public-domain natural patch."""
    case_dir = ensure_dir(RESULT_DIR / "natural_patch")
    high_reference = np.load(SOURCE_ASSET)
    low_guide = downscale_block_average(high_reference, NATURAL_LOW_SIZE)
    nearest, nearest_seconds = timed_call(upscale, low_guide, NATURAL_HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, NATURAL_HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, NATURAL_HIGH_SIZE, 3)
    base_texture = seeded_texture(bilinear.shape, seed=EXPERIMENT_SEED)

    outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
    }
    timings = {
        "nearest": nearest_seconds,
        "bilinear": bilinear_seconds,
        "bicubic": bicubic_seconds,
    }
    confidences: dict[str, np.ndarray] = {}

    for cfg in GRID_CONFIGS:
        rendered, confidence, seconds = decode_grid_point(
            bilinear,
            base_texture,
            cfg,
            decoder_seed=NATURAL_DECODER_SEED,
            decode_config=NATURAL_DECODE_CONFIG,
        )
        outputs[str(cfg["id"])] = rendered
        confidences[str(cfg["id"])] = confidence
        timings[str(cfg["id"])] = seconds

    metrics = {name: natural_image_metrics(image, high_reference) for name, image in outputs.items()}
    save_case_artifacts(
        case_dir,
        outputs,
        reference=high_reference,
        low_guide=low_guide,
        baseline_for_diff=bilinear,
        confidences=confidences,
        comparison_names=["nearest", "bilinear", "bicubic", "current_tex0", "high_data_tex0", "flat_conf_tex0", "current_tex035"],
    )
    return {"metrics": metrics, "timings": timings}


def save_case_artifacts(
    case_dir: Path,
    outputs: dict[str, np.ndarray],
    *,
    reference: np.ndarray,
    low_guide: np.ndarray,
    baseline_for_diff: np.ndarray,
    confidences: dict[str, np.ndarray],
    comparison_names: list[str],
) -> None:
    """Save PNG artifacts for one grid case."""
    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    for name, image in outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
        if name not in {"nearest", "bilinear", "bicubic"}:
            save_grayscale_png(case_dir / f"diff_{name}_vs_bilinear.png", np.abs(image - baseline_for_diff))
            save_grayscale_png(case_dir / f"diff_{name}_vs_reference.png", np.abs(image - reference))
    for name, confidence in confidences.items():
        save_grayscale_png(case_dir / f"confidence_{name}.png", confidence)
    guide_for_strip = low_guide
    if guide_for_strip.shape != reference.shape:
        guide_for_strip = upscale(low_guide, reference.shape[0], 0)
    save_comparison_png(case_dir / "comparison.png", [reference, guide_for_strip, *[outputs[name] for name in comparison_names]])


def best_model_by(metric_map: dict[str, dict[str, float | None]], metric_name: str) -> str:
    """Return the grid config id with the smallest metric value."""
    best_name = ""
    best_value = float("inf")
    for cfg in GRID_CONFIGS:
        name = str(cfg["id"])
        value = metric_map[name][metric_name]
        if value is not None and float(value) < best_value:
            best_name = name
            best_value = float(value)
    return best_name


def build_notes(config: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Build the top-level Japanese experiment note."""
    cross_rows = format_cross_rows(metrics["cross"]["metrics"], metrics["cross"]["timings"])
    natural_rows = format_natural_rows(metrics["natural_patch"]["metrics"], metrics["natural_patch"]["timings"])
    cross_best = best_model_by(metrics["cross"]["metrics"], "mad_vs_reference")
    natural_best = best_model_by(metrics["natural_patch"]["metrics"], "mad_vs_gt")
    return f"""# Model D Weight Grid

## Question

Model D candidate の confidence / data fidelity / texture の主要重みを小規模gridで振ると、nearest / bilinear / bicubic baseline より悪化している原因を切り分けられるか。

## Hypothesis

Issue #37 の texture ablation では、white-noise texture_strength は synthetic cross のbaseline指標を改善しなかった。そのため、`texture_strength=0` を含めて `lambda_data` と confidence map の効き方を分けると、textureよりも data fidelity / confidence 設定が悪化要因として見える可能性がある。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_010_model_d_weight_grid.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Cross decoder seed: {config["cross_decoder_seed"]}
- Natural patch decoder seed: {config["natural_decoder_seed"]}
- Cross: {config["cross_low_size"]}x{config["cross_low_size"]} guide to {config["cross_high_size"]}x{config["cross_high_size"]} output
- Natural patch: {config["natural_low_size"]}x{config["natural_low_size"]} guide to {config["natural_high_size"]}x{config["natural_high_size"]} output
- Grid configs: `config.json` の `grid_configs`
- Common model params: `{config["common_model_params"]}`
- Cross decode config: `{config["cross_decode_config"]}`
- Natural decode config: `{config["natural_decode_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

baselineは cross / natural patch の両方で nearest、bilinear、bicubic upscaling とした。Model D grid はすべて同じ low guide と decoder seed を使う。自然画像patchでは128x128画像をGround Truthとし、32x32 block-average guideから復元した。

## Metrics

### Cross

| Output | MAD vs reference | PSNR | SSIM | Gradient MAD | Edge leakage | Mean error | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{cross_rows}

### Natural Patch

| Output | MAD vs GT | PSNR | SSIM | Gradient MAD | Strong-edge MAD | Mean error | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{natural_rows}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Notes: `notes.md`
- Cross artifacts: `cross/`
- Natural patch artifacts: `natural_patch/`
- Each case includes baseline PNGs, grid rendered PNGs, confidence maps, difference maps, and `comparison.png`.

## Images

![Cross comparison](cross/comparison.png)

![Natural patch comparison](natural_patch/comparison.png)

## Result

Cross の Model D grid内では `{cross_best}` が最小MADだった。Natural patch の Model D grid内では `{natural_best}` が最小MADだった。

## Interpretation

この小規模gridでは、Model D grid のどの条件も nearest / bilinear / bicubic baseline を総合的に上回ったとは解釈しない。`texture_strength=0` を含めてもbaseline差分は残るため、Issue #37 の結果と合わせると、white-noise textureだけではなく、現行の relaxation、confidence map、data fidelity、pairwise interaction の組み合わせ自体を再設計またはより細かく切り分ける必要がある。

confidence map の効果と texture term の効果は混ぜて解釈しない。`flat_conf_tex0` は confidence の空間変化を外した比較条件であり、`current_tex035` は texture 経路を残した現行寄り条件である。両者を直接「質感生成の良し悪し」として扱わず、baseline差分とmetricsの変化として読む。

## Limitations

- gridは6条件のみで、最適化探索ではない。
- crossと1枚のpublic-domain自然画像patchだけの結果である。
- Global SSIM は依存なしの全画像SSIMであり、windowed SSIMではない。
- 現行実装の texture はdraft仕様の線形項そのものではなく、texture target二乗項と初期状態混入を含む。
- decode timeはこの環境の小画像runに限る。
- この結果はsuper-resolutionやcompressionの成立を示さず、またそれらを一般に否定する結果でもない。

## Next

- 次に進めるなら、現行Model Dの式を固定したgrid拡大よりも、data fidelity / pairwise interaction / confidence map の設計を分離した小さな対照実験にする。Follow-up: Issue #61。
- structured texture prior を評価する場合も、`texture_strength=0` と white-noise baseline を含め、意味的ディテール生成とは断定しない。
"""


def format_cross_rows(metrics: dict[str, dict[str, float | None]], timings: dict[str, float]) -> str:
    names = ["nearest", "bilinear", "bicubic", *[str(cfg["id"]) for cfg in GRID_CONFIGS]]
    return "\n".join(
        "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {grad:.6f} | {edge:.6f} | {bias:.6f} | {seconds:.6f} |".format(
            name=name,
            mad=metrics[name]["mad_vs_reference"],
            psnr=metrics[name]["psnr_vs_reference"],
            ssim=metrics[name]["ssim_global_vs_reference"],
            grad=metrics[name]["gradient_mad_vs_reference"],
            edge=metrics[name]["edge_leakage"],
            bias=metrics[name]["mean_error_vs_reference"],
            seconds=timings[name],
        )
        for name in names
    )


def format_natural_rows(metrics: dict[str, dict[str, float]], timings: dict[str, float]) -> str:
    names = ["nearest", "bilinear", "bicubic", *[str(cfg["id"]) for cfg in GRID_CONFIGS]]
    return "\n".join(
        "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {grad:.6f} | {edge:.6f} | {bias:.6f} | {seconds:.6f} |".format(
            name=name,
            mad=metrics[name]["mad_vs_gt"],
            psnr=metrics[name]["psnr_vs_gt"],
            ssim=metrics[name]["ssim_global_vs_gt"],
            grad=metrics[name]["gradient_mad_vs_gt"],
            edge=metrics[name]["strong_edge_mad_vs_gt"],
            bias=metrics[name]["mean_error"],
            seconds=timings[name],
        )
        for name in names
    )


def main() -> None:
    ensure_dir(RESULT_DIR)
    results = {
        "cross": run_cross_case(),
        "natural_patch": run_natural_case(),
    }
    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "cross_decoder_seed": CROSS_DECODER_SEED,
        "natural_decoder_seed": NATURAL_DECODER_SEED,
        "cross_low_size": CROSS_LOW_SIZE,
        "cross_high_size": CROSS_HIGH_SIZE,
        "cross_low_width": CROSS_LOW_WIDTH,
        "cross_high_width": CROSS_HIGH_WIDTH,
        "natural_low_size": NATURAL_LOW_SIZE,
        "natural_high_size": NATURAL_HIGH_SIZE,
        "source_asset": SOURCE_ASSET.as_posix(),
        "grid_configs": GRID_CONFIGS,
        "common_model_params": COMMON_MODEL_PARAMS,
        "cross_decode_config": CROSS_DECODE_CONFIG,
        "natural_decode_config": NATURAL_DECODE_CONFIG,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", results)
    (RESULT_DIR / "notes.md").write_text(build_notes(config, results), encoding="utf-8")

    for case_name, case_results in results.items():
        print(case_name)
        metric_name = "mad_vs_reference" if case_name == "cross" else "mad_vs_gt"
        for name, values in case_results["metrics"].items():
            print(f"  {name}: {metric_name}={values[metric_name]:.6f} time={case_results['timings'][name]:.6f}s")


if __name__ == "__main__":
    main()
