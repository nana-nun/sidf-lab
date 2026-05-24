"""Compare Model D white-noise and structured texture priors."""

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
from sidf_lab.texture import fractal_value_noise, smoothed_noise, white_noise


RESULT_DIR = Path("results/2026-05-24-issue-63-structured-texture-prior")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-05-24"
EXPERIMENT_SEED = 20260524
CROSS_DECODER_SEED = 6630
NATURAL_DECODER_SEED = 6631
CROSS_LOW_SIZE = 16
CROSS_HIGH_SIZE = 64
CROSS_LOW_WIDTH = 2
CROSS_HIGH_WIDTH = 7
NATURAL_LOW_SIZE = 32
NATURAL_HIGH_SIZE = 128
TEXTURE_SIGMA = 0.035
TEXTURE_STRENGTH = 0.35

TEXTURE_CONFIGS = [
    {"id": "texture_0", "label": "texture strength 0", "kind": "none", "strength": 0.0},
    {"id": "white_noise", "label": "white noise", "kind": "white", "strength": TEXTURE_STRENGTH},
    {"id": "smoothed_noise", "label": "smoothed noise", "kind": "smoothed", "strength": TEXTURE_STRENGTH},
    {"id": "fractal_value_noise", "label": "fractal value noise", "kind": "fractal", "strength": TEXTURE_STRENGTH},
]

MODEL_D_PARAMS = {
    "j_base": 1.8,
    "lambda_data": 6.0,
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
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def normalize_texture(field: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    centered = values - float(values.mean())
    std = float(centered.std())
    if std == 0.0 or sigma == 0.0:
        return np.zeros_like(centered)
    return centered / std * sigma


def texture_for(kind: str, shape: tuple[int, int], seed: int) -> np.ndarray:
    if kind == "none":
        return np.zeros(shape, dtype=np.float64)
    if kind == "white":
        return normalize_texture(white_noise(shape, seed=seed, sigma=TEXTURE_SIGMA), TEXTURE_SIGMA)
    if kind == "smoothed":
        return smoothed_noise(shape, seed=seed, sigma=TEXTURE_SIGMA, radius=3)
    if kind == "fractal":
        return fractal_value_noise(
            shape,
            seed=seed,
            octaves=4,
            base_frequency=2.0,
            lacunarity=2.0,
            gain=0.5,
            sigma=TEXTURE_SIGMA,
        )
    raise ValueError(f"unknown texture kind: {kind}")


def texture_preview(field: np.ndarray) -> np.ndarray:
    if not np.any(field):
        return np.full_like(field, 0.5, dtype=np.float64)
    return np.clip(field / (4.0 * TEXTURE_SIGMA) + 0.5, 0.0, 1.0)


def metric_with_gradient(candidate: np.ndarray, reference: np.ndarray, foreground_mask: np.ndarray) -> dict[str, float | None]:
    values = comparison_summary(candidate, reference=reference, foreground_mask=foreground_mask)
    values["gradient_mad_vs_reference"] = mad(gradient_magnitude(candidate), gradient_magnitude(reference))
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    values["mean_error_vs_reference"] = float(diff.mean())
    return values


def run_cross_case() -> dict[str, Any]:
    case_dir = ensure_dir(RESULT_DIR / "cross")
    high_reference = cross(size=CROSS_HIGH_SIZE, width=CROSS_HIGH_WIDTH, value=0.5)
    low_guide = cross(size=CROSS_LOW_SIZE, width=CROSS_LOW_WIDTH, value=0.5)
    foreground_mask = high_reference > 0.0
    nearest, nearest_seconds = timed_call(upscale, low_guide, CROSS_HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, CROSS_HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, CROSS_HIGH_SIZE, 3)
    confidence, confidence_seconds = timed_call(
        gradient_confidence,
        bilinear,
        min_confidence=0.2,
        max_confidence=1.0,
        scale=4.0,
    )

    outputs = {"nearest": nearest, "bilinear": bilinear, "bicubic": bicubic}
    timings = {"nearest": nearest_seconds, "bilinear": bilinear_seconds, "bicubic": bicubic_seconds, "confidence": confidence_seconds}
    textures: dict[str, np.ndarray] = {}
    for index, cfg in enumerate(TEXTURE_CONFIGS):
        texture = texture_for(str(cfg["kind"]), bilinear.shape, seed=EXPERIMENT_SEED + index)
        rendered, seconds = timed_call(
            model_d_decode,
            bilinear,
            confidence,
            texture,
            decoder_seed=CROSS_DECODER_SEED,
            texture_weight=float(cfg["strength"]),
            **MODEL_D_PARAMS,
            **CROSS_DECODE_CONFIG,
        )
        name = str(cfg["id"])
        outputs[name] = rendered
        timings[name] = seconds
        textures[name] = texture

    metrics = {name: metric_with_gradient(image, high_reference, foreground_mask) for name, image in outputs.items()}
    save_case_artifacts(
        case_dir,
        outputs,
        textures,
        reference=high_reference,
        low_guide=low_guide,
        confidence=confidence,
        baseline_for_diff=bilinear,
        comparison_names=["nearest", "bilinear", "bicubic", "texture_0", "white_noise", "smoothed_noise", "fractal_value_noise"],
    )
    return {"metrics": metrics, "timings": timings}


def run_natural_case() -> dict[str, Any]:
    case_dir = ensure_dir(RESULT_DIR / "natural_patch")
    high_reference = np.load(SOURCE_ASSET)
    low_guide = downscale_block_average(high_reference, NATURAL_LOW_SIZE)
    nearest, nearest_seconds = timed_call(upscale, low_guide, NATURAL_HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, NATURAL_HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, NATURAL_HIGH_SIZE, 3)
    confidence, confidence_seconds = timed_call(
        gradient_confidence,
        bilinear,
        min_confidence=0.2,
        max_confidence=1.0,
        scale=4.0,
    )

    outputs = {"nearest": nearest, "bilinear": bilinear, "bicubic": bicubic}
    timings = {"nearest": nearest_seconds, "bilinear": bilinear_seconds, "bicubic": bicubic_seconds, "confidence": confidence_seconds}
    textures: dict[str, np.ndarray] = {}
    for index, cfg in enumerate(TEXTURE_CONFIGS):
        texture = texture_for(str(cfg["kind"]), bilinear.shape, seed=EXPERIMENT_SEED + 100 + index)
        rendered, seconds = timed_call(
            model_d_decode,
            bilinear,
            confidence,
            texture,
            decoder_seed=NATURAL_DECODER_SEED,
            texture_weight=float(cfg["strength"]),
            **MODEL_D_PARAMS,
            **NATURAL_DECODE_CONFIG,
        )
        name = str(cfg["id"])
        outputs[name] = rendered
        timings[name] = seconds
        textures[name] = texture

    metrics = {name: natural_image_metrics(image, high_reference) for name, image in outputs.items()}
    save_case_artifacts(
        case_dir,
        outputs,
        textures,
        reference=high_reference,
        low_guide=low_guide,
        confidence=confidence,
        baseline_for_diff=bilinear,
        comparison_names=["nearest", "bilinear", "bicubic", "texture_0", "white_noise", "smoothed_noise", "fractal_value_noise"],
    )
    return {"metrics": metrics, "timings": timings}


def save_case_artifacts(
    case_dir: Path,
    outputs: dict[str, np.ndarray],
    textures: dict[str, np.ndarray],
    *,
    reference: np.ndarray,
    low_guide: np.ndarray,
    confidence: np.ndarray,
    baseline_for_diff: np.ndarray,
    comparison_names: list[str],
) -> None:
    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    save_grayscale_png(case_dir / "confidence.png", confidence)
    for name, image in outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
        if name not in {"nearest", "bilinear", "bicubic"}:
            save_grayscale_png(case_dir / f"diff_{name}_vs_bilinear.png", np.abs(image - baseline_for_diff))
            save_grayscale_png(case_dir / f"diff_{name}_vs_reference.png", np.abs(image - reference))
    for name, texture in textures.items():
        save_grayscale_png(case_dir / f"texture_field_{name}.png", texture_preview(texture))
    guide_for_strip = low_guide
    if guide_for_strip.shape != reference.shape:
        guide_for_strip = upscale(low_guide, reference.shape[0], 0)
    save_comparison_png(
        case_dir / "comparison.png",
        [reference, guide_for_strip, confidence, *[outputs[name] for name in comparison_names]],
    )
    save_comparison_png(
        case_dir / "texture_fields.png",
        [texture_preview(textures[name]) for name in ["texture_0", "white_noise", "smoothed_noise", "fractal_value_noise"]],
    )


def best_model_by(metric_map: dict[str, dict[str, float | None]], metric_name: str) -> str:
    best_name = ""
    best_value = float("inf")
    for cfg in TEXTURE_CONFIGS:
        name = str(cfg["id"])
        value = metric_map[name][metric_name]
        if value is not None and float(value) < best_value:
            best_name = name
            best_value = float(value)
    return best_name


def format_cross_rows(metrics: dict[str, dict[str, float | None]], timings: dict[str, float]) -> str:
    names = ["nearest", "bilinear", "bicubic", *[str(cfg["id"]) for cfg in TEXTURE_CONFIGS]]
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
    names = ["nearest", "bilinear", "bicubic", *[str(cfg["id"]) for cfg in TEXTURE_CONFIGS]]
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


def build_notes(config: dict[str, Any], results: dict[str, Any]) -> str:
    cross_rows = format_cross_rows(results["cross"]["metrics"], results["cross"]["timings"])
    natural_rows = format_natural_rows(results["natural_patch"]["metrics"], results["natural_patch"]["timings"])
    cross_best = best_model_by(results["cross"]["metrics"], "mad_vs_reference")
    natural_best = best_model_by(results["natural_patch"]["metrics"], "mad_vs_gt")
    return f"""# Structured Texture Prior Comparison

## Question

Model D の現行 white-noise texture baseline と比べて、smoothed noise / fractal value noise は cross と自然画像patchの再構成指標に改善要因として見えるか。

## Hypothesis

white noise は粒状感に寄りやすいため、smoothed noise や fractal value noise は視覚的な粒状差分を変える可能性がある。ただし、現行 Model D の relaxation / confidence / data fidelity の組み合わせでは、structured texture prior だけで nearest / bilinear / bicubic baseline を上回るとは仮定しない。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_014_structured_texture_prior.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Cross decoder seed: {config["cross_decoder_seed"]}
- Natural patch decoder seed: {config["natural_decoder_seed"]}
- Cross: {config["cross_low_size"]}x{config["cross_low_size"]} guide to {config["cross_high_size"]}x{config["cross_high_size"]} output
- Natural patch: {config["natural_low_size"]}x{config["natural_low_size"]} guide to {config["natural_high_size"]}x{config["natural_high_size"]} output
- Texture sigma: {config["texture_sigma"]}
- Texture configs: `config.json` の `texture_configs`
- Common Model D params: `{config["model_d_params"]}`
- Cross decode config: `{config["cross_decode_config"]}`
- Natural decode config: `{config["natural_decode_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

baseline は cross / natural patch の両方で nearest、bilinear、bicubic upscaling とした。Model D 条件には `texture_0`、`white_noise`、`smoothed_noise`、`fractal_value_noise` を含めた。structured texture prior は、white-noise baseline との差分として評価し、意味的ディテール生成とは扱わない。

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
- Each case includes baseline PNGs, texture field PNGs, rendered PNGs, difference maps, confidence map, and `comparison.png`.

## Images

![Cross comparison](cross/comparison.png)

![Cross texture fields](cross/texture_fields.png)

![Natural patch comparison](natural_patch/comparison.png)

![Natural patch texture fields](natural_patch/texture_fields.png)

## Result

Cross の Model D texture 条件内では `{cross_best}` が最小MADだった。Natural patch の Model D texture 条件内では `{natural_best}` が最小MADだった。

## Interpretation

このrunでは、structured texture prior 候補が単純補間 baseline を総合的に上回ったとは解釈しない。white noise と structured texture の差は同じ settings で比較できる形になったが、cross と自然画像patchのどちらでも texture prior だけを改善要因として断定するには足りない。結果は、現行 Model D の texture 経路に structured field を入れたときの小規模な差分記録として扱う。

## Limitations

- cross と1枚の public-domain 自然画像patchだけの小規模比較である。
- 現行実装の texture は draft 仕様の線形項そのものではなく、texture target 二乗項と初期状態混入を含む。
- 同じ decoder seed を使っているが、texture field が異なるため完全に同一の Markov chain 比較ではない。
- Global SSIM は依存なしの全画像SSIMであり、windowed SSIMではない。
- decode time はこの環境の小画像runに限る。
- この結果は意味的ディテール生成、super-resolution、compression の成立を示さない。

## Next

- confidence map や pairwise term の再設計候補は Issue #67 で扱う。
- structured texture を続ける場合は、より自然な texture 評価に向いた複数patchと、texture経路自体の式の見直しを別Issueで検討する。
"""


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
        "texture_sigma": TEXTURE_SIGMA,
        "texture_strength": TEXTURE_STRENGTH,
        "texture_configs": TEXTURE_CONFIGS,
        "model_d_params": MODEL_D_PARAMS,
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
