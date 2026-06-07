"""Compare Model D with low-guide-only edge-aware filtering baselines."""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

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


RESULT_DIR = Path("results/2026-06-07-issue-74-guided-filter-baselines")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-06-07"
EXPERIMENT_SEED = 20260607
CROSS_DECODER_SEED = 6740
NATURAL_DECODER_SEED = 6741
CROSS_LOW_SIZE = 16
CROSS_HIGH_SIZE = 64
CROSS_LOW_WIDTH = 2
CROSS_HIGH_WIDTH = 7
NATURAL_LOW_SIZE = 32
NATURAL_HIGH_SIZE = 128

EDGE_AWARE_CONFIGS = {
    "guided_filter": {
        "description": "self-guided filter on the bilinear-upscaled low guide",
        "radius": 3,
        "epsilon": 0.01**2,
        "high_resolution_guidance": False,
    },
    "joint_bilateral": {
        "description": "nearest-upscaled values refined with the bilinear-upscaled low guide",
        "radius": 3,
        "sigma_spatial": 2.0,
        "sigma_range": 0.08,
        "high_resolution_guidance": False,
    },
    "bilateral_smoothing": {
        "description": "bilateral smoothing of the bilinear-upscaled low guide",
        "radius": 3,
        "sigma_spatial": 2.0,
        "sigma_range": 0.08,
        "high_resolution_guidance": False,
    },
}
MODEL_D_PARAMS = {
    "j_base": 1.8,
    "lambda_data": 6.0,
    "gamma": 35.0,
    "texture_weight": 0.35,
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
CONFIDENCE_CONFIG = {
    "min_confidence": 0.2,
    "max_confidence": 1.0,
    "scale": 4.0,
}
OUTPUT_NAMES = [
    "nearest",
    "bilinear",
    "bicubic",
    "guided_filter",
    "joint_bilateral",
    "bilateral_smoothing",
    "model_d",
]


def timed_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run a callable and return its result plus elapsed seconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def guided_filter(
    guidance: np.ndarray,
    source: np.ndarray,
    *,
    radius: int,
    epsilon: float,
) -> np.ndarray:
    """Apply the grayscale guided filter using a square local window."""
    if radius < 1:
        raise ValueError("radius must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    guide = np.asarray(guidance, dtype=np.float64)
    values = np.asarray(source, dtype=np.float64)
    if guide.shape != values.shape:
        raise ValueError("guidance and source must have the same shape")

    mean_guide = box_mean(guide, radius)
    mean_values = box_mean(values, radius)
    correlation_guide = box_mean(guide * guide, radius)
    correlation_cross = box_mean(guide * values, radius)
    variance_guide = correlation_guide - mean_guide * mean_guide
    covariance = correlation_cross - mean_guide * mean_values
    coefficient_a = covariance / (variance_guide + epsilon)
    coefficient_b = mean_values - coefficient_a * mean_guide
    mean_a = box_mean(coefficient_a, radius)
    mean_b = box_mean(coefficient_b, radius)
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


def box_mean(image: np.ndarray, radius: int) -> np.ndarray:
    """Return a reflect-padded square-window mean using an integral image."""
    values = np.asarray(image, dtype=np.float64)
    size = 2 * radius + 1
    padded = np.pad(values, radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    window_sum = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return window_sum / float(size * size)


def joint_bilateral_filter(
    source: np.ndarray,
    guidance: np.ndarray,
    *,
    radius: int,
    sigma_spatial: float,
    sigma_range: float,
) -> np.ndarray:
    """Apply a deterministic grayscale joint bilateral filter."""
    if radius < 1:
        raise ValueError("radius must be positive")
    if sigma_spatial <= 0.0 or sigma_range <= 0.0:
        raise ValueError("bilateral sigmas must be positive")
    values = np.asarray(source, dtype=np.float64)
    guide = np.asarray(guidance, dtype=np.float64)
    if values.shape != guide.shape:
        raise ValueError("source and guidance must have the same shape")

    padded_values = np.pad(values, radius, mode="reflect")
    padded_guide = np.pad(guide, radius, mode="reflect")
    height, width = values.shape
    weighted_sum = np.zeros_like(values)
    weight_sum = np.zeros_like(values)
    center_guide = guide
    spatial_denominator = 2.0 * sigma_spatial * sigma_spatial
    range_denominator = 2.0 * sigma_range * sigma_range

    for dy in range(-radius, radius + 1):
        row_start = radius + dy
        for dx in range(-radius, radius + 1):
            col_start = radius + dx
            shifted_values = padded_values[row_start : row_start + height, col_start : col_start + width]
            shifted_guide = padded_guide[row_start : row_start + height, col_start : col_start + width]
            spatial_weight = np.exp(-(dy * dy + dx * dx) / spatial_denominator)
            range_weight = np.exp(-((shifted_guide - center_guide) ** 2) / range_denominator)
            weight = spatial_weight * range_weight
            weighted_sum += weight * shifted_values
            weight_sum += weight
    return np.clip(weighted_sum / weight_sum, 0.0, 1.0)


def synthetic_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    foreground_mask: np.ndarray,
) -> dict[str, float | None]:
    """Return synthetic cross metrics with gradient error and bias."""
    values = comparison_summary(candidate, reference=reference, foreground_mask=foreground_mask)
    values["gradient_mad_vs_reference"] = mad(gradient_magnitude(candidate), gradient_magnitude(reference))
    values["mean_error_vs_reference"] = float((np.asarray(candidate) - np.asarray(reference)).mean())
    return values


def make_outputs(
    low_guide: np.ndarray,
    high_size: int,
    *,
    decoder_seed: int,
    decode_config: dict[str, float | int],
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray]:
    """Build interpolation, edge-aware, and Model D outputs."""
    nearest, nearest_seconds = timed_call(upscale, low_guide, high_size, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, high_size, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, high_size, 3)

    guided_cfg = EDGE_AWARE_CONFIGS["guided_filter"]
    guided, guided_seconds = timed_call(
        guided_filter,
        bilinear,
        bilinear,
        radius=int(guided_cfg["radius"]),
        epsilon=float(guided_cfg["epsilon"]),
    )
    bilateral_cfg = EDGE_AWARE_CONFIGS["joint_bilateral"]
    joint_bilateral, joint_seconds = timed_call(
        joint_bilateral_filter,
        nearest,
        bilinear,
        radius=int(bilateral_cfg["radius"]),
        sigma_spatial=float(bilateral_cfg["sigma_spatial"]),
        sigma_range=float(bilateral_cfg["sigma_range"]),
    )
    smoothing_cfg = EDGE_AWARE_CONFIGS["bilateral_smoothing"]
    bilateral_smoothing, smoothing_seconds = timed_call(
        joint_bilateral_filter,
        bilinear,
        bilinear,
        radius=int(smoothing_cfg["radius"]),
        sigma_spatial=float(smoothing_cfg["sigma_spatial"]),
        sigma_range=float(smoothing_cfg["sigma_range"]),
    )

    confidence = gradient_confidence(bilinear, **CONFIDENCE_CONFIG)
    texture = seeded_texture(bilinear.shape, seed=EXPERIMENT_SEED)
    model_d, model_d_seconds = timed_call(
        model_d_decode,
        bilinear,
        confidence,
        texture,
        decoder_seed=decoder_seed,
        **MODEL_D_PARAMS,
        **decode_config,
    )
    outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
        "guided_filter": guided,
        "joint_bilateral": joint_bilateral,
        "bilateral_smoothing": bilateral_smoothing,
        "model_d": model_d,
    }
    timings = {
        "nearest": nearest_seconds,
        "bilinear": bilinear_seconds,
        "bicubic": bicubic_seconds,
        "guided_filter": guided_seconds,
        "joint_bilateral": joint_seconds,
        "bilateral_smoothing": smoothing_seconds,
        "model_d": model_d_seconds,
    }
    return outputs, timings, confidence


def save_case_artifacts(
    case_dir: Path,
    outputs: dict[str, np.ndarray],
    *,
    reference: np.ndarray,
    low_guide: np.ndarray,
    confidence: np.ndarray,
) -> None:
    """Save the main PNGs and per-method reference differences."""
    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    save_grayscale_png(case_dir / "confidence.png", confidence)
    for name, image in outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
        save_grayscale_png(case_dir / f"diff_{name}_vs_reference.png", np.abs(image - reference))
    guide_preview = upscale(low_guide, reference.shape[0], 0)
    save_comparison_png(
        case_dir / "comparison.png",
        [reference, guide_preview, *[outputs[name] for name in OUTPUT_NAMES]],
    )
    save_comparison_png(
        case_dir / "difference_comparison.png",
        [np.abs(outputs[name] - reference) for name in OUTPUT_NAMES],
    )


def run_cross_case() -> dict[str, Any]:
    """Run all baselines on the synthetic cross."""
    case_dir = ensure_dir(RESULT_DIR / "cross")
    reference = cross(size=CROSS_HIGH_SIZE, width=CROSS_HIGH_WIDTH, value=0.5)
    low_guide = cross(size=CROSS_LOW_SIZE, width=CROSS_LOW_WIDTH, value=0.5)
    outputs, timings, confidence = make_outputs(
        low_guide,
        CROSS_HIGH_SIZE,
        decoder_seed=CROSS_DECODER_SEED,
        decode_config=CROSS_DECODE_CONFIG,
    )
    foreground_mask = reference > 0.0
    metrics = {
        name: synthetic_metrics(image, reference, foreground_mask)
        for name, image in outputs.items()
    }
    save_case_artifacts(
        case_dir,
        outputs,
        reference=reference,
        low_guide=low_guide,
        confidence=confidence,
    )
    return {"metrics": metrics, "timings": timings}


def run_natural_case() -> dict[str, Any]:
    """Run all baselines on the public-domain natural patch."""
    case_dir = ensure_dir(RESULT_DIR / "natural_patch")
    reference = np.load(SOURCE_ASSET)
    low_guide = downscale_block_average(reference, NATURAL_LOW_SIZE)
    outputs, timings, confidence = make_outputs(
        low_guide,
        NATURAL_HIGH_SIZE,
        decoder_seed=NATURAL_DECODER_SEED,
        decode_config=NATURAL_DECODE_CONFIG,
    )
    metrics = {name: natural_image_metrics(image, reference) for name, image in outputs.items()}
    save_case_artifacts(
        case_dir,
        outputs,
        reference=reference,
        low_guide=low_guide,
        confidence=confidence,
    )
    return {"metrics": metrics, "timings": timings}


def format_cross_rows(metrics: dict[str, dict[str, float | None]], timings: dict[str, float]) -> str:
    """Format the synthetic cross metric table."""
    return "\n".join(
        "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {grad:.6f} | {edge:.6f} | {seconds:.6f} |".format(
            name=name,
            mad=metrics[name]["mad_vs_reference"],
            psnr=metrics[name]["psnr_vs_reference"],
            ssim=metrics[name]["ssim_global_vs_reference"],
            grad=metrics[name]["gradient_mad_vs_reference"],
            edge=metrics[name]["edge_leakage"],
            seconds=timings[name],
        )
        for name in OUTPUT_NAMES
    )


def format_natural_rows(metrics: dict[str, dict[str, float]], timings: dict[str, float]) -> str:
    """Format the natural patch metric table."""
    return "\n".join(
        "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {grad:.6f} | {edge:.6f} | {seconds:.6f} |".format(
            name=name,
            mad=metrics[name]["mad_vs_gt"],
            psnr=metrics[name]["psnr_vs_gt"],
            ssim=metrics[name]["ssim_global_vs_gt"],
            grad=metrics[name]["gradient_mad_vs_gt"],
            edge=metrics[name]["strong_edge_mad_vs_gt"],
            seconds=timings[name],
        )
        for name in OUTPUT_NAMES
    )


def best_by(metrics: dict[str, dict[str, float | None]], metric_name: str) -> str:
    """Return the output name with the smallest non-null metric."""
    return min(
        OUTPUT_NAMES,
        key=lambda name: float(metrics[name][metric_name])
        if metrics[name][metric_name] is not None
        else float("inf"),
    )


def build_notes(config: dict[str, Any], results: dict[str, Any]) -> str:
    """Build the Japanese experiment note."""
    cross_rows = format_cross_rows(results["cross"]["metrics"], results["cross"]["timings"])
    natural_rows = format_natural_rows(
        results["natural_patch"]["metrics"],
        results["natural_patch"]["timings"],
    )
    cross_best = best_by(results["cross"]["metrics"], "mad_vs_reference")
    natural_best = best_by(results["natural_patch"]["metrics"], "mad_vs_gt")
    return f"""# Guided Filter系BaselineとModel Dの比較

## Question

低解像度guideだけから作るedge-aware filtering baselineは、nearest / bilinear / bicubicおよび現行Model D candidateと比べて、crossと自然画像patchの再構成指標を改善するか。

## Hypothesis

guided filter、joint bilateral refinement、bilateral smoothingは、単純補間より局所的なedge-aware処理を行う。ただし独立した高解像度guidanceを使わないため、低解像度guideで失われた詳細を復元するとは仮定しない。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_015_guided_filter_baselines.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Cross decoder seed: {config["cross_decoder_seed"]}
- Natural patch decoder seed: {config["natural_decoder_seed"]}
- Cross: {config["cross_low_size"]}x{config["cross_low_size"]} guide to {config["cross_high_size"]}x{config["cross_high_size"]} output
- Natural patch: {config["natural_low_size"]}x{config["natural_low_size"]} guide to {config["natural_high_size"]}x{config["natural_high_size"]} output
- Edge-aware configs: `config.json` の `edge_aware_configs`
- Model D params: `{config["model_d_params"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

nearest、bilinear、bicubicに加え、次のedge-aware baselineを比較した。

- `guided_filter`: bilinear-upscaled low guideを入力と自己guidanceの両方に使う。
- `joint_bilateral`: nearest-upscaled値を、bilinear-upscaled low guideで重み付けしてrefineする。
- `bilateral_smoothing`: bilinear-upscaled low guideを入力とguidanceの両方に使う。

いずれも独立した高解像度guidance imageは使わない。したがって、一般的なjoint bilateral upsamplingで高解像度RGB guidanceを利用する条件とは分けて扱う。

## Metrics

### Cross

| Output | MAD vs reference | PSNR | SSIM | Gradient MAD | Edge leakage | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{cross_rows}

### Natural Patch

| Output | MAD vs GT | PSNR | SSIM | Gradient MAD | Strong-edge MAD | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{natural_rows}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Cross artifacts: `cross/`
- Natural patch artifacts: `natural_patch/`
- 各caseに入力、全出力、referenceとの差分、confidence map、比較PNGを保存した。

## Images

![Cross comparison](cross/comparison.png)

![Cross difference comparison](cross/difference_comparison.png)

![Natural patch comparison](natural_patch/comparison.png)

![Natural patch difference comparison](natural_patch/difference_comparison.png)

## Result

CrossでMADが最小だったのは `{cross_best}`、natural patchでMADが最小だったのは `{natural_best}` だった。

## Interpretation

Crossでは `joint_bilateral` がedge-aware条件内の最小MAD `0.019354` だったが、nearestの `0.013794` には届かなかった。Natural patchでは `joint_bilateral` のMAD `0.043522` とgradient MAD `0.073171` はbilinearの `0.044369` と `0.094319` より小さかったが、MADではbicubicの `0.042397` が最小だった。

現行Model Dはcross MAD `0.047228`、natural patch MAD `0.057154` で、このrunの補間およびedge-aware baselineを上回らなかった。これは低解像度guideだけから構成したbaselineとの最小比較であり、高解像度guidanceを持つ既存手法全般の性能を示すものではない。またModel Dまたはedge-aware baselineの優劣は、crossと1枚のnatural patchで観測された範囲に限る。

## Limitations

- crossと1枚のpublic-domain自然画像patchだけの比較である。
- guided filterとbilateral系のパラメータは小規模な固定値であり、網羅的探索はしていない。
- すべてのedge-aware baselineはupscaled low guideからguidanceを作る。独立した高解像度guidanceを使う標準的なjoint upsampling条件とは異なる。
- Global SSIMは依存なしの全画像SSIMであり、windowed SSIMではない。
- decode timeはこの環境の小画像runに限る。
- この結果はsuper-resolutionやcompressionの成立を示さない。

## Next

- confidence mapとpairwise termの再設計候補はIssue #67で、今回のedge-aware baselineを比較対象として利用できる。
- 独立した高解像度guidanceを使う比較が必要になった場合は、SIDFのlow-guide-only条件と別条件として明示する。
"""


def main() -> None:
    """Run the experiment and save artifacts."""
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
        "guidance_policy": "No independent high-resolution guidance; all guidance derives from the low guide.",
        "edge_aware_configs": EDGE_AWARE_CONFIGS,
        "model_d_params": MODEL_D_PARAMS,
        "cross_decode_config": CROSS_DECODE_CONFIG,
        "natural_decode_config": NATURAL_DECODE_CONFIG,
        "confidence_config": CONFIDENCE_CONFIG,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", results)
    (RESULT_DIR / "notes.md").write_text(build_notes(config, results), encoding="utf-8")

    for case_name, case_results in results.items():
        metric_name = "mad_vs_reference" if case_name == "cross" else "mad_vs_gt"
        print(case_name)
        for name in OUTPUT_NAMES:
            values = case_results["metrics"][name]
            print(
                f"  {name}: {metric_name}={values[metric_name]:.6f} "
                f"time={case_results['timings'][name]:.6f}s"
            )


if __name__ == "__main__":
    main()
