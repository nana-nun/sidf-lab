"""Run Model D term-isolation comparisons."""

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

from experiments.exp_005_model_d_shape_benchmark import model_d_decode, save_comparison_png, upscale
from experiments.exp_008_model_d_natural_patch import (
    downscale_block_average,
    gradient_magnitude,
    natural_image_metrics,
)
from sidf_lab.confidence import gradient_confidence
from sidf_lab.guides import cross
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json
from sidf_lab.metrics import comparison_summary, mad


RESULT_DIR = Path("results/2026-05-24-issue-61-model-d-term-isolation")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-05-24"
EXPERIMENT_SEED = 20260524
CROSS_DECODER_SEED = 6600
NATURAL_DECODER_SEED = 6601
CROSS_LOW_SIZE = 16
CROSS_HIGH_SIZE = 64
CROSS_LOW_WIDTH = 2
CROSS_HIGH_WIDTH = 7
NATURAL_LOW_SIZE = 32
NATURAL_HIGH_SIZE = 128

TERM_CONFIGS = [
    {
        "id": "data_only_uniform",
        "label": "data fidelity only, uniform confidence",
        "lambda_data": 6.0,
        "j_base": 0.0,
        "confidence_mode": "uniform",
    },
    {
        "id": "data_only_conf",
        "label": "data fidelity only, gradient confidence",
        "lambda_data": 6.0,
        "j_base": 0.0,
        "confidence_mode": "gradient",
    },
    {
        "id": "pairwise_only",
        "label": "pairwise interaction only",
        "lambda_data": 0.0,
        "j_base": 1.8,
        "confidence_mode": "uniform",
    },
    {
        "id": "data_pairwise_uniform",
        "label": "data plus pairwise, uniform confidence",
        "lambda_data": 6.0,
        "j_base": 1.8,
        "confidence_mode": "uniform",
    },
    {
        "id": "data_pairwise_conf",
        "label": "data plus pairwise, gradient confidence",
        "lambda_data": 6.0,
        "j_base": 1.8,
        "confidence_mode": "gradient",
    },
]

COMMON_MODEL_PARAMS = {
    "gamma": 35.0,
    "texture_weight": 0.0,
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


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run a callable and return its result plus elapsed seconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def make_confidence(guide: np.ndarray, mode: str) -> np.ndarray:
    """Return either uniform or gradient-based confidence."""
    if mode == "uniform":
        return np.ones_like(guide, dtype=np.float64)
    if mode == "gradient":
        return gradient_confidence(guide, **CONFIDENCE_CONFIG)
    raise ValueError(f"unknown confidence mode: {mode}")


def decode_term_config(
    guide: np.ndarray,
    cfg: dict[str, Any],
    *,
    decoder_seed: int,
    decode_config: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decode one term-isolation condition."""
    confidence = make_confidence(guide, str(cfg["confidence_mode"]))
    zero_texture = np.zeros_like(guide, dtype=np.float64)
    rendered, seconds = timed_call(
        model_d_decode,
        guide,
        confidence,
        zero_texture,
        decoder_seed=decoder_seed,
        lambda_data=float(cfg["lambda_data"]),
        j_base=float(cfg["j_base"]),
        **COMMON_MODEL_PARAMS,
        **decode_config,
    )
    return rendered, confidence, seconds


def synthetic_metrics(candidate: np.ndarray, reference: np.ndarray, foreground_mask: np.ndarray) -> dict[str, float | None]:
    """Return synthetic-shape metrics with gradient MAD and bias."""
    values = comparison_summary(candidate, reference=reference, foreground_mask=foreground_mask)
    values["gradient_mad_vs_reference"] = mad(gradient_magnitude(candidate), gradient_magnitude(reference))
    values["mean_error_vs_reference"] = float((np.asarray(candidate) - np.asarray(reference)).mean())
    return values


def run_cross_case() -> dict[str, Any]:
    """Run term isolation on a synthetic cross."""
    case_dir = ensure_dir(RESULT_DIR / "cross")
    reference = cross(size=CROSS_HIGH_SIZE, width=CROSS_HIGH_WIDTH, value=0.5)
    low_guide = cross(size=CROSS_LOW_SIZE, width=CROSS_LOW_WIDTH, value=0.5)
    foreground_mask = reference > 0.0
    nearest, nearest_seconds = timed_call(upscale, low_guide, CROSS_HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, CROSS_HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, CROSS_HIGH_SIZE, 3)

    outputs = {"nearest": nearest, "bilinear": bilinear, "bicubic": bicubic}
    timings = {"nearest": nearest_seconds, "bilinear": bilinear_seconds, "bicubic": bicubic_seconds}
    confidences: dict[str, np.ndarray] = {}

    for cfg in TERM_CONFIGS:
        rendered, confidence, seconds = decode_term_config(
            bilinear,
            cfg,
            decoder_seed=CROSS_DECODER_SEED,
            decode_config=CROSS_DECODE_CONFIG,
        )
        name = str(cfg["id"])
        outputs[name] = rendered
        timings[name] = seconds
        confidences[name] = confidence

    metrics = {name: synthetic_metrics(image, reference, foreground_mask) for name, image in outputs.items()}
    save_case_artifacts(
        case_dir,
        outputs,
        reference=reference,
        low_guide=low_guide,
        baseline_for_diff=bilinear,
        confidences=confidences,
        comparison_names=["nearest", "bilinear", "bicubic", *[str(cfg["id"]) for cfg in TERM_CONFIGS]],
    )
    return {"metrics": metrics, "timings": timings}


def run_natural_case() -> dict[str, Any]:
    """Run term isolation on the saved public-domain natural patch."""
    case_dir = ensure_dir(RESULT_DIR / "natural_patch")
    reference = np.load(SOURCE_ASSET)
    low_guide = downscale_block_average(reference, NATURAL_LOW_SIZE)
    nearest, nearest_seconds = timed_call(upscale, low_guide, NATURAL_HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, NATURAL_HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, NATURAL_HIGH_SIZE, 3)

    outputs = {"nearest": nearest, "bilinear": bilinear, "bicubic": bicubic}
    timings = {"nearest": nearest_seconds, "bilinear": bilinear_seconds, "bicubic": bicubic_seconds}
    confidences: dict[str, np.ndarray] = {}

    for cfg in TERM_CONFIGS:
        rendered, confidence, seconds = decode_term_config(
            bilinear,
            cfg,
            decoder_seed=NATURAL_DECODER_SEED,
            decode_config=NATURAL_DECODE_CONFIG,
        )
        name = str(cfg["id"])
        outputs[name] = rendered
        timings[name] = seconds
        confidences[name] = confidence

    metrics = {name: natural_image_metrics(image, reference) for name, image in outputs.items()}
    save_case_artifacts(
        case_dir,
        outputs,
        reference=reference,
        low_guide=low_guide,
        baseline_for_diff=bilinear,
        confidences=confidences,
        comparison_names=["nearest", "bilinear", "bicubic", *[str(cfg["id"]) for cfg in TERM_CONFIGS]],
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
    """Save PNG artifacts for one case."""
    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    for name, image in outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
        if name not in {"nearest", "bilinear", "bicubic"}:
            save_grayscale_png(case_dir / f"diff_{name}_vs_bilinear.png", np.abs(image - baseline_for_diff))
            save_grayscale_png(case_dir / f"diff_{name}_vs_reference.png", np.abs(image - reference))
    for name, confidence in confidences.items():
        save_grayscale_png(case_dir / f"confidence_{name}.png", confidence)
    guide_for_strip = low_guide if low_guide.shape == reference.shape else upscale(low_guide, reference.shape[0], 0)
    save_comparison_png(case_dir / "comparison.png", [reference, guide_for_strip, *[outputs[name] for name in comparison_names]])


def best_term_by(metric_map: dict[str, dict[str, float | None]], metric_name: str) -> str:
    """Return the term config id with the smallest metric value."""
    best_name = ""
    best_value = float("inf")
    for cfg in TERM_CONFIGS:
        name = str(cfg["id"])
        value = metric_map[name][metric_name]
        if value is not None and float(value) < best_value:
            best_name = name
            best_value = float(value)
    return best_name


def format_cross_rows(metrics: dict[str, dict[str, float | None]], timings: dict[str, float]) -> str:
    """Format the synthetic cross metric table."""
    names = ["nearest", "bilinear", "bicubic", *[str(cfg["id"]) for cfg in TERM_CONFIGS]]
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
    """Format the natural patch metric table."""
    names = ["nearest", "bilinear", "bicubic", *[str(cfg["id"]) for cfg in TERM_CONFIGS]]
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
    """Build top-level Japanese experiment notes."""
    cross_rows = format_cross_rows(results["cross"]["metrics"], results["cross"]["timings"])
    natural_rows = format_natural_rows(results["natural_patch"]["metrics"], results["natural_patch"]["timings"])
    cross_best = best_term_by(results["cross"]["metrics"], "mad_vs_reference")
    natural_best = best_term_by(results["natural_patch"]["metrics"], "mad_vs_gt")
    return f"""# Model D Term Isolation

## Question

Model D candidate の data fidelity、pairwise interaction、confidence map を項ごとに分けると、nearest / bilinear / bicubic baseline より悪化している要因をより直接的に確認できるか。

## Hypothesis

Issue #56 では `flat_conf_tex0` がgrid内で相対的に良かったが、単純補間を上回らなかった。このため、gradient confidence の空間変化、pairwise interaction、data fidelity のどれが悪化に寄与しているかを、texture_strength=0 固定の対照実験で分離する必要がある。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_011_model_d_term_isolation.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Cross decoder seed: {config["cross_decoder_seed"]}
- Natural patch decoder seed: {config["natural_decoder_seed"]}
- Texture strength: 0.0 for all Model D term conditions
- Cross: {config["cross_low_size"]}x{config["cross_low_size"]} guide to {config["cross_high_size"]}x{config["cross_high_size"]} output
- Natural patch: {config["natural_low_size"]}x{config["natural_low_size"]} guide to {config["natural_high_size"]}x{config["natural_high_size"]} output
- Term configs: `config.json` の `term_configs`
- Common model params: `{config["common_model_params"]}`
- Cross decode config: `{config["cross_decode_config"]}`
- Natural decode config: `{config["natural_decode_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

baselineは cross / natural patch の両方で nearest、bilinear、bicubic upscaling とした。Model D term conditions はすべて `texture_strength=0` とし、data fidelity、pairwise interaction、confidence weighting の有無を分けた。

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
- Each case includes baseline PNGs, term-condition rendered PNGs, confidence maps, difference maps, and `comparison.png`.

## Images

![Cross term isolation comparison](cross/comparison.png)

![Natural patch term isolation comparison](natural_patch/comparison.png)

## Result

Cross の term conditions 内では `{cross_best}` が最小MADだった。Natural patch の term conditions 内では `{natural_best}` が最小MADだった。

## Interpretation

このrunでは、data fidelity only、pairwise only、data+pairwise、confidence-weighted data のいずれも nearest / bilinear / bicubic baseline を総合的に上回ったとは解釈しない。特に `data_only_uniform` や `data_pairwise_uniform` が既存の `data_pairwise_conf` より良い場合でも、それは Model D がbaselineを改善したことではなく、現行gradient confidenceの空間重み付けがこの設定では有利に働いていない可能性を示す切り分け結果である。

pairwise-only 条件は、guideへのdata fidelityを持たないため、画像復元条件としては不十分である。これはpairwise interaction単体の挙動を見るための対照条件であり、候補モデルとして採用する条件ではない。

## Limitations

- crossと1枚のpublic-domain自然画像patchだけの結果である。
- term isolation は現行 `model_d_decode` のパラメータを使った対照実験であり、別形式のdecoder objectiveを実装したものではない。
- Global SSIM は依存なしの全画像SSIMであり、windowed SSIMではない。
- decode timeはこの環境の小画像runに限る。
- この結果はsuper-resolutionやcompressionの成立を示さず、またそれらを一般に否定する結果でもない。

## Next

- 現行Model D式の単純な重み探索はいったん止め、confidence map の作り方または pairwise term の設計を別候補として再設計する。Follow-up: Issue #67。
- structured texture prior を評価する場合も、今回のような term-isolated baseline と white-noise baseline を含める。
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
        "term_configs": TERM_CONFIGS,
        "common_model_params": COMMON_MODEL_PARAMS,
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
        print(case_name)
        metric_name = "mad_vs_reference" if case_name == "cross" else "mad_vs_gt"
        for name, values in case_results["metrics"].items():
            print(f"  {name}: {metric_name}={values[metric_name]:.6f} time={case_results['timings'][name]:.6f}s")


if __name__ == "__main__":
    main()
