"""Run Model C/D decode-time scaling benchmarks over synthetic guides."""

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
    save_grayscale_png,
    seeded_texture,
    upscale,
)
from sidf_lab.anneal import AnnealConfig, model_c_decode
from sidf_lab.confidence import gradient_confidence
from sidf_lab.energy import ModelCParams
from sidf_lab.guides import cross
from sidf_lab.io import ensure_dir, save_json
from sidf_lab.metrics import comparison_summary


RESULT_DIR = Path("results/2026-05-17-decode-time-scaling")
DATE = "2026-05-17"
EXPERIMENT_SEED = 20260517
DECODER_SEED_BASE = 6100
SIZES = [32, 64, 128, 256]
SWEEPS = 12


def timed_call(label: str, func: Any, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, float | str]]:
    """Run a callable and return its result with elapsed time metadata."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, {"label": label, "seconds": float(elapsed)}


def json_safe(value: Any) -> Any:
    """Return a JSON-safe copy with non-finite floats replaced by null."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def foreground_mask_for_cross(size: int, width: int) -> np.ndarray:
    """Return the synthetic cross foreground mask used by the benchmark."""
    return cross(size=size, width=width, value=1.0) > 0.0


def benchmark_size(size: int, index: int, model_c_params: ModelCParams, model_d_params: dict[str, float]) -> dict[str, Any]:
    """Run Model C, Model D, and baseline timing for one output size."""
    size_dir = ensure_dir(RESULT_DIR / f"{size}x{size}")
    high_width = max(2, size // 8)
    low_size = max(8, size // 4)
    low_width = max(1, low_size // 8)

    high_reference = cross(size=size, width=high_width, value=0.5)
    low_guide = cross(size=low_size, width=low_width, value=0.5)
    mask = foreground_mask_for_cross(size=size, width=high_width)

    nearest, nearest_time = timed_call("nearest", upscale, low_guide, size, 0)
    bilinear, bilinear_time = timed_call("bilinear", upscale, low_guide, size, 1)
    bicubic, bicubic_time = timed_call("bicubic", upscale, low_guide, size, 3)
    confidence, confidence_time = timed_call(
        "confidence",
        gradient_confidence,
        bilinear,
        min_confidence=0.2,
        max_confidence=1.0,
        scale=4.0,
    )
    texture = seeded_texture(bilinear.shape, seed=EXPERIMENT_SEED + index)

    model_c_config = AnnealConfig(
        decoder_seed=DECODER_SEED_BASE + index,
        sweeps=SWEEPS,
        temp_start=0.35,
        temp_end=0.01,
        proposal_sigma=0.08,
    )
    model_c_rendered, model_c_time = timed_call(
        "model_c_decode",
        model_c_decode,
        high_reference,
        model_c_params,
        model_c_config,
    )
    model_d_rendered, model_d_time = timed_call(
        "model_d_decode",
        model_d_decode,
        bilinear,
        confidence,
        texture,
        decoder_seed=DECODER_SEED_BASE + 100 + index,
        sweeps=SWEEPS,
        temp_start=0.35,
        temp_end=0.01,
        proposal_sigma=0.08,
        **model_d_params,
    )

    outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
        "model_c": model_c_rendered,
        "model_d": model_d_rendered,
    }
    metrics = {
        name: comparison_summary(image, reference=high_reference, foreground_mask=mask)
        for name, image in outputs.items()
    }
    timings = {
        "nearest_seconds": nearest_time["seconds"],
        "bilinear_seconds": bilinear_time["seconds"],
        "bicubic_seconds": bicubic_time["seconds"],
        "confidence_seconds": confidence_time["seconds"],
        "model_c_decode_seconds": model_c_time["seconds"],
        "model_d_decode_seconds": model_d_time["seconds"],
    }

    diff_model_d_vs_bilinear = np.abs(model_d_rendered - bilinear)
    save_json(
        size_dir / "metrics.json",
        json_safe({
            "size": size,
            "low_size": low_size,
            "sweeps": SWEEPS,
            "timings": timings,
            "metrics": metrics,
        }),
    )
    save_grayscale_png(size_dir / "high_reference.png", high_reference)
    save_grayscale_png(size_dir / "low_guide.png", low_guide)
    save_grayscale_png(size_dir / "nearest.png", nearest)
    save_grayscale_png(size_dir / "bilinear.png", bilinear)
    save_grayscale_png(size_dir / "bicubic.png", bicubic)
    save_grayscale_png(size_dir / "confidence.png", confidence)
    save_grayscale_png(size_dir / "rendered_model_c.png", model_c_rendered)
    save_grayscale_png(size_dir / "rendered_model_d.png", model_d_rendered)
    save_grayscale_png(size_dir / "diff_model_d_vs_bilinear.png", diff_model_d_vs_bilinear)
    save_comparison_png(
        size_dir / "comparison.png",
        [
            high_reference,
            nearest,
            bilinear,
            bicubic,
            confidence,
            model_c_rendered,
            model_d_rendered,
            diff_model_d_vs_bilinear,
        ],
    )

    return {
        "size": size,
        "low_size": low_size,
        "sweeps": SWEEPS,
        "timings": timings,
        "metrics": metrics,
    }


def make_notes(config: dict[str, Any], results: list[dict[str, Any]]) -> str:
    """Build the top-level experiment notes."""
    rows = "\n".join(
        "| {size} | {nearest:.6f} | {bilinear:.6f} | {bicubic:.6f} | {model_c:.6f} | {model_d:.6f} |".format(
            size=f"{row['size']}x{row['size']}",
            nearest=row["timings"]["nearest_seconds"],
            bilinear=row["timings"]["bilinear_seconds"],
            bicubic=row["timings"]["bicubic_seconds"],
            model_c=row["timings"]["model_c_decode_seconds"],
            model_d=row["timings"]["model_d_decode_seconds"],
        )
        for row in results
    )
    metric_rows = "\n".join(
        "| {size} | {model_c_mad:.6f} | {model_d_mad:.6f} | {model_c_edge:.6f} | {model_d_edge:.6f} |".format(
            size=f"{row['size']}x{row['size']}",
            model_c_mad=row["metrics"]["model_c"]["mad_vs_reference"],
            model_d_mad=row["metrics"]["model_d"]["mad_vs_reference"],
            model_c_edge=row["metrics"]["model_c"]["edge_leakage"],
            model_d_edge=row["metrics"]["model_d"]["edge_leakage"],
        )
        for row in results
    )
    return f"""# Model C/D Decode Time Scaling Benchmark

## Question

Model C/Model D の Python decode time は、画像サイズに対してどの程度伸びるか。

## Hypothesis

現行のMetropolis型updateは各sweepで全画素を走査するため、decode timeはおおむね画素数とsweep数に比例して増える。32x32や64x64の小画像結果だけでは、128x128以上の実用性は判断できない。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_006_decode_time_scaling.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Decoder seed base: {config["decoder_seed_base"]}
- Output sizes: {config["sizes"]}
- Sweeps: {config["sweeps"]}
- Shape: synthetic cross
- Model C config: `{config["model_c_params"]}`
- Model D config: `{config["model_d_params"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

baseline timingはnearest、bilinear、bicubic upscalingとした。Model Dはbilinear guide、gradient confidence、seeded texture termを使う。すべて同じsynthetic high-resolution crossをcomparison referenceとしてmetricsを計算する。

## Result

| Size | Nearest seconds | Bilinear seconds | Bicubic seconds | Model C decode seconds | Model D decode seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

| Size | Model C MAD | Model D MAD | Model C edge leakage | Model D edge leakage |
| --- | ---: | ---: | ---: | ---: |
{metric_rows}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Per-size metrics: `<size>x<size>/metrics.json`
- Per-size PNGs: `high_reference.png`, `low_guide.png`, `nearest.png`, `bilinear.png`, `bicubic.png`, `confidence.png`, `rendered_model_c.png`, `rendered_model_d.png`, `diff_model_d_vs_bilinear.png`, `comparison.png`

## Images

各sizeの `comparison.png` に、reference、baselines、confidence、Model C、Model D、differenceを横並びで保存した。

![32x32 comparison](32x32/comparison.png)

![64x64 comparison](64x64/comparison.png)

![128x128 comparison](128x128/comparison.png)

![256x256 comparison](256x256/comparison.png)

## Interpretation

decode timeはこのPython実装と実行環境に依存する。今回の結果は、現時点の制限を測るためのbaselineであり、実用圧縮形式としての性能を示すものではない。

## Limitations

- synthetic crossのみで、自然画像や複雑なtextureでは未確認。
- sweepsを12に固定したため、過去の80 sweeps実験と品質を直接比較しない。
- 256x256は短時間benchmarkとして実行しただけで、収束性や品質の十分性は評価していない。
- Rust実装、固定小数点、並列化、より効率的なupdate scheduleは未評価。

## Next

- 収束品質とdecode timeのtrade-offをsweep数別に見る。
- Model Dのtexture term ablationをIssue #37で確認する。
- 自然画像Ground Truthでの評価をIssue #36で扱う。
"""


def main() -> None:
    ensure_dir(RESULT_DIR)
    model_c_params = ModelCParams(j_base=2.0, lambda_data=5.0, gamma=40.0)
    model_d_params = {
        "j_base": 1.8,
        "lambda_data": 6.0,
        "gamma": 35.0,
        "texture_weight": 0.35,
    }
    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "decoder_seed_base": DECODER_SEED_BASE,
        "sizes": SIZES,
        "sweeps": SWEEPS,
        "shape": "synthetic cross",
        "model_c_params": {
            "j_base": model_c_params.j_base,
            "lambda_data": model_c_params.lambda_data,
            "gamma": model_c_params.gamma,
        },
        "model_d_params": model_d_params,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    results = [
        benchmark_size(size, index, model_c_params, model_d_params)
        for index, size in enumerate(SIZES)
    ]
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", json_safe({"sizes": results}))
    (RESULT_DIR / "notes.md").write_text(make_notes(config, results), encoding="utf-8")
    for row in results:
        print(
            f"{row['size']}x{row['size']}: "
            f"model_c={row['timings']['model_c_decode_seconds']:.3f}s "
            f"model_d={row['timings']['model_d_decode_seconds']:.3f}s "
            f"bilinear={row['timings']['bilinear_seconds']:.6f}s"
        )


if __name__ == "__main__":
    main()
