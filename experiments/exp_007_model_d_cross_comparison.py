"""Run a saved Model D cross comparison against simple upscaling baselines."""

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
from sidf_lab.confidence import gradient_confidence
from sidf_lab.guides import cross
from sidf_lab.io import ensure_dir, save_json
from sidf_lab.metrics import comparison_summary


RESULT_DIR = Path("results/2026-05-17-model-d-cross-comparison")
DATE = "2026-05-17"
EXPERIMENT_SEED = 20260517
DECODER_SEED = 6200
LOW_SIZE = 16
HIGH_SIZE = 64
LOW_WIDTH = 2
HIGH_WIDTH = 7


def timed_call(label: str, func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run a callable and return its result plus elapsed seconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, float(elapsed)


def build_notes(config: dict[str, Any], metrics: dict[str, dict[str, float | None]], timings: dict[str, float]) -> str:
    """Build top-level notes for the saved experiment."""
    rows = "\n".join(
        "| {name} | {mad:.6f} | {edge:.6f} | {width:.6f} | {fg:.6f} | {bg:.6f} | {seconds:.6f} |".format(
            name=name,
            mad=values["mad_vs_reference"],
            edge=values["edge_leakage"],
            width=values["edge_width_pixels"],
            fg=values["foreground_mean"],
            bg=values["background_mean"],
            seconds=timings[name],
        )
        for name, values in metrics.items()
    )
    return f"""# Model D Cross Baseline Comparison

## Question

Model D は cross の low-resolution guide から 64x64 output を生成するとき、nearest / bilinear / bicubic と比べて何を改善し、何を悪化させるか。

## Hypothesis

Model D は confidence map と edge-aware interaction により、bilinear より境界付近の拘束を強める可能性がある。一方で、white-noise texture term は synthetic reference とのMADや背景漏れを悪化させる可能性がある。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_007_model_d_cross_comparison.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Decoder seed: {config["decoder_seed"]}
- Low guide size: {config["low_size"]}x{config["low_size"]}
- Output size: {config["high_size"]}x{config["high_size"]}
- Shape: synthetic cross
- Model: Model D candidate
- Model config: `{config["model_d_params"]}`
- Decode config: `{config["decode_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

baselineは nearest、bilinear、bicubic upscaling。metricsのreferenceは同じsynthetic crossを64x64で生成した比較用参照であり、実画像のGround Truthではない。

## Result

| Output | MAD vs synthetic reference | Edge leakage | Edge width pixels | Foreground mean | Background mean | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Low guide image: `low_guide.png`
- Synthetic comparison reference: `high_reference.png`
- Baseline images: `nearest.png`, `bilinear.png`, `bicubic.png`
- Confidence map: `confidence.png`
- Model D rendered image: `rendered_model_d.png`
- Difference maps: `diff_model_d_vs_nearest.png`, `diff_model_d_vs_bilinear.png`, `diff_model_d_vs_bicubic.png`
- Comparison strip: `comparison.png`

## Images

![Comparison of reference, nearest, bilinear, bicubic, confidence, Model D, and difference](comparison.png)

![Low-resolution guide](low_guide.png)

![Confidence map](confidence.png)

![Model D rendered output](rendered_model_d.png)

![Absolute difference between Model D and bilinear](diff_model_d_vs_bilinear.png)

## Interpretation

この結果は synthetic cross 上の比較であり、Model D の一般的な超解像性能を示すものではない。Model D が単純補間より優れているかは、MAD、edge leakage、edge width、背景平均、差分画像を分けて読む必要がある。

今回のrunでは、Model D は MAD、edge leakage、background mean で nearest / bilinear / bicubic を改善しなかった。edge width は bilinear よりわずかに小さいが、MADと背景漏れの悪化を伴うため、総合的な改善とは解釈しない。

## Limitations

- 実画像のGround Truth比較ではない。
- synthetic cross は Model D に有利または不利な単純条件であり、自然画像の復元性能は評価していない。
- white-noise texture term は意味的ディテールではない。
- decode timeはこの環境の小画像runに限る。

## Next

- 自然画像Ground Truthでの評価は Issue #36 で扱う。
- texture term の寄与は Issue #37 で ablation として確認する。
- guided filter / guided upsampling との位置づけは Issue #14 で整理する。
"""


def main() -> None:
    ensure_dir(RESULT_DIR)
    high_reference = cross(size=HIGH_SIZE, width=HIGH_WIDTH, value=0.5)
    low_guide = cross(size=LOW_SIZE, width=LOW_WIDTH, value=0.5)
    foreground_mask = high_reference > 0.0

    nearest, nearest_seconds = timed_call("nearest", upscale, low_guide, HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call("bilinear", upscale, low_guide, HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call("bicubic", upscale, low_guide, HIGH_SIZE, 3)
    confidence, confidence_seconds = timed_call(
        "confidence",
        gradient_confidence,
        bilinear,
        min_confidence=0.2,
        max_confidence=1.0,
        scale=4.0,
    )
    texture = seeded_texture(bilinear.shape, seed=EXPERIMENT_SEED)
    model_d_params = {
        "j_base": 1.8,
        "lambda_data": 6.0,
        "gamma": 35.0,
        "texture_weight": 0.35,
    }
    decode_config = {
        "sweeps": 35,
        "temp_start": 0.35,
        "temp_end": 0.01,
        "proposal_sigma": 0.08,
    }
    rendered_model_d, model_d_seconds = timed_call(
        "model_d",
        model_d_decode,
        bilinear,
        confidence,
        texture,
        decoder_seed=DECODER_SEED,
        **decode_config,
        **model_d_params,
    )

    outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
        "model_d": rendered_model_d,
    }
    metrics = {
        name: comparison_summary(image, reference=high_reference, foreground_mask=foreground_mask)
        for name, image in outputs.items()
    }
    timings = {
        "nearest": nearest_seconds,
        "bilinear": bilinear_seconds,
        "bicubic": bicubic_seconds,
        "confidence": confidence_seconds,
        "model_d": model_d_seconds,
    }
    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "decoder_seed": DECODER_SEED,
        "low_size": LOW_SIZE,
        "high_size": HIGH_SIZE,
        "low_width": LOW_WIDTH,
        "high_width": HIGH_WIDTH,
        "shape": "synthetic cross",
        "model": "Model D candidate",
        "model_d_params": model_d_params,
        "decode_config": decode_config,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }

    diff_nearest = np.abs(rendered_model_d - nearest)
    diff_bilinear = np.abs(rendered_model_d - bilinear)
    diff_bicubic = np.abs(rendered_model_d - bicubic)

    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", {"outputs": metrics, "timings": timings})
    save_grayscale_png(RESULT_DIR / "low_guide.png", low_guide)
    save_grayscale_png(RESULT_DIR / "high_reference.png", high_reference)
    save_grayscale_png(RESULT_DIR / "nearest.png", nearest)
    save_grayscale_png(RESULT_DIR / "bilinear.png", bilinear)
    save_grayscale_png(RESULT_DIR / "bicubic.png", bicubic)
    save_grayscale_png(RESULT_DIR / "confidence.png", confidence)
    save_grayscale_png(RESULT_DIR / "rendered_model_d.png", rendered_model_d)
    save_grayscale_png(RESULT_DIR / "diff_model_d_vs_nearest.png", diff_nearest)
    save_grayscale_png(RESULT_DIR / "diff_model_d_vs_bilinear.png", diff_bilinear)
    save_grayscale_png(RESULT_DIR / "diff_model_d_vs_bicubic.png", diff_bicubic)
    save_comparison_png(
        RESULT_DIR / "comparison.png",
        [
            high_reference,
            nearest,
            bilinear,
            bicubic,
            confidence,
            rendered_model_d,
            diff_bilinear,
        ],
    )
    (RESULT_DIR / "notes.md").write_text(build_notes(config, metrics, timings), encoding="utf-8")

    for name, values in metrics.items():
        print(
            f"{name}: mad={values['mad_vs_reference']:.6f} "
            f"edge={values['edge_leakage']:.6f} "
            f"time={timings[name]:.6f}s"
        )


if __name__ == "__main__":
    main()
