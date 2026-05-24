"""Measure Model C/D quality and decode time across sweep counts."""

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


RESULT_DIR = Path("results/2026-05-24-decode-sweep-quality")
DATE = "2026-05-24"
EXPERIMENT_SEED = 20260524
DECODER_SEED_BASE = 6500
SIZES = [64, 128]
SWEEPS = [1, 4, 12, 24]


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
    """Return the synthetic cross foreground mask used by this benchmark."""
    return cross(size=size, width=width, value=1.0) > 0.0


def baseline_outputs(size: int) -> dict[str, Any]:
    """Build the reference, low guide, baselines, confidence map, and texture."""
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
    texture = seeded_texture(bilinear.shape, seed=EXPERIMENT_SEED + size)

    outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
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
    }
    return {
        "size": size,
        "low_size": low_size,
        "high_reference": high_reference,
        "low_guide": low_guide,
        "foreground_mask": mask,
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
        "confidence": confidence,
        "texture": texture,
        "baseline_metrics": metrics,
        "baseline_timings": timings,
    }


def run_size(
    size: int,
    index: int,
    model_c_params: ModelCParams,
    model_d_params: dict[str, float],
) -> dict[str, Any]:
    """Run all sweep counts for one output size."""
    size_dir = ensure_dir(RESULT_DIR / f"{size}x{size}")
    base = baseline_outputs(size)

    save_grayscale_png(size_dir / "high_reference.png", base["high_reference"])
    save_grayscale_png(size_dir / "low_guide.png", base["low_guide"])
    save_grayscale_png(size_dir / "nearest.png", base["nearest"])
    save_grayscale_png(size_dir / "bilinear.png", base["bilinear"])
    save_grayscale_png(size_dir / "bicubic.png", base["bicubic"])
    save_grayscale_png(size_dir / "confidence.png", base["confidence"])

    runs = []
    comparison_images = [
        base["high_reference"],
        base["nearest"],
        base["bilinear"],
        base["bicubic"],
        base["confidence"],
    ]
    for sweep_index, sweeps in enumerate(SWEEPS):
        model_c_config = AnnealConfig(
            decoder_seed=DECODER_SEED_BASE + index * 100 + sweep_index,
            sweeps=sweeps,
            temp_start=0.35,
            temp_end=0.01,
            proposal_sigma=0.08,
        )
        model_c_rendered, model_c_time = timed_call(
            "model_c_decode",
            model_c_decode,
            base["high_reference"],
            model_c_params,
            model_c_config,
        )
        model_d_rendered, model_d_time = timed_call(
            "model_d_decode",
            model_d_decode,
            base["bilinear"],
            base["confidence"],
            base["texture"],
            decoder_seed=DECODER_SEED_BASE + 1000 + index * 100 + sweep_index,
            sweeps=sweeps,
            temp_start=0.35,
            temp_end=0.01,
            proposal_sigma=0.08,
            **model_d_params,
        )

        model_c_metrics = comparison_summary(
            model_c_rendered,
            reference=base["high_reference"],
            foreground_mask=base["foreground_mask"],
        )
        model_d_metrics = comparison_summary(
            model_d_rendered,
            reference=base["high_reference"],
            foreground_mask=base["foreground_mask"],
        )
        diff_model_d_vs_bilinear = np.abs(model_d_rendered - base["bilinear"])

        sweep_name = f"sweeps_{sweeps:02d}"
        save_grayscale_png(size_dir / f"rendered_model_c_{sweep_name}.png", model_c_rendered)
        save_grayscale_png(size_dir / f"rendered_model_d_{sweep_name}.png", model_d_rendered)
        save_grayscale_png(size_dir / f"diff_model_d_{sweep_name}_vs_bilinear.png", diff_model_d_vs_bilinear)
        save_comparison_png(
            size_dir / f"comparison_{sweep_name}.png",
            [
                base["high_reference"],
                base["bilinear"],
                base["confidence"],
                model_c_rendered,
                model_d_rendered,
                diff_model_d_vs_bilinear,
            ],
        )

        comparison_images.append(model_c_rendered)
        comparison_images.append(model_d_rendered)
        runs.append(
            {
                "sweeps": sweeps,
                "timings": {
                    "model_c_decode_seconds": model_c_time["seconds"],
                    "model_d_decode_seconds": model_d_time["seconds"],
                },
                "metrics": {
                    "model_c": model_c_metrics,
                    "model_d": model_d_metrics,
                },
            }
        )

    save_comparison_png(size_dir / "comparison_sweep_summary.png", comparison_images)
    result = {
        "size": size,
        "low_size": base["low_size"],
        "sweeps": SWEEPS,
        "baseline_timings": base["baseline_timings"],
        "baseline_metrics": base["baseline_metrics"],
        "runs": runs,
    }
    save_json(size_dir / "metrics.json", json_safe(result))
    return result


def make_notes(config: dict[str, Any], results: list[dict[str, Any]]) -> str:
    """Build the top-level experiment notes."""
    baseline_rows = "\n".join(
        "| {size} | {nearest_t:.6f} | {bilinear_t:.6f} | {bicubic_t:.6f} | {nearest_mad:.6f} | {bilinear_mad:.6f} | {bicubic_mad:.6f} |".format(
            size=f"{row['size']}x{row['size']}",
            nearest_t=row["baseline_timings"]["nearest_seconds"],
            bilinear_t=row["baseline_timings"]["bilinear_seconds"],
            bicubic_t=row["baseline_timings"]["bicubic_seconds"],
            nearest_mad=row["baseline_metrics"]["nearest"]["mad_vs_reference"],
            bilinear_mad=row["baseline_metrics"]["bilinear"]["mad_vs_reference"],
            bicubic_mad=row["baseline_metrics"]["bicubic"]["mad_vs_reference"],
        )
        for row in results
    )
    sweep_rows = "\n".join(
        "| {size} | {sweeps} | {model_c_t:.6f} | {model_d_t:.6f} | {model_c_mad:.6f} | {model_d_mad:.6f} | {model_c_ssim:.6f} | {model_d_ssim:.6f} | {model_c_edge:.6f} | {model_d_edge:.6f} |".format(
            size=f"{row['size']}x{row['size']}",
            sweeps=run["sweeps"],
            model_c_t=run["timings"]["model_c_decode_seconds"],
            model_d_t=run["timings"]["model_d_decode_seconds"],
            model_c_mad=run["metrics"]["model_c"]["mad_vs_reference"],
            model_d_mad=run["metrics"]["model_d"]["mad_vs_reference"],
            model_c_ssim=run["metrics"]["model_c"]["ssim_global_vs_reference"],
            model_d_ssim=run["metrics"]["model_d"]["ssim_global_vs_reference"],
            model_c_edge=run["metrics"]["model_c"]["edge_leakage"],
            model_d_edge=run["metrics"]["model_d"]["edge_leakage"],
        )
        for row in results
        for run in row["runs"]
    )
    return f"""# Model C/D Decode Sweep Quality Benchmark

## Question

Model C / Model D の sweep 数を増やしたとき、decode time と synthetic cross に対する比較指標はどう変わるか。

## Hypothesis

現行の Python Metropolis 型 relaxation decoder は、同じ画像サイズでは sweep 数にほぼ比例して decode time が増える。品質指標は短い sweep から改善する可能性があるが、単純補間 baseline との差は別に確認する必要がある。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_012_decode_sweep_quality.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Decoder seed base: {config["decoder_seed_base"]}
- Output sizes: {config["sizes"]}
- Sweeps: {config["sweeps"]}
- Shape: synthetic cross
- Model C config: `{config["model_c_params"]}`
- Model D config: `{config["model_d_params"]}`
- Decode config: `{config["decode_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

baseline は nearest、bilinear、bicubic upscaling とした。baseline はサイズごとに一度だけ計測し、sweep 数による変化とは分けて扱う。metrics の reference は同じ synthetic cross を高解像度で生成した比較用参照であり、実画像の Ground Truth ではない。

| Size | Nearest seconds | Bilinear seconds | Bicubic seconds | Nearest MAD | Bilinear MAD | Bicubic MAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{baseline_rows}

## Result

| Size | Sweeps | Model C seconds | Model D seconds | Model C MAD | Model D MAD | Model C SSIM | Model D SSIM | Model C edge leakage | Model D edge leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{sweep_rows}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Per-size metrics: `<size>x<size>/metrics.json`
- Per-size PNGs: `high_reference.png`, `low_guide.png`, `nearest.png`, `bilinear.png`, `bicubic.png`, `confidence.png`
- Per-sweep PNGs: `rendered_model_c_sweeps_*.png`, `rendered_model_d_sweeps_*.png`, `diff_model_d_sweeps_*_vs_bilinear.png`, `comparison_sweeps_*.png`

## Images

各sizeの `comparison_sweep_summary.png` に、reference、baselines、confidence、各sweepの Model C / Model D 出力を横並びで保存した。

![64x64 sweep comparison](64x64/comparison_sweep_summary.png)

![128x128 sweep comparison](128x128/comparison_sweep_summary.png)

## Interpretation

今回の測定では、同じ画像サイズ内で sweep 数を増やすと Model C / Model D の decode time はおおむね増加した。Model D は sweep 数を増やすと synthetic reference への MAD が改善したが、64x64 / 128x128 とも bilinear / bicubic baseline の MAD よりは悪かった。Model C はこの multi-resolution 比較では高解像度 synthetic reference を直接 guide としているため、Model D や low-resolution baseline と同じ役割の復元器として比較しない。

## Limitations

- synthetic cross のみで、自然画像patchやsoft gradientでは未確認。
- metrics の reference は synthetic high-resolution cross であり、実画像の Ground Truth ではない。
- 画素数 scaling は `results/2026-05-17-decode-time-scaling/` の結果を参照し、本実験では同一サイズ内の sweep scaling を中心に読む。
- 現行 Python 実装の実行時間であり、Rust core、固定小数点、並列化、近似更新では未評価。
- Model D はこの条件でも単純補間 baseline を上回っておらず、実用圧縮形式や super-resolution 性能を示す結果ではない。

## Next

- Model C / D の更新ループ高速化や Rust core 化を検討するときは、この sweep scaling を Python 実装の制限として参照する。
- 品質改善の検討は、sweep 数だけを増やすよりも Issue #67 の confidence map / pairwise term 再設計候補と分けて進める。
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
    decode_config = {
        "temp_start": 0.35,
        "temp_end": 0.01,
        "proposal_sigma": 0.08,
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
        "decode_config": decode_config,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    results = [
        run_size(size, index, model_c_params, model_d_params)
        for index, size in enumerate(SIZES)
    ]
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", json_safe({"sizes": results}))
    (RESULT_DIR / "notes.md").write_text(make_notes(config, results), encoding="utf-8")
    for row in results:
        for run in row["runs"]:
            print(
                f"{row['size']}x{row['size']} sweeps={run['sweeps']}: "
                f"model_c={run['timings']['model_c_decode_seconds']:.3f}s "
                f"model_d={run['timings']['model_d_decode_seconds']:.3f}s "
                f"model_d_mad={run['metrics']['model_d']['mad_vs_reference']:.6f}"
            )


if __name__ == "__main__":
    main()
