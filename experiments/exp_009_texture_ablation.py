"""Run a small Model D texture-strength ablation."""

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
from sidf_lab.confidence import gradient_confidence
from sidf_lab.guides import cross
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json
from sidf_lab.metrics import comparison_summary


RESULT_DIR = Path("results/2026-05-24-texture-ablation")
DATE = "2026-05-24"
EXPERIMENT_SEED = 20260524
DECODER_SEED = 6400
LOW_SIZE = 16
HIGH_SIZE = 64
LOW_WIDTH = 2
HIGH_WIDTH = 7
TEXTURE_STRENGTHS = [0.0, 0.1, 0.35, 0.7]


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run a callable and return its result plus elapsed seconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def safe_name(strength: float) -> str:
    """Return a filesystem-friendly strength label."""
    return f"{strength:.2f}".replace(".", "_")


def texture_strength_to_field(base_texture: np.ndarray, strength: float) -> np.ndarray:
    """Map texture strength to the texture field used by the current decoder.

    Strength 0.0 removes texture from both the initial state and texture target.
    Non-zero strengths scale the same deterministic zero-mean texture field and
    also use the same value as the current Model D ``texture_weight``.
    """
    if strength == 0.0:
        return np.zeros_like(base_texture)
    return base_texture * (strength / 0.35)


def output_metrics(candidate: np.ndarray, reference: np.ndarray, foreground_mask: np.ndarray) -> dict[str, float | None]:
    """Return comparison metrics plus explicit brightness-bias summaries."""
    summary = comparison_summary(candidate, reference=reference, foreground_mask=foreground_mask)
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    summary["mean_error_vs_reference"] = float(diff.mean())
    summary["absolute_mean_error_vs_reference"] = float(abs(diff.mean()))
    summary["global_mean"] = float(np.mean(candidate))
    return summary


def build_notes(
    config: dict[str, Any],
    metrics: dict[str, dict[str, float | None]],
    timings: dict[str, float],
    bias_observation: str,
) -> str:
    """Build the Japanese experiment note."""
    rows = "\n".join(
        "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {bias:.6f} | {fg:.6f} | {bg:.6f} | {edge:.6f} | {seconds:.6f} |".format(
            name=name,
            mad=values["mad_vs_reference"],
            psnr=values["psnr_vs_reference"],
            ssim=values["ssim_global_vs_reference"],
            bias=values["mean_error_vs_reference"],
            fg=values["foreground_mean"],
            bg=values["background_mean"],
            edge=values["edge_leakage"],
            seconds=timings[name],
        )
        for name, values in metrics.items()
    )
    return f"""# Model D Texture Term Ablation

## Question

Model D candidate の white-noise texture term は、同一 guide / seed の reconstruction に対して改善、悪化、無影響のどれに見えるか。特に、輝度を一方向へ押す bias が見えるか。

## Hypothesis

現行の white-noise texture は意味的ディテールではなく粒状ノイズに近い。texture_strength を上げても、synthetic cross reference に対する PSNR / SSIM / MAD / edge leakage は改善せず、背景平均や輝度biasを悪化させる可能性がある。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_009_texture_ablation.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Decoder seed: {config["decoder_seed"]}
- Low guide size: {config["low_size"]}x{config["low_size"]}
- Output size: {config["high_size"]}x{config["high_size"]}
- Shape: synthetic cross
- Texture strengths: `{config["texture_strengths"]}`
- Model: Model D candidate texture ablation
- Model config except texture: `{config["base_model_d_params"]}`
- Decode config: `{config["decode_config"]}`
- Texture mapping: {config["texture_mapping"]}
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

baselineは nearest、bilinear、bicubic upscaling とした。ablationの主比較は `texture_strength=0.00` と非ゼロ値の Model D output である。

metricsのreferenceは同じsynthetic crossを64x64で生成した比較用参照であり、自然画像のGround Truthではない。

## Metrics

| Output | MAD vs reference | PSNR vs reference | Global SSIM vs reference | Mean error | Foreground mean | Background mean | Edge leakage | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Notes: `notes.md`
- Low guide image: `low_guide.png`
- Synthetic comparison reference: `high_reference.png`
- Baseline images: `nearest.png`, `bilinear.png`, `bicubic.png`
- Confidence map: `confidence.png`
- Texture field image: `texture_field.png`
- Rendered Model D images: `rendered_texture_*.png`
- Difference maps vs bilinear: `diff_texture_*_vs_bilinear.png`
- Difference maps vs reference: `diff_texture_*_vs_reference.png`
- Comparison strip: `comparison.png`

## Images

![Comparison of reference, baselines, confidence, texture field, and texture ablation outputs](comparison.png)

![Low-resolution guide](low_guide.png)

![Confidence map](confidence.png)

![Texture field](texture_field.png)

![Texture strength 0.00 output](rendered_texture_0_00.png)

![Texture strength 0.35 output](rendered_texture_0_35.png)

![Texture strength 0.70 output](rendered_texture_0_70.png)

![Absolute difference between texture strength 0.70 and reference](diff_texture_0_70_vs_reference.png)

## Result

{bias_observation}

## Interpretation

このrunでは、white-noise texture term は synthetic cross に対する意味のある質感生成としては扱えない。非ゼロtexture_strengthの差は小さく非単調で、PSNR / SSIM / MAD / edge leakage は baseline より改善しなかったため、現行設定では改善要因とは解釈しない。

ただし、現行実装の texture は draft 仕様に書かれた線形項 `texture_strength * sum t_i v_i` そのものではなく、`s_i + texture_i` を平坦部の target とする二乗項と初期状態への混入で効いている。この結果は「現行実装の white-noise texture 経路」の評価であり、structured texture prior 全体の否定ではない。

## Limitations

- synthetic cross 1条件のみの小規模 ablation である。
- metricsのreferenceは実画像Ground Truthではない。
- 現行実装は線形 texture term ではなく、texture target 二乗項と初期状態混入を含む。
- 同じ decoder seed を使っているが、texture_strength により初期状態が変わるため、完全に同一のMarkov chain比較ではない。
- decode timeはこの環境の小画像runに限る。

## Next

- Issue #56 では、この結果を前提に `texture_strength=0` を含め、confidence / data / texture 重みを分けて小規模gridで再評価する。
- Issue #15 / #48 の structured texture prior を使う場合も、white noise baselineとの差分として評価し、意味的ディテール生成とは断定しない。
"""


def main() -> None:
    ensure_dir(RESULT_DIR)
    high_reference = cross(size=HIGH_SIZE, width=HIGH_WIDTH, value=0.5)
    low_guide = cross(size=LOW_SIZE, width=LOW_WIDTH, value=0.5)
    foreground_mask = high_reference > 0.0

    nearest, nearest_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 3)
    confidence, confidence_seconds = timed_call(
        gradient_confidence,
        bilinear,
        min_confidence=0.2,
        max_confidence=1.0,
        scale=4.0,
    )
    base_texture = seeded_texture(bilinear.shape, seed=EXPERIMENT_SEED)
    base_model_d_params = {
        "j_base": 1.8,
        "lambda_data": 6.0,
        "gamma": 35.0,
    }
    decode_config = {
        "sweeps": 35,
        "temp_start": 0.35,
        "temp_end": 0.01,
        "proposal_sigma": 0.08,
    }

    outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
    }
    timings = {
        "nearest": nearest_seconds,
        "bilinear": bilinear_seconds,
        "bicubic": bicubic_seconds,
        "confidence": confidence_seconds,
    }

    for strength in TEXTURE_STRENGTHS:
        label = f"texture_{strength:.2f}"
        texture = texture_strength_to_field(base_texture, strength)
        rendered, seconds = timed_call(
            model_d_decode,
            bilinear,
            confidence,
            texture,
            decoder_seed=DECODER_SEED,
            texture_weight=strength,
            **decode_config,
            **base_model_d_params,
        )
        outputs[label] = rendered
        timings[label] = seconds

    metrics = {
        name: output_metrics(image, reference=high_reference, foreground_mask=foreground_mask)
        for name, image in outputs.items()
    }
    texture_rows = [metrics[f"texture_{strength:.2f}"] for strength in TEXTURE_STRENGTHS]
    texture_biases = [float(row["mean_error_vs_reference"]) for row in texture_rows]
    texture_backgrounds = [float(row["background_mean"]) for row in texture_rows]
    bias_observation = (
        "texture_strength を 0.00 から 0.70 へ変えても、mean error は "
        f"{min(texture_biases):.6f} から {max(texture_biases):.6f} の範囲に留まり、"
        f"background mean も {min(texture_backgrounds):.6f} から {max(texture_backgrounds):.6f} の範囲で非単調だった。"
        "このrunでは、texture_strength に比例した単純な一方向biasは確認できない。一方で、"
        f"texture_strength=0.00 でも background mean は {texture_backgrounds[0]:.6f} で、bilinear の background mean "
        f"{metrics['bilinear']['background_mean']:.6f} より高かったため、現行Model D relaxation経路そのものが背景側の明るさと差分を増やしている可能性がある。"
    )

    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "decoder_seed": DECODER_SEED,
        "low_size": LOW_SIZE,
        "high_size": HIGH_SIZE,
        "low_width": LOW_WIDTH,
        "high_width": HIGH_WIDTH,
        "shape": "synthetic cross",
        "model": "Model D candidate texture ablation",
        "texture_strengths": TEXTURE_STRENGTHS,
        "texture_mapping": "strength 0.0 uses a zero texture field and texture_weight=0.0; non-zero strengths scale the same deterministic zero-mean texture field by strength / 0.35 and use strength as texture_weight.",
        "base_model_d_params": base_model_d_params,
        "decode_config": decode_config,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }

    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", {"outputs": metrics, "timings": timings})
    save_grayscale_png(RESULT_DIR / "low_guide.png", low_guide)
    save_grayscale_png(RESULT_DIR / "high_reference.png", high_reference)
    save_grayscale_png(RESULT_DIR / "nearest.png", nearest)
    save_grayscale_png(RESULT_DIR / "bilinear.png", bilinear)
    save_grayscale_png(RESULT_DIR / "bicubic.png", bicubic)
    save_grayscale_png(RESULT_DIR / "confidence.png", confidence)
    save_grayscale_png(RESULT_DIR / "texture_field.png", np.clip(base_texture / 0.14 + 0.5, 0.0, 1.0))

    comparison_images = [high_reference, nearest, bilinear, bicubic, confidence, np.clip(base_texture / 0.14 + 0.5, 0.0, 1.0)]
    for strength in TEXTURE_STRENGTHS:
        label = f"texture_{strength:.2f}"
        file_label = safe_name(strength)
        rendered = outputs[label]
        save_grayscale_png(RESULT_DIR / f"rendered_texture_{file_label}.png", rendered)
        save_grayscale_png(RESULT_DIR / f"diff_texture_{file_label}_vs_bilinear.png", np.abs(rendered - bilinear))
        save_grayscale_png(RESULT_DIR / f"diff_texture_{file_label}_vs_reference.png", np.abs(rendered - high_reference))
        comparison_images.append(rendered)

    comparison_images.append(np.abs(outputs["texture_0.70"] - high_reference))
    save_comparison_png(RESULT_DIR / "comparison.png", comparison_images)
    (RESULT_DIR / "notes.md").write_text(build_notes(config, metrics, timings, bias_observation), encoding="utf-8")

    for name, values in metrics.items():
        print(
            f"{name}: mad={values['mad_vs_reference']:.6f} "
            f"psnr={values['psnr_vs_reference']:.3f} "
            f"ssim={values['ssim_global_vs_reference']:.6f} "
            f"bias={values['mean_error_vs_reference']:.6f} "
            f"edge={values['edge_leakage']:.6f} "
            f"time={timings[name]:.6f}s"
        )


if __name__ == "__main__":
    main()
