"""Compare lightweight gradient metrics for Model D and interpolation baselines."""

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
from experiments.exp_008_model_d_natural_patch import downscale_block_average
from sidf_lab.confidence import gradient_confidence
from sidf_lab.guides import cross
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json
from sidf_lab.metrics import (
    gradient_magnitude,
    mad,
    perceptual_gradient_summary,
    psnr,
    ssim_global,
)


RESULT_DIR = Path("results/2026-06-14-issue-78-perceptual-gradient-metrics")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-06-14"
EXPERIMENT_SEED = 20260614
CROSS_DECODER_SEED = 7800
NATURAL_DECODER_SEED = 7801

MODEL_PARAMS = {
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


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def evaluate(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "mad": mad(reference, candidate),
        "psnr": psnr(reference, candidate),
        "ssim_global": ssim_global(reference, candidate),
    }
    values.update(perceptual_gradient_summary(reference, candidate))
    return values


def run_case(
    case_name: str,
    reference: np.ndarray,
    low_guide: np.ndarray,
    *,
    decoder_seed: int,
    decode_config: dict[str, float | int],
) -> dict[str, Any]:
    case_dir = ensure_dir(RESULT_DIR / case_name)
    size = reference.shape[0]
    nearest, nearest_seconds = timed_call(upscale, low_guide, size, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, size, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, size, 3)
    confidence = gradient_confidence(bilinear, min_confidence=0.2, max_confidence=1.0, scale=4.0)
    texture = seeded_texture(bilinear.shape, seed=EXPERIMENT_SEED)
    model_d, model_d_seconds = timed_call(
        model_d_decode,
        bilinear,
        confidence,
        texture,
        decoder_seed=decoder_seed,
        **decode_config,
        **MODEL_PARAMS,
    )

    outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
        "model_d": model_d,
    }
    timings = {
        "nearest": nearest_seconds,
        "bilinear": bilinear_seconds,
        "bicubic": bicubic_seconds,
        "model_d": model_d_seconds,
    }
    metrics = {name: evaluate(image, reference) for name, image in outputs.items()}

    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    save_grayscale_png(case_dir / "confidence.png", confidence)
    for name, image in outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
        save_grayscale_png(case_dir / f"gradient_{name}.png", gradient_magnitude(image))
        save_grayscale_png(case_dir / f"diff_{name}_vs_reference.png", np.abs(image - reference))
    save_grayscale_png(case_dir / "gradient_reference.png", gradient_magnitude(reference))
    guide_display = low_guide if low_guide.shape == reference.shape else upscale(low_guide, size, 0)
    save_comparison_png(
        case_dir / "comparison.png",
        [reference, guide_display, nearest, bilinear, bicubic, confidence, model_d],
    )
    save_comparison_png(
        case_dir / "gradient_comparison.png",
        [gradient_magnitude(reference), *[gradient_magnitude(outputs[name]) for name in outputs]],
    )
    return {"metrics": metrics, "timings": timings}


def format_rows(case: dict[str, Any]) -> str:
    rows = []
    for name in ("nearest", "bilinear", "bicubic", "model_d"):
        values = case["metrics"][name]
        rows.append(
            "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {grad:.6f} | "
            "{corr:.6f} | {orientation:.3f} | {laplacian:.6f} | {seconds:.6f} |".format(
                name=name,
                mad=values["mad"],
                psnr=values["psnr"],
                ssim=values["ssim_global"],
                grad=values["gradient_magnitude_mad"],
                corr=values["gradient_magnitude_correlation"],
                orientation=values["strong_edge_orientation_error_degrees"],
                laplacian=values["laplacian_mad"],
                seconds=case["timings"][name],
            )
        )
    return "\n".join(rows)


def build_notes(config: dict[str, Any], results: dict[str, Any]) -> str:
    return f"""# Model D Perceptual and Gradient Metrics

## Question

MAD / PSNR / global SSIMだけでは区別しにくい境界・勾配変化を、依存追加なしの勾配系指標でどこまで補足できるか。

## Hypothesis

Model Dの粒状変化や境界方向の乱れは、画素差だけでなくraw gradient magnitude、勾配位置相関、強エッジ方向、Laplacian応答の差として現れる。hard edgeのcrossと自然画像patchでは、同じ指標でも解釈が異なる。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_016_perceptual_gradient_metrics.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Cross decoder seed: {config["cross_decoder_seed"]}
- Natural patch decoder seed: {config["natural_decoder_seed"]}
- Cross: 16x16 guide to 64x64 output
- Natural patch: 32x32 guide to 128x128 output
- Model params: `{config["model_params"]}`
- Cross decode config: `{config["cross_decode_config"]}`
- Natural decode config: `{config["natural_decode_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

nearest、bilinear、bicubic upscalingをbaselineとし、現行Model D candidateと比較した。crossはsynthetic high-resolution reference、natural patchはPublic Domain画像cropをGround Truthとして使った。

## Metrics

- `gradient_magnitude_mad`: 画像ごとの最大値正規化を行わず、raw gradient magnitudeの絶対差を平均する。勾配強度の差を見る。
- `gradient_magnitude_correlation`: gradient magnitude mapのPearson相関。勾配の強弱が同じ位置に現れるかを見るが、絶対強度差は単独では表さない。
- `strong_edge_orientation_error_degrees`: referenceの非ゼロ勾配上位25%で、符号を区別しない方向誤差を度数で平均する。
- `laplacian_mad`: 4近傍Laplacian応答の絶対差。細かな振動、ringing、粒状変化にも反応する。
- LPIPSは追加dependencyと学習済み重みを必要とするため、このrunでは使用しない。

### Cross

| Output | MAD | PSNR | SSIM | Gradient magnitude MAD | Gradient correlation | Strong-edge orientation error | Laplacian MAD | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{format_rows(results["cross"])}

### Natural Patch

| Output | MAD | PSNR | SSIM | Gradient magnitude MAD | Gradient correlation | Strong-edge orientation error | Laplacian MAD | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{format_rows(results["natural_patch"])}

## Saved Artifacts

- `config.json`
- `metrics.json`
- `notes.md`
- `cross/comparison.png`, `cross/gradient_comparison.png`
- `natural_patch/comparison.png`, `natural_patch/gradient_comparison.png`
- 各caseのreference、guide、baseline、Model D、confidence、gradient map、reference差分PNG

## Images

![Cross output comparison](cross/comparison.png)

![Cross gradient comparison](cross/gradient_comparison.png)

![Natural patch output comparison](natural_patch/comparison.png)

![Natural patch gradient comparison](natural_patch/gradient_comparison.png)

## Result

crossではnearestのgradient magnitude MADが最小 `0.013946`、gradient correlationが最大 `0.716052` だった。Model Dはgradient magnitude MAD `0.044586`、gradient correlation `0.605796`、Laplacian MAD `0.117570` で、今回のbaselineより勾配強度差と局所高周波差が大きかった。strong-edge orientation errorだけを見るとnearest `12.046` 度に対してModel D `9.258` 度だったが、bilinear `2.081` 度とbicubic `2.013` 度より大きかった。

natural patchではbicubicがgradient magnitude MAD最小 `0.029326`、gradient correlation最大 `0.612533`、orientation error最小 `33.759` 度、Laplacian MAD最小 `0.087473` だった。Model Dはgradient magnitude MAD `0.033867`、gradient correlation `0.304795`、orientation error `39.389` 度、Laplacian MAD `0.136666` だった。

既存の最大値正規化gradient MADとは異なり、今回のgradient magnitude MADは勾配強度の絶対差を保持する。gradient correlationは位置関係、orientation errorは強エッジ方向、Laplacian MADは局所的な高周波差を別々に表す。

## Interpretation

各指標は単独の品質順位ではなく、画素差と構造差のどこでbaselineとModel Dが異なるかを読む補助値として扱う。今回のModel DはMAD / SSIMの悪化と同時にgradient magnitude MAD、gradient correlation、Laplacian MADでもbaselineを上回らなかった。crossのnearestは方向誤差ではModel Dより悪い一方、MADとgradient magnitude MADでは良く、指標間で順位が一致しない例になった。

crossでは境界位置と方向が既知なので方向誤差を直接読みやすい。natural patchではtexture、弱勾配、撮像由来の構造が混在するため、Laplacian MADやgradient correlationの悪化をそのまま知覚品質の悪化と同一視しない。

このrunはModel Dの「真の優位性」、super-resolution、compressionの成立を示すものではない。

## Limitations

- crossと1枚の128x128自然画像patchだけの小規模比較である。
- global SSIMはwindowed SSIMではない。
- gradientとLaplacianは単純な有限差分であり、人間の知覚モデルではない。
- strong-edge thresholdはreferenceの非ゼロ勾配上位25%に固定した。
- LPIPSなど学習済み知覚指標との相関は未確認。
- decode timeはこの環境の小画像runに限る。

## Next

- 複数の自然画像patchへ広げる場合は、ライセンスとcrop手順を固定し、指標間の順位一致・不一致を集計する。
- LPIPSを導入する場合はoptional dependencyとして分離し、モデル重み、version、offline再現性を記録する。
"""


def main() -> None:
    ensure_dir(RESULT_DIR)
    cross_reference = cross(size=64, width=7, value=0.5)
    cross_low = cross(size=16, width=2, value=0.5)
    natural_reference = np.load(SOURCE_ASSET)
    natural_low = downscale_block_average(natural_reference, 32)

    results = {
        "cross": run_case(
            "cross",
            cross_reference,
            cross_low,
            decoder_seed=CROSS_DECODER_SEED,
            decode_config=CROSS_DECODE_CONFIG,
        ),
        "natural_patch": run_case(
            "natural_patch",
            natural_reference,
            natural_low,
            decoder_seed=NATURAL_DECODER_SEED,
            decode_config=NATURAL_DECODE_CONFIG,
        ),
    }
    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "cross_decoder_seed": CROSS_DECODER_SEED,
        "natural_decoder_seed": NATURAL_DECODER_SEED,
        "source_asset": SOURCE_ASSET.as_posix(),
        "source_page": "https://commons.wikimedia.org/wiki/File:Landscape.jpg",
        "model_params": MODEL_PARAMS,
        "cross_decode_config": CROSS_DECODE_CONFIG,
        "natural_decode_config": NATURAL_DECODE_CONFIG,
        "strong_edge_quantile": 0.75,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", results)
    (RESULT_DIR / "notes.md").write_text(build_notes(config, results), encoding="utf-8")

    for case_name, case in results.items():
        print(case_name)
        for name, values in case["metrics"].items():
            print(
                f"  {name}: mad={values['mad']:.6f} "
                f"gradient_mad={values['gradient_magnitude_mad']:.6f} "
                f"gradient_corr={values['gradient_magnitude_correlation']:.6f}"
            )


if __name__ == "__main__":
    main()
