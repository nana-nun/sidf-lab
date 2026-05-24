"""Run Model D on a small public-domain natural image patch."""

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
from sidf_lab.io import ensure_dir, save_json
from sidf_lab.metrics import mad, psnr, ssim_global


RESULT_DIR = Path("results/2026-05-17-issue-36-model-d-natural-patch")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-05-17"
EXPERIMENT_SEED = 20260517
DECODER_SEED = 6300
LOW_SIZE = 32
HIGH_SIZE = 128


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run a callable and return its result plus elapsed seconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def downscale_block_average(image: np.ndarray, size: int) -> np.ndarray:
    """Downscale a square image by integer block averaging."""
    source = np.asarray(image, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("image must be a square 2D array")
    if source.shape[0] % size != 0:
        raise ValueError("target size must evenly divide source size")
    factor = source.shape[0] // size
    return source.reshape(size, factor, size, factor).mean(axis=(1, 3))


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Return a simple normalized gradient magnitude image."""
    gy, gx = np.gradient(np.asarray(image, dtype=np.float64))
    mag = np.sqrt(gx * gx + gy * gy)
    max_value = float(mag.max())
    if max_value == 0.0:
        return mag
    return mag / max_value


def natural_image_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compute GT and gradient-based metrics for natural-image reconstruction."""
    ref_grad = gradient_magnitude(reference)
    cand_grad = gradient_magnitude(candidate)
    strong_edge_mask = ref_grad >= float(np.quantile(ref_grad, 0.75))
    flat_mask = ref_grad <= float(np.quantile(ref_grad, 0.50))
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "mad_vs_gt": mad(candidate, reference),
        "psnr_vs_gt": psnr(reference, candidate),
        "ssim_global_vs_gt": ssim_global(reference, candidate),
        "mean_error": float(diff.mean()),
        "absolute_mean_error": float(abs(diff.mean())),
        "gradient_mad_vs_gt": mad(cand_grad, ref_grad),
        "strong_edge_mad_vs_gt": float(np.mean(np.abs(diff)[strong_edge_mask])),
        "flat_region_mad_vs_gt": float(np.mean(np.abs(diff)[flat_mask])),
    }


def build_notes(
    config: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    timings: dict[str, float],
) -> str:
    """Build the Japanese experiment note."""
    rows = "\n".join(
        "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {grad:.6f} | {edge:.6f} | {bias:.6f} | {seconds:.6f} |".format(
            name=name,
            mad=values["mad_vs_gt"],
            psnr=values["psnr_vs_gt"],
            ssim=values["ssim_global_vs_gt"],
            grad=values["gradient_mad_vs_gt"],
            edge=values["strong_edge_mad_vs_gt"],
            bias=values["mean_error"],
            seconds=timings[name],
        )
        for name, values in metrics.items()
    )
    return f"""# Model D Natural Patch GT Evaluation

## Question

Public Domain の自然画像patchをGround Truthとして、32x32 low-resolution guideから128x128 outputを作るとき、Model D candidate は nearest / bilinear / bicubic baseline と比べてどの指標を改善または悪化させるか。

## Hypothesis

Model D は confidence map と edge-aware interaction により、強い勾配付近を補間baselineより保つ可能性がある。一方で、現行のwhite-noise texture termは自然画像のGround Truth差分や輝度biasを悪化させる可能性がある。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_008_model_d_natural_patch.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Decoder seed: {config["decoder_seed"]}
- Ground Truth size: {config["high_size"]}x{config["high_size"]}
- Low guide size: {config["low_size"]}x{config["low_size"]}
- Low guide generation: {config["low_guide_method"]}
- Source asset: `{config["source_asset"]}`
- Source page: {config["source_page"]}
- License note: {config["license_note"]}
- Model: Model D candidate
- Model config: `{config["model_d_params"]}`
- Decode config: `{config["decode_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

baselineは、low guideを128x128へ戻す nearest、bilinear、bicubic upscaling とした。Ground Truth は同じ自然画像cropの128x128 grayscale画像であり、low guideはそのblock-average縮小から作った。

## Result

| Output | MAD vs GT | PSNR vs GT | Global SSIM vs GT | Gradient MAD | Strong-edge MAD | Mean error | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Ground Truth image: `high_reference.png`
- Low guide image: `low_guide.png`
- Baseline images: `nearest.png`, `bilinear.png`, `bicubic.png`
- Confidence map: `confidence.png`
- Model D rendered image: `rendered_model_d.png`
- Difference maps: `diff_model_d_vs_bilinear.png`, `diff_model_d_vs_gt.png`, `diff_bilinear_vs_gt.png`
- Comparison strip: `comparison.png`

## Images

![Comparison of GT, low guide upscaled, baselines, confidence, Model D, and differences](comparison.png)

![Ground Truth natural image patch](high_reference.png)

![Low-resolution guide](low_guide.png)

![Confidence map](confidence.png)

![Model D rendered output](rendered_model_d.png)

![Absolute difference between Model D and Ground Truth](diff_model_d_vs_gt.png)

## Interpretation

この結果は1枚の小さなpublic-domain画像cropに対する最小評価であり、super-resolutionやcompressionの成立を示すものではない。Model Dの評価は、Ground Truth差分、勾配差分、強エッジ帯の差分、輝度biasをbaselineと分けて読む。

今回のrunでは、Model D が nearest / bilinear / bicubic に対して総合的に改善したとは解釈しない。white-noise texture termと現在の重みが、自然画像GTとの差分をどの程度増やすかを見るための初期測定として扱う。

## Limitations

- サンプルは1枚の128x128 cropのみ。
- 元画像は絵画の写真であり、カメラ撮影の自然風景や標準画像データセット全体を代表しない。
- Global SSIM は依存なしの全画像SSIMであり、windowed SSIMではない。
- edge leakage は自然画像の明確なforeground/background境界ではないため使っていない。代替としてgradient MADとstrong-edge MADを保存した。
- decode timeはこの環境の小画像runに限る。

## Next

- texture term の寄与は Issue #37 のablationで分離する。
- 画像サンプル数を増やす場合は、ライセンス、crop位置、low guide生成手順を固定する。
- Model Dの重み再調整やtexture prior候補は、Issue #15 と接続して検討する。
"""


def main() -> None:
    ensure_dir(RESULT_DIR)
    high_reference = np.load(SOURCE_ASSET)
    low_guide = downscale_block_average(high_reference, LOW_SIZE)

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
    texture = seeded_texture(bilinear.shape, seed=EXPERIMENT_SEED)
    model_d_params = {
        "j_base": 1.8,
        "lambda_data": 6.0,
        "gamma": 35.0,
        "texture_weight": 0.35,
    }
    decode_config = {
        "sweeps": 18,
        "temp_start": 0.35,
        "temp_end": 0.01,
        "proposal_sigma": 0.08,
    }
    rendered_model_d, model_d_seconds = timed_call(
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
    metrics = {name: natural_image_metrics(image, high_reference) for name, image in outputs.items()}
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
        "low_guide_method": "4x4 block average from the 128x128 Ground Truth crop",
        "source_asset": SOURCE_ASSET.as_posix(),
        "source_page": "https://commons.wikimedia.org/wiki/File:Landscape.jpg",
        "source_title": "Meindert Hobbema: Landscape",
        "license_note": "Wikimedia Commons marks the faithful reproduction of this public-domain artwork as Public Domain / PD-Art.",
        "model": "Model D candidate",
        "model_d_params": model_d_params,
        "decode_config": decode_config,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }

    diff_model_d_vs_bilinear = np.abs(rendered_model_d - bilinear)
    diff_model_d_vs_gt = np.abs(rendered_model_d - high_reference)
    diff_bilinear_vs_gt = np.abs(bilinear - high_reference)

    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", {"outputs": metrics, "timings": timings})
    save_grayscale_png(RESULT_DIR / "high_reference.png", high_reference)
    save_grayscale_png(RESULT_DIR / "low_guide.png", low_guide)
    save_grayscale_png(RESULT_DIR / "nearest.png", nearest)
    save_grayscale_png(RESULT_DIR / "bilinear.png", bilinear)
    save_grayscale_png(RESULT_DIR / "bicubic.png", bicubic)
    save_grayscale_png(RESULT_DIR / "confidence.png", confidence)
    save_grayscale_png(RESULT_DIR / "rendered_model_d.png", rendered_model_d)
    save_grayscale_png(RESULT_DIR / "diff_model_d_vs_bilinear.png", diff_model_d_vs_bilinear)
    save_grayscale_png(RESULT_DIR / "diff_model_d_vs_gt.png", diff_model_d_vs_gt)
    save_grayscale_png(RESULT_DIR / "diff_bilinear_vs_gt.png", diff_bilinear_vs_gt)
    save_comparison_png(
        RESULT_DIR / "comparison.png",
        [
            high_reference,
            upscale(low_guide, HIGH_SIZE, 0),
            bilinear,
            bicubic,
            confidence,
            rendered_model_d,
            diff_model_d_vs_gt,
            diff_bilinear_vs_gt,
        ],
    )
    (RESULT_DIR / "notes.md").write_text(build_notes(config, metrics, timings), encoding="utf-8")

    for name, values in metrics.items():
        print(
            f"{name}: mad={values['mad_vs_gt']:.6f} "
            f"psnr={values['psnr_vs_gt']:.3f} "
            f"ssim={values['ssim_global_vs_gt']:.6f} "
            f"time={timings[name]:.6f}s"
        )


if __name__ == "__main__":
    main()
