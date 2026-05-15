"""Run the Model C freeze candidate benchmark over several synthetic shapes."""

from __future__ import annotations

import platform
import struct
import sys
import time
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from sidf_lab.anneal import AnnealConfig, model_c_decode
from sidf_lab.energy import ModelCParams
from sidf_lab.guides import add_noise, circle, cross, diagonal, horizontal_gradient
from sidf_lab.io import ensure_dir, save_json
from sidf_lab.metrics import edge_leakage, mad, region_summary


RESULT_DIR = Path("results/2026-05-16-model-c-freeze-benchmark")
DATE = "2026-05-16"
EXPERIMENT_SEED = 20260516
DECODER_SEED_BASE = 4200
SIZE = 32
NOISE_SIGMA = 0.03


def save_grayscale_png(path: str | Path, image: np.ndarray) -> None:
    """Save a [0, 1] grayscale image as an 8-bit PNG without optional deps."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pixels = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    height, width = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )
    target.write_bytes(png)


def save_comparison_png(path: str | Path, images: list[np.ndarray], gap: int = 2) -> None:
    """Save a simple horizontal comparison strip."""
    if not images:
        raise ValueError("images must not be empty")
    normalized = [np.asarray(image, dtype=np.float64) for image in images]
    height = normalized[0].shape[0]
    if any(image.ndim != 2 or image.shape[0] != height for image in normalized):
        raise ValueError("all images must be 2D and have the same height")
    separator = np.ones((height, gap), dtype=np.float64)
    parts: list[np.ndarray] = []
    for index, image in enumerate(normalized):
        if index:
            parts.append(separator)
        parts.append(image)
    save_grayscale_png(path, np.concatenate(parts, axis=1))


def thin_line(size: int, value: float = 0.5) -> np.ndarray:
    """Return a one-pixel vertical guide through the image center."""
    guide = np.zeros((size, size), dtype=np.float64)
    guide[:, size // 2] = value
    return guide


def masks_for_shape(name: str, clean_guide: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Return foreground/background masks and a note about their meaning."""
    if name == "soft_gradient":
        foreground = clean_guide >= 0.5
        background = clean_guide < 0.5
        note = "soft_gradientでは foreground/background は右半分/左半分の便宜的な分割。edge leakageは不適用。"
    else:
        foreground = clean_guide > 0.0
        background = ~foreground
        note = "foregroundはclean guideで非ゼロの画素、backgroundはそれ以外。"
    return foreground, background, note


def gradient_visual_note(rendered: np.ndarray) -> str:
    """Describe simple visual checks for a soft gradient output."""
    row_mean = rendered.mean(axis=0)
    backward_steps = int(np.sum(np.diff(row_mean) < -0.02))
    large_jumps = int(np.sum(np.abs(np.diff(row_mean)) > 0.12))
    if backward_steps == 0 and large_jumps == 0:
        return "列平均では大きな逆行や急な段差は見られず、今回のrunでは階調の硬い分断は目立たない。"
    return (
        f"列平均で backward_steps={backward_steps}, large_jumps={large_jumps} を検出した。"
        "視覚確認では階調の分断や局所的な段差に注意が必要。"
    )


def make_shape_notes(
    shape_name: str,
    config: dict[str, object],
    metrics: dict[str, float | None | str],
    mask_note: str,
    visual_note: str,
) -> str:
    edge_leakage_row = (
        "Not applicable"
        if metrics["model_c_edge_leakage"] is None
        else f'{metrics["model_c_edge_leakage"]:.6f}'
    )
    baseline_edge_row = (
        "Not applicable"
        if metrics["baseline_edge_leakage"] is None
        else f'{metrics["baseline_edge_leakage"]:.6f}'
    )
    return f"""# Model C Freeze Benchmark: {shape_name}

## Question

Model Cは `{shape_name}` guideで、noisy static guide direct displayに対して安定した再構成候補を出せるか。

## Hypothesis

hard edgeを持つshapeでは、data fidelityとedge-aware interactionにより背景漏れを抑えられる。soft gradientでは、edge leakageより階調の連続性と段差の有無を重視する。

## Setup

- Command: `$env:PYTHONPATH = "src"; python experiments/exp_004_shape_benchmark.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Decoder seed: {config["decoder_seed"]}
- Input guide: synthetic `{shape_name}` with deterministic Gaussian noise
- Input size: {config["input_size"]}x{config["input_size"]}
- Output size: {config["output_size"]}x{config["output_size"]}
- Model: Model C
- Model config: `{config["model_c_params"]}`
- Anneal config: `{config["anneal_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

baselineは、noiseを加えたstatic guideをそのまま表示する `baseline_direct.png` とした。

## Metrics

| Metric | Baseline direct | Model C |
| --- | ---: | ---: |
| MAD vs clean guide | {metrics["baseline_mad_vs_clean"]:.6f} | {metrics["model_c_mad_vs_clean"]:.6f} |
| Foreground mean | {metrics["baseline_foreground_mean"]:.6f} | {metrics["model_c_foreground_mean"]:.6f} |
| Background mean | {metrics["baseline_background_mean"]:.6f} | {metrics["model_c_background_mean"]:.6f} |
| Edge leakage | {baseline_edge_row} | {edge_leakage_row} |
| Foreground variance | {metrics["baseline_foreground_variance"]:.6f} | {metrics["model_c_foreground_variance"]:.6f} |
| Background variance | {metrics["baseline_background_variance"]:.6f} | {metrics["model_c_background_variance"]:.6f} |
| Decode time seconds | 0.000000 | {metrics["decode_time_seconds"]:.6f} |

Mask note: {mask_note}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Notes: `notes.md`
- Clean guide image: `guide_clean.png`
- Static guide image: `static_guide.png`
- Baseline image: `baseline_direct.png`
- Rendered image: `rendered_model_c.png`
- Difference image: `diff_model_c_vs_baseline.png`
- Comparison image: `comparison.png`

## Images

### Comparison

![Comparison of clean guide, static guide, baseline direct, Model C, and absolute difference](comparison.png)

### Clean Guide

![Clean synthetic guide](guide_clean.png)

### Static Guide

![Noisy static guide](static_guide.png)

### Baseline Direct

![Baseline direct rendering of the static guide](baseline_direct.png)

### Model C Rendered

![Model C rendered output](rendered_model_c.png)

### Difference

![Absolute difference between Model C and baseline direct](diff_model_c_vs_baseline.png)

## Result

このshapeのmetricsとPNG成果物を保存した。crossについてはfreeze criteriaの暫定目安と比較する。

## Interpretation

{visual_note}

## Limitations

- synthetic guideのみで、実画像パッチでは未確認。
- decode timeは環境依存。
- edge widthは今回の小さい32x32 synthetic guidesでは安定した定義を置けなかったため未計算。
- Rust固定小数点や環境非依存のbit-perfect再現性は未確認。関連Issue: #16。

## Next

- cross以外の合格目安を別途定義する。
- Rust移植前にPRNG、丸め、固定小数点の再現性要件をIssue #16で整理する。
"""


def make_summary_notes(summary: list[dict[str, object]]) -> str:
    rows = "\n".join(
        "| {shape} | {mad:.6f} | {bg:.6f} | {edge} | {time:.6f} |".format(
            shape=row["shape"],
            mad=row["model_c_mad_vs_clean"],
            bg=row["model_c_background_mean"],
            edge="N/A" if row["model_c_edge_leakage"] is None else f"{row['model_c_edge_leakage']:.6f}",
            time=row["decode_time_seconds"],
        )
        for row in summary
    )
    cross = next(row for row in summary if row["shape"] == "cross")
    cross_pass = (
        cross["model_c_background_mean"] <= 0.02
        and cross["model_c_edge_leakage"] <= 0.02
        and cross["model_c_mad_vs_clean"] <= 0.03
    )
    return f"""# Model C Freeze Benchmark Summary

## Question

Model CはRust移植前の基準実装候補として、cross以外の基本shapeでも保存形式つきで評価できるか。

## Hypothesis

crossでは既存の暫定目安を満たし、diagonal / circle / thin lineでも大きな背景漏れは抑えられる。soft gradientではエッジ指標ではなく階調の連続性を確認する必要がある。

## Setup

- Command: `$env:PYTHONPATH = "src"; python experiments/exp_004_shape_benchmark.py`
- Date: {DATE}
- Experiment seed: {EXPERIMENT_SEED}
- Decoder seed base: {DECODER_SEED_BASE}
- Input size: {SIZE}x{SIZE}
- Output size: {SIZE}x{SIZE}
- Static noise sigma: {NOISE_SIGMA}
- Model: Model C

## Baseline

全shapeで、noiseを加えたstatic guideをそのまま表示する `baseline_direct.png` をbaselineとした。

## Metrics

| Shape | Model C MAD | Model C background mean | Model C edge leakage | Decode time seconds |
| --- | ---: | ---: | ---: | ---: |
{rows}

## Saved Artifacts

- Summary metrics: `summary_metrics.json`
- Per-shape artifacts: `<shape>/config.json`, `<shape>/metrics.json`, `<shape>/notes.md`, `<shape>/*.png`

## Images

各shapeの `notes.md` に、Git管理される主要PNGへのMarkdown画像参照を保存した。

## Result

cross baseline criteria pass: `{cross_pass}`.

## Interpretation

このbenchmarkは、Model CをRust移植前の候補として評価するための保存形式を作った段階である。結果はshapeごとのsynthetic条件に限られ、一般画像品質や実用圧縮性能を示すものではない。

## Limitations

- cross以外の合格目安は未定義。
- soft gradientはedge leakageで評価しにくく、視覚的な階調確認を併記した。
- edge widthは今回未計算。
- Rust固定小数点やbit-perfect再現性はIssue #16で別途扱う。

## Next

- Issue #16 でRust移植前の再現性要件を整理する。
- cross以外のshapeに対する暫定合格目安を定義する。
"""


def run_shape(
    index: int,
    name: str,
    guide_factory: Callable[[], np.ndarray],
    params: ModelCParams,
    anneal_template: AnnealConfig,
) -> dict[str, object]:
    shape_dir = ensure_dir(RESULT_DIR / name)
    clean_guide = guide_factory()
    static_guide = add_noise(clean_guide, seed=EXPERIMENT_SEED + index, sigma=NOISE_SIGMA)
    baseline_direct = static_guide.copy()
    decoder_seed = DECODER_SEED_BASE + index
    anneal_config = AnnealConfig(
        decoder_seed=decoder_seed,
        sweeps=anneal_template.sweeps,
        temp_start=anneal_template.temp_start,
        temp_end=anneal_template.temp_end,
        proposal_sigma=anneal_template.proposal_sigma,
    )

    start = time.perf_counter()
    rendered = model_c_decode(static_guide, params, anneal_config)
    decode_time = time.perf_counter() - start

    foreground_mask, background_mask, mask_note = masks_for_shape(name, clean_guide)
    baseline_fg = region_summary(baseline_direct, foreground_mask)
    baseline_bg = region_summary(baseline_direct, background_mask)
    model_c_fg = region_summary(rendered, foreground_mask)
    model_c_bg = region_summary(rendered, background_mask)

    hard_edge = name != "soft_gradient"
    metrics: dict[str, float | None | str] = {
        "baseline_mad_vs_clean": mad(baseline_direct, clean_guide),
        "model_c_mad_vs_clean": mad(rendered, clean_guide),
        "baseline_foreground_mean": baseline_fg["mean"],
        "model_c_foreground_mean": model_c_fg["mean"],
        "baseline_background_mean": baseline_bg["mean"],
        "model_c_background_mean": model_c_bg["mean"],
        "baseline_edge_leakage": edge_leakage(baseline_direct, foreground_mask, radius=2) if hard_edge else None,
        "model_c_edge_leakage": edge_leakage(rendered, foreground_mask, radius=2) if hard_edge else None,
        "baseline_foreground_variance": baseline_fg["variance"],
        "model_c_foreground_variance": model_c_fg["variance"],
        "baseline_background_variance": baseline_bg["variance"],
        "model_c_background_variance": model_c_bg["variance"],
        "decode_time_seconds": float(decode_time),
        "edge_width_pixels": None,
    }

    config = {
        "date": DATE,
        "shape": name,
        "experiment_seed": EXPERIMENT_SEED + index,
        "decoder_seed": decoder_seed,
        "input_size": SIZE,
        "output_size": SIZE,
        "static_noise_sigma": NOISE_SIGMA,
        "model": "Model C",
        "model_c_params": {
            "j_base": params.j_base,
            "lambda_data": params.lambda_data,
            "gamma": params.gamma,
        },
        "anneal_config": {
            "sweeps": anneal_config.sweeps,
            "temp_start": anneal_config.temp_start,
            "temp_end": anneal_config.temp_end,
            "proposal_sigma": anneal_config.proposal_sigma,
        },
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }

    diff = np.abs(rendered - baseline_direct)
    visual_note = gradient_visual_note(rendered) if name == "soft_gradient" else "hard edge shapeとして、背景平均とedge leakageを中心に見る。"
    save_json(shape_dir / "config.json", config)
    save_json(shape_dir / "metrics.json", metrics)
    save_grayscale_png(shape_dir / "guide_clean.png", clean_guide)
    save_grayscale_png(shape_dir / "static_guide.png", static_guide)
    save_grayscale_png(shape_dir / "baseline_direct.png", baseline_direct)
    save_grayscale_png(shape_dir / "rendered_model_c.png", rendered)
    save_grayscale_png(shape_dir / "diff_model_c_vs_baseline.png", diff)
    save_comparison_png(
        shape_dir / "comparison.png",
        [clean_guide, static_guide, baseline_direct, rendered, diff],
    )
    (shape_dir / "notes.md").write_text(
        make_shape_notes(name, config, metrics, mask_note, visual_note),
        encoding="utf-8",
    )

    summary = {"shape": name, **metrics}
    return summary


def main() -> None:
    ensure_dir(RESULT_DIR)
    params = ModelCParams(j_base=2.0, lambda_data=5.0, gamma=40.0)
    anneal_template = AnnealConfig(
        decoder_seed=DECODER_SEED_BASE,
        sweeps=80,
        temp_start=0.5,
        temp_end=0.01,
        proposal_sigma=0.12,
    )
    shapes: list[tuple[str, Callable[[], np.ndarray]]] = [
        ("cross", lambda: cross(size=SIZE, width=4, value=0.5)),
        ("diagonal", lambda: diagonal(size=SIZE, width=2, value=0.5)),
        ("circle", lambda: circle(size=SIZE, radius=8.0, value=0.5)),
        ("thin_line", lambda: thin_line(size=SIZE, value=0.5)),
        ("soft_gradient", lambda: horizontal_gradient(size=SIZE)),
    ]
    summary = [
        run_shape(index, name, guide_factory, params, anneal_template)
        for index, (name, guide_factory) in enumerate(shapes)
    ]
    save_json(RESULT_DIR / "summary_metrics.json", {"shapes": summary})
    (RESULT_DIR / "notes.md").write_text(make_summary_notes(summary), encoding="utf-8")
    for row in summary:
        edge = "N/A" if row["model_c_edge_leakage"] is None else f"{row['model_c_edge_leakage']:.6f}"
        print(
            f"{row['shape']}: mad={row['model_c_mad_vs_clean']:.6f} "
            f"bg={row['model_c_background_mean']:.6f} edge={edge}"
        )


if __name__ == "__main__":
    main()
