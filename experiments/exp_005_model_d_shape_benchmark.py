"""Run a saved Model D shape benchmark over low-resolution synthetic guides."""

from __future__ import annotations

import math
import platform
import struct
import sys
import time
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from sidf_lab.confidence import gradient_confidence
from sidf_lab.energy import valid_neighbors
from sidf_lab.guides import circle, diagonal, horizontal_gradient
from sidf_lab.io import ensure_dir, save_json
from sidf_lab.metrics import comparison_summary


RESULT_DIR = Path("results/2026-05-16-issue-30-model-d-shape-benchmark")
DATE = "2026-05-16"
EXPERIMENT_SEED = 20260516
DECODER_SEED_BASE = 5300
LOW_SIZE = 16
HIGH_SIZE = 64


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


def upscale(image: np.ndarray, size: int, order: int) -> np.ndarray:
    """Resize a square guide to ``size`` with nearest, bilinear, or bicubic interpolation."""
    source = np.asarray(image, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("image must be a square 2D array")
    if size <= 0:
        raise ValueError("size must be positive")
    if order == 0:
        return _resize_nearest(source, size)
    if order == 1:
        return _resize_bilinear(source, size)
    if order == 3:
        return _resize_bicubic(source, size)
    raise ValueError("order must be 0, 1, or 3")


def _source_positions(source_size: int, target_size: int) -> np.ndarray:
    if target_size == 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(0.0, source_size - 1, target_size, dtype=np.float64)


def _resize_nearest(source: np.ndarray, size: int) -> np.ndarray:
    positions = _source_positions(source.shape[0], size)
    indices = np.rint(positions).astype(int)
    return np.clip(source[np.ix_(indices, indices)], 0.0, 1.0)


def _resize_bilinear(source: np.ndarray, size: int) -> np.ndarray:
    positions = _source_positions(source.shape[0], size)
    y0 = np.floor(positions).astype(int)
    x0 = y0.copy()
    y1 = np.clip(y0 + 1, 0, source.shape[0] - 1)
    x1 = y1.copy()
    wy = positions - y0
    wx = wy.copy()
    top = (1.0 - wx)[None, :] * source[np.ix_(y0, x0)] + wx[None, :] * source[np.ix_(y0, x1)]
    bottom = (1.0 - wx)[None, :] * source[np.ix_(y1, x0)] + wx[None, :] * source[np.ix_(y1, x1)]
    resized = (1.0 - wy)[:, None] * top + wy[:, None] * bottom
    return np.clip(resized, 0.0, 1.0)


def _cubic_kernel(distance: np.ndarray) -> np.ndarray:
    """Catmull-Rom cubic interpolation kernel."""
    a = -0.5
    x = np.abs(distance)
    weights = np.zeros_like(x, dtype=np.float64)
    mask1 = x <= 1.0
    mask2 = (x > 1.0) & (x < 2.0)
    weights[mask1] = (a + 2.0) * x[mask1] ** 3 - (a + 3.0) * x[mask1] ** 2 + 1.0
    weights[mask2] = (
        a * x[mask2] ** 3
        - 5.0 * a * x[mask2] ** 2
        + 8.0 * a * x[mask2]
        - 4.0 * a
    )
    return weights


def _resize_bicubic(source: np.ndarray, size: int) -> np.ndarray:
    positions = _source_positions(source.shape[0], size)
    resized = np.zeros((size, size), dtype=np.float64)
    limit = source.shape[0] - 1
    for out_y, y in enumerate(positions):
        y_base = math.floor(float(y))
        y_indices = np.clip(np.arange(y_base - 1, y_base + 3), 0, limit)
        y_weights = _cubic_kernel(y - np.arange(y_base - 1, y_base + 3))
        for out_x, x in enumerate(positions):
            x_base = math.floor(float(x))
            x_indices = np.clip(np.arange(x_base - 1, x_base + 3), 0, limit)
            x_weights = _cubic_kernel(x - np.arange(x_base - 1, x_base + 3))
            patch = source[np.ix_(y_indices, x_indices)]
            resized[out_y, out_x] = float(y_weights @ patch @ x_weights)
    return np.clip(resized, 0.0, 1.0)


def seeded_texture(shape: tuple[int, int], seed: int, sigma: float = 0.035) -> np.ndarray:
    """Return a deterministic zero-centered fine texture field."""
    rng = np.random.default_rng(seed)
    texture = rng.normal(0.0, sigma, shape)
    return texture - float(texture.mean())


def model_d_local_energy(
    value: float,
    state: np.ndarray,
    guide: np.ndarray,
    confidence: np.ndarray,
    texture: np.ndarray,
    i: int,
    j: int,
    *,
    j_base: float,
    lambda_data: float,
    gamma: float,
    texture_weight: float,
) -> float:
    """Compute a local confidence-aware Model D candidate energy."""
    height, width = guide.shape
    s_i = float(guide[i, j])
    c_i = float(confidence[i, j])
    target = float(np.clip(s_i + texture[i, j], 0.0, 1.0))
    energy = lambda_data * c_i * (value - s_i) ** 2
    energy += texture_weight * (1.0 - c_i) * (value - target) ** 2
    for ni, nj in valid_neighbors(i, j, height, width):
        s_n = float(guide[ni, nj])
        n_val = float(state[ni, nj])
        j_ij = j_base * math.exp(-gamma * (s_i - s_n) ** 2)
        energy += j_ij * (value - n_val) ** 2
    return energy


def model_d_decode(
    guide: np.ndarray,
    confidence: np.ndarray,
    texture: np.ndarray,
    *,
    decoder_seed: int,
    sweeps: int,
    temp_start: float,
    temp_end: float,
    proposal_sigma: float,
    j_base: float,
    lambda_data: float,
    gamma: float,
    texture_weight: float,
) -> np.ndarray:
    """Decode a high-resolution guide with confidence-aware relaxation."""
    rng = np.random.default_rng(decoder_seed)
    state = np.clip(guide + 0.25 * texture, 0.0, 1.0)
    height, width = state.shape
    temperatures = np.geomspace(temp_start, temp_end, sweeps)

    for temp in temperatures:
        for idx in rng.permutation(height * width):
            i, j = divmod(int(idx), width)
            old_value = float(state[i, j])
            new_value = float(np.clip(old_value + rng.normal(0.0, proposal_sigma), 0.0, 1.0))
            old_energy = model_d_local_energy(
                old_value,
                state,
                guide,
                confidence,
                texture,
                i,
                j,
                j_base=j_base,
                lambda_data=lambda_data,
                gamma=gamma,
                texture_weight=texture_weight,
            )
            new_energy = model_d_local_energy(
                new_value,
                state,
                guide,
                confidence,
                texture,
                i,
                j,
                j_base=j_base,
                lambda_data=lambda_data,
                gamma=gamma,
                texture_weight=texture_weight,
            )
            delta = new_energy - old_energy
            if delta < 0.0 or rng.random() < math.exp(-delta / float(temp)):
                state[i, j] = new_value
    return state


def masks_for_shape(name: str, high_reference: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Return a hard-edge mask when the shape has one."""
    if name == "soft_gradient":
        return high_reference >= 0.5, "soft_gradient uses a right-half/left-half split only for mean summaries."
    return high_reference > 0.0, "foreground is the non-zero region in the clean high-resolution synthetic reference."


def gradient_naturalness(rendered: np.ndarray) -> dict[str, int | str]:
    """Return simple column-mean checks for soft-gradient continuity."""
    column_mean = rendered.mean(axis=0)
    backward_steps = int(np.sum(np.diff(column_mean) < -0.02))
    large_jumps = int(np.sum(np.abs(np.diff(column_mean)) > 0.12))
    if backward_steps == 0 and large_jumps == 0:
        note = "列平均では大きな逆行や急な段差は見られない。"
    else:
        note = "列平均に逆行または急な段差があり、階調の分断を追加確認する必要がある。"
    return {"backward_steps": backward_steps, "large_jumps": large_jumps, "note": note}


def metrics_for_outputs(
    outputs: dict[str, np.ndarray],
    reference: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    hard_edge: bool,
) -> dict[str, dict[str, float | None]]:
    """Compute shared metrics for baselines and Model D output."""
    metrics: dict[str, dict[str, float | None]] = {}
    for name, image in outputs.items():
        summary = comparison_summary(
            image,
            reference=reference,
            foreground_mask=foreground_mask,
            edge_radius=2,
            edge_width_radius=8,
        )
        if not hard_edge:
            summary["edge_leakage"] = None
            summary["edge_width_pixels"] = None
        metrics[name] = summary
    return metrics


def make_shape_notes(
    shape: str,
    config: dict[str, object],
    metrics: dict[str, dict[str, float | None]],
    mask_note: str,
    gradient_check: dict[str, int | str] | None,
) -> str:
    rows = "\n".join(
        "| {name} | {mad:.6f} | {edge} | {width} | {fg:.6f} | {bg:.6f} |".format(
            name=name,
            mad=values["mad_vs_reference"],
            edge="N/A" if values["edge_leakage"] is None else f"{values['edge_leakage']:.6f}",
            width="N/A" if values["edge_width_pixels"] is None else f"{values['edge_width_pixels']:.6f}",
            fg=values["foreground_mean"],
            bg=values["background_mean"],
        )
        for name, values in metrics.items()
    )
    gradient_text = ""
    if gradient_check is not None:
        gradient_text = (
            f"\nSoft gradient check: backward_steps={gradient_check['backward_steps']}, "
            f"large_jumps={gradient_check['large_jumps']}. {gradient_check['note']}\n"
        )

    return f"""# Model D Shape Benchmark: {shape}

## Question

Model Dは `{shape}` の16x16 low-resolution guideから、64x64出力で境界または階調を妥当に扱えるか。

## Hypothesis

hard edge shapeではconfidence mapとedge-aware interactionにより、bilinear/bicubic baselineと同程度以上に境界を保つ可能性がある。soft gradientではedge leakageではなく、列平均の逆行や急な段差が少ないことを確認する。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_005_model_d_shape_benchmark.py`
- Date: {config["date"]}
- Shape: {shape}
- Experiment seed: {config["experiment_seed"]}
- Decoder seed: {config["decoder_seed"]}
- Low guide size: {config["low_size"]}x{config["low_size"]}
- Output size: {config["high_size"]}x{config["high_size"]}
- Model: Model D candidate
- Model config: `{config["model_d_params"]}`
- Decode config: `{config["decode_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

baselineはnearest、bilinear、bicubicの3種類のlow-resolution guide upscalingとした。metricsのreferenceは同じsynthetic shapeを64x64で生成した比較用参照であり、実画像のGround Truthではない。

## Result

| Output | MAD vs synthetic reference | Edge leakage | Edge width pixels | Foreground mean | Background mean |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

Decode time seconds: {config["decode_time_seconds"]:.6f}

Mask note: {mask_note}
{gradient_text}
## Images

![Comparison of low guide, nearest, bilinear, bicubic, confidence, Model D, and difference](comparison.png)

![Low-resolution guide](low_guide.png)

![Confidence map](confidence.png)

![Model D rendered output](rendered_model_d.png)

![Absolute difference between Model D and bilinear](diff_model_d_vs_bilinear.png)

## Interpretation

この結果はsynthetic guide上の比較であり、Model Dの一般的な超解像性能を示すものではない。hard edge shapeではedge leakageとedge width、soft gradientでは階調の連続性メモを中心に読む。

## Limitations

- 比較用referenceはsyntheticに生成した高解像度shapeであり、実画像のGround Truthではない。
- Model D候補はPython/NumPy実装で、環境非依存のbit-perfect再現性は未確認。
- texture termは白色ノイズに近く、意味的ディテールや自然な質感を生成するものではない。
- decode timeはこの環境の小画像runに限る。

## Next

- Issue #6 でcrossを含むbilinear/bicubic比較指標との整合を確認する。
- Issue #14 でguided filter / guided upsamplingとの位置づけを調査する。
"""


def make_summary_notes(
    config: dict[str, object],
    shape_results: list[dict[str, object]],
) -> str:
    rows = "\n".join(
        "| {shape} | {model_mad:.6f} | {bilinear_mad:.6f} | {model_edge} | {model_width} | {time:.6f} |".format(
            shape=row["shape"],
            model_mad=row["metrics"]["model_d"]["mad_vs_reference"],
            bilinear_mad=row["metrics"]["bilinear"]["mad_vs_reference"],
            model_edge="N/A"
            if row["metrics"]["model_d"]["edge_leakage"] is None
            else f"{row['metrics']['model_d']['edge_leakage']:.6f}",
            model_width="N/A"
            if row["metrics"]["model_d"]["edge_width_pixels"] is None
            else f"{row['metrics']['model_d']['edge_width_pixels']:.6f}",
            time=row["decode_time_seconds"],
        )
        for row in shape_results
    )
    return f"""# Model D Shape Benchmark

## Question

Model Dがcross以外の低解像度guideでも境界と階調を妥当に扱えるかを、保存形式つきbenchmarkとして確認する。

## Hypothesis

diagonal line、circle、thin lineではconfidence mapとedge-aware interactionが境界付近の崩れを抑える可能性がある。soft gradientでは、confidence mapが階調を硬く分断しないことを確認する必要がある。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_005_model_d_shape_benchmark.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Decoder seed base: {config["decoder_seed_base"]}
- Low guide size: {config["low_size"]}x{config["low_size"]}
- Output size: {config["high_size"]}x{config["high_size"]}
- Shapes: diagonal, circle, thin_line, soft_gradient
- Model: Model D candidate
- Model config: `{config["model_d_params"]}`

## Baseline

baselineはnearest、bilinear、bicubic upscaling。metricsのreferenceは同じsynthetic shapeを64x64で生成した比較用参照であり、実画像のGround Truthではない。

## Result

| Shape | Model D MAD | Bilinear MAD | Model D edge leakage | Model D edge width | Decode time seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Notes: `notes.md`
- Per-shape artifacts: `<shape>/low_guide.png`, `<shape>/nearest.png`, `<shape>/bilinear.png`, `<shape>/bicubic.png`, `<shape>/confidence.png`, `<shape>/rendered_model_d.png`, `<shape>/diff_model_d_vs_bilinear.png`, `<shape>/comparison.png`, `<shape>/config.json`, `<shape>/metrics.json`, `<shape>/notes.md`

## Interpretation

このbenchmarkはModel D候補をshape coverageの観点で記録するためのもの。結果はsynthetic guideに限定され、実用圧縮性能、一般画像品質、または超解像性能を断定するものではない。

## Limitations

- 実画像パッチでは未検証。
- white-noise texture termのため、質感は意味的ディテールではない。
- Python/NumPy実装の結果であり、Rust固定小数点やbit-perfect再現性は未確認。
- Issue #6 の比較指標とは矛盾しない形で保存したが、crossを含む統一比較は別Issueで扱う。

## Next

- Issue #6 でModel Dとbilinear/bicubicの統一比較指標を整理する。
- Issue #14 でguided filter / guided upsamplingとの比較観点を調査する。
"""


def run_shape(
    index: int,
    shape: str,
    low_factory: Callable[[], np.ndarray],
    high_factory: Callable[[], np.ndarray],
    model_params: dict[str, float],
    decode_config: dict[str, float | int],
) -> dict[str, object]:
    shape_dir = ensure_dir(RESULT_DIR / shape)
    low_guide = low_factory()
    high_reference = high_factory()
    nearest = upscale(low_guide, HIGH_SIZE, order=0)
    bilinear = upscale(low_guide, HIGH_SIZE, order=1)
    bicubic = upscale(low_guide, HIGH_SIZE, order=3)
    confidence = gradient_confidence(bilinear, min_confidence=0.2, max_confidence=1.0, scale=4.0)
    texture = seeded_texture(bilinear.shape, seed=EXPERIMENT_SEED + index)
    decoder_seed = DECODER_SEED_BASE + index

    start = time.perf_counter()
    rendered = model_d_decode(
        bilinear,
        confidence,
        texture,
        decoder_seed=decoder_seed,
        sweeps=int(decode_config["sweeps"]),
        temp_start=float(decode_config["temp_start"]),
        temp_end=float(decode_config["temp_end"]),
        proposal_sigma=float(decode_config["proposal_sigma"]),
        **model_params,
    )
    decode_time = time.perf_counter() - start

    foreground_mask, mask_note = masks_for_shape(shape, high_reference)
    hard_edge = shape != "soft_gradient"
    outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
        "model_d": rendered,
    }
    metrics = metrics_for_outputs(outputs, high_reference, foreground_mask, hard_edge=hard_edge)
    gradient_check = gradient_naturalness(rendered) if shape == "soft_gradient" else None

    shape_config = {
        "date": DATE,
        "shape": shape,
        "experiment_seed": EXPERIMENT_SEED + index,
        "decoder_seed": decoder_seed,
        "low_size": LOW_SIZE,
        "high_size": HIGH_SIZE,
        "model": "Model D candidate",
        "model_d_params": model_params,
        "decode_config": decode_config,
        "decode_time_seconds": float(decode_time),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }

    diff = np.abs(rendered - bilinear)
    save_json(shape_dir / "config.json", shape_config)
    save_json(
        shape_dir / "metrics.json",
        {"outputs": metrics, "gradient_check": gradient_check, "mask_note": mask_note},
    )
    save_grayscale_png(shape_dir / "low_guide.png", low_guide)
    save_grayscale_png(shape_dir / "high_reference.png", high_reference)
    save_grayscale_png(shape_dir / "nearest.png", nearest)
    save_grayscale_png(shape_dir / "bilinear.png", bilinear)
    save_grayscale_png(shape_dir / "bicubic.png", bicubic)
    save_grayscale_png(shape_dir / "confidence.png", confidence)
    save_grayscale_png(shape_dir / "rendered_model_d.png", rendered)
    save_grayscale_png(shape_dir / "diff_model_d_vs_bilinear.png", diff)
    save_comparison_png(
        shape_dir / "comparison.png",
        [
            upscale(low_guide, HIGH_SIZE, order=0),
            nearest,
            bilinear,
            bicubic,
            confidence,
            rendered,
            diff,
        ],
    )
    (shape_dir / "notes.md").write_text(
        make_shape_notes(shape, shape_config, metrics, mask_note, gradient_check),
        encoding="utf-8",
    )
    return {
        "shape": shape,
        "decode_time_seconds": float(decode_time),
        "metrics": metrics,
        "gradient_check": gradient_check,
        "mask_note": mask_note,
    }


def main() -> None:
    ensure_dir(RESULT_DIR)
    model_params = {
        "j_base": 1.8,
        "lambda_data": 6.0,
        "gamma": 35.0,
        "texture_weight": 0.35,
    }
    decode_config: dict[str, float | int] = {
        "sweeps": 35,
        "temp_start": 0.35,
        "temp_end": 0.01,
        "proposal_sigma": 0.08,
    }
    shapes: list[tuple[str, Callable[[], np.ndarray], Callable[[], np.ndarray]]] = [
        ("diagonal", lambda: diagonal(size=LOW_SIZE, width=1, value=0.5), lambda: diagonal(size=HIGH_SIZE, width=4, value=0.5)),
        ("circle", lambda: circle(size=LOW_SIZE, radius=4.0, value=0.5), lambda: circle(size=HIGH_SIZE, radius=16.0, value=0.5)),
        ("thin_line", lambda: thin_line(size=LOW_SIZE, value=0.5), lambda: thin_line(size=HIGH_SIZE, value=0.5)),
        ("soft_gradient", lambda: horizontal_gradient(size=LOW_SIZE), lambda: horizontal_gradient(size=HIGH_SIZE)),
    ]
    shape_results = [
        run_shape(index, shape, low_factory, high_factory, model_params, decode_config)
        for index, (shape, low_factory, high_factory) in enumerate(shapes)
    ]
    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "decoder_seed_base": DECODER_SEED_BASE,
        "low_size": LOW_SIZE,
        "high_size": HIGH_SIZE,
        "model": "Model D candidate",
        "model_d_params": model_params,
        "decode_config": decode_config,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", {"shapes": shape_results})
    (RESULT_DIR / "notes.md").write_text(make_summary_notes(config, shape_results), encoding="utf-8")
    for row in shape_results:
        model_d = row["metrics"]["model_d"]
        edge = "N/A" if model_d["edge_leakage"] is None else f"{model_d['edge_leakage']:.6f}"
        print(
            f"{row['shape']}: model_d_mad={model_d['mad_vs_reference']:.6f} "
            f"edge={edge} decode_time={row['decode_time_seconds']:.3f}s"
        )


if __name__ == "__main__":
    main()
