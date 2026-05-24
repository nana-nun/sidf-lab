"""Compare Model C and Perona-Malik style diffusion over several shapes."""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from sidf_lab.anneal import AnnealConfig, model_c_decode
from sidf_lab.energy import ModelCParams
from sidf_lab.guides import add_noise, circle, horizontal_gradient
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json
from sidf_lab.metrics import edge_leakage, mad, psnr, region_summary, ssim_global


RESULT_DIR = Path("results/2026-05-24-issue-64-model-c-perona-malik-shapes")
DATE = "2026-05-24"
EXPERIMENT_SEED = 20260524
DECODER_SEED_BASE = 6464
SIZE = 48
LOW_VALUE = 0.08
HIGH_VALUE = 0.62
NOISE_SIGMA = 0.045


def diagonal_step(size: int) -> np.ndarray:
    rows, cols = np.indices((size, size))
    image = np.full((size, size), LOW_VALUE, dtype=np.float64)
    image[cols >= rows] = HIGH_VALUE
    return image


def circle_step(size: int) -> np.ndarray:
    base = np.full((size, size), LOW_VALUE, dtype=np.float64)
    mask = circle(size, radius=size * 0.25, value=1.0) > 0.0
    base[mask] = HIGH_VALUE
    return base


def soft_gradient(size: int) -> np.ndarray:
    return LOW_VALUE + (HIGH_VALUE - LOW_VALUE) * horizontal_gradient(size)


def hard_edge_mask(shape_name: str, clean: np.ndarray) -> np.ndarray | None:
    if shape_name == "soft_gradient":
        return None
    return clean > (LOW_VALUE + HIGH_VALUE) / 2.0


def horizontal_pair_values(image: np.ndarray, scale: float, gamma: float) -> np.ndarray:
    diff = np.asarray(image, dtype=np.float64)[:, 1:] - np.asarray(image, dtype=np.float64)[:, :-1]
    return scale * np.exp(-gamma * (diff**2))


def vertical_pair_values(image: np.ndarray, scale: float, gamma: float) -> np.ndarray:
    diff = np.asarray(image, dtype=np.float64)[1:, :] - np.asarray(image, dtype=np.float64)[:-1, :]
    return scale * np.exp(-gamma * (diff**2))


def horizontal_conductance(image: np.ndarray, kappa: float) -> np.ndarray:
    diff = np.asarray(image, dtype=np.float64)[:, 1:] - np.asarray(image, dtype=np.float64)[:, :-1]
    return np.exp(-((diff / kappa) ** 2))


def vertical_conductance(image: np.ndarray, kappa: float) -> np.ndarray:
    diff = np.asarray(image, dtype=np.float64)[1:, :] - np.asarray(image, dtype=np.float64)[:-1, :]
    return np.exp(-((diff / kappa) ** 2))


def pair_display(horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    h_map = np.pad(horizontal, ((0, 0), (0, 1)), mode="edge")
    v_map = np.pad(vertical, ((0, 1), (0, 0)), mode="edge")
    return 0.5 * (h_map + v_map)


def boundary_pair_stats(horizontal: np.ndarray, vertical: np.ndarray, mask: np.ndarray | None) -> dict[str, float | None]:
    if mask is None:
        values = np.concatenate([horizontal.ravel(), vertical.ravel()])
        return {
            "flat_pair_mean": float(values.mean()),
            "boundary_pair_mean": None,
            "boundary_to_flat_ratio": None,
        }

    h_boundary = mask[:, 1:] != mask[:, :-1]
    v_boundary = mask[1:, :] != mask[:-1, :]
    h_flat = ~h_boundary
    v_flat = ~v_boundary
    boundary_values = np.concatenate([horizontal[h_boundary], vertical[v_boundary]])
    flat_values = np.concatenate([horizontal[h_flat], vertical[v_flat]])
    boundary_mean = float(boundary_values.mean()) if boundary_values.size else None
    flat_mean = float(flat_values.mean()) if flat_values.size else None
    ratio = None if boundary_mean is None or flat_mean is None or flat_mean == 0.0 else boundary_mean / flat_mean
    return {
        "flat_pair_mean": flat_mean,
        "boundary_pair_mean": boundary_mean,
        "boundary_to_flat_ratio": None if ratio is None else float(ratio),
    }


def perona_malik_diffuse(initial: np.ndarray, steps: int, dt: float, kappa: float) -> np.ndarray:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 < dt <= 0.25:
        raise ValueError("dt must be in (0, 0.25]")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")

    state = np.asarray(initial, dtype=np.float64).copy()
    for _ in range(steps):
        north = np.zeros_like(state)
        south = np.zeros_like(state)
        west = np.zeros_like(state)
        east = np.zeros_like(state)
        north[1:, :] = state[:-1, :] - state[1:, :]
        south[:-1, :] = state[1:, :] - state[:-1, :]
        west[:, 1:] = state[:, :-1] - state[:, 1:]
        east[:, :-1] = state[:, 1:] - state[:, :-1]
        update = (
            np.exp(-((north / kappa) ** 2)) * north
            + np.exp(-((south / kappa) ** 2)) * south
            + np.exp(-((west / kappa) ** 2)) * west
            + np.exp(-((east / kappa) ** 2)) * east
        )
        state = np.clip(state + dt * update, 0.0, 1.0)
    return state


def gradient_mad(candidate: np.ndarray, reference: np.ndarray) -> float:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    cand_grad = np.hypot(np.diff(cand, axis=1, append=cand[:, -1:]), np.diff(cand, axis=0, append=cand[-1:, :]))
    ref_grad = np.hypot(np.diff(ref, axis=1, append=ref[:, -1:]), np.diff(ref, axis=0, append=ref[-1:, :]))
    return mad(cand_grad, ref_grad)


def soft_gradient_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    row_mean = np.asarray(candidate, dtype=np.float64).mean(axis=0)
    ref_mean = np.asarray(reference, dtype=np.float64).mean(axis=0)
    slope = np.diff(row_mean)
    ref_slope = np.diff(ref_mean)
    second_diff = np.diff(row_mean, n=2)
    return {
        "backward_steps_lt_minus_0_005": int(np.sum(slope < -0.005)),
        "slope_mae_vs_clean": float(np.mean(np.abs(slope - ref_slope))),
        "mean_abs_second_diff": float(np.mean(np.abs(second_diff))),
        "max_abs_second_diff": float(np.max(np.abs(second_diff))),
    }


def metrics_for(name: str, image: np.ndarray, clean: np.ndarray, mask: np.ndarray | None) -> dict[str, object]:
    row: dict[str, object] = {
        "name": name,
        "mad_vs_clean": mad(image, clean),
        "psnr_vs_clean": psnr(clean, image),
        "ssim_global_vs_clean": ssim_global(clean, image),
        "gradient_mad_vs_clean": gradient_mad(image, clean),
    }
    if mask is None:
        row["edge_leakage"] = None
        row.update(soft_gradient_metrics(image, clean))
    else:
        foreground = region_summary(image, mask)
        background = region_summary(image, ~mask)
        row.update(
            {
                "foreground_mean": foreground["mean"],
                "background_mean": background["mean"],
                "foreground_variance": foreground["variance"],
                "background_variance": background["variance"],
                "edge_leakage": edge_leakage(image, mask, radius=2),
            }
        )
    return row


def save_comparison_png(path: str | Path, images: list[np.ndarray], gap: int = 2) -> None:
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


def fmt_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def make_shape_notes(shape_name: str, config: dict[str, object], metrics: dict[str, object]) -> str:
    output_rows = "\n".join(
        "| {name} | {mad} | {psnr} | {ssim} | {grad} | {edge} | {time} |".format(
            name=row["name"],
            mad=fmt_float(row["mad_vs_clean"]),
            psnr=fmt_float(row["psnr_vs_clean"], 3),
            ssim=fmt_float(row["ssim_global_vs_clean"]),
            grad=fmt_float(row["gradient_mad_vs_clean"]),
            edge=fmt_float(row.get("edge_leakage")),
            time=fmt_float(row.get("decode_time_seconds")),
        )
        for row in metrics["outputs"]
    )
    pair_rows = "\n".join(
        "| {name} | {flat} | {boundary} | {ratio} |".format(
            name=row["name"],
            flat=fmt_float(row["flat_pair_mean"]),
            boundary=fmt_float(row["boundary_pair_mean"]),
            ratio=fmt_float(row["boundary_to_flat_ratio"]),
        )
        for row in metrics["pair_weights"]
    )
    soft_rows = ""
    if shape_name == "soft_gradient":
        soft_rows = "\n".join(
            "| {name} | {backward} | {slope} | {second} | {max_second} |".format(
                name=row["name"],
                backward=row["backward_steps_lt_minus_0_005"],
                slope=fmt_float(row["slope_mae_vs_clean"]),
                second=fmt_float(row["mean_abs_second_diff"]),
                max_second=fmt_float(row["max_abs_second_diff"]),
            )
            for row in metrics["outputs"]
        )
        soft_rows = f"""
## Soft Gradient Alternative Metrics

soft gradient は明確な foreground/background 境界を持たないため、edge leakage は不適用とした。代替として、列平均の逆行数、clean gradient に対する slope error、二階差分の大きさを保存した。

| Output | Backward steps | Slope MAE vs clean | Mean abs second diff | Max abs second diff |
| --- | ---: | ---: | ---: | ---: |
{soft_rows}
"""

    return f"""# Model C と Perona-Malik 型 diffusion の形状別比較: {shape_name}

## Question

Model C の guide-derived fixed weight と Perona-Malik 型 diffusion の state-derived conductance の違いは、`{shape_name}` でも確認できるか。

## Hypothesis

hard edge shape では、両者とも大きな局所差をまたぐ混合を弱めるが、Model C は noisy guide から固定weightを作り、Perona-Malik 型 diffusion は現在状態から conductance を更新するため、weight / conductance map の読み方は一致しない。soft gradient では明確な境界がないため、edge leakage ではなく階調の単調性と滑らかさで確認する。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_013_model_c_perona_malik_shapes.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Decoder seed: {config["decoder_seed"]}
- Input: {config["size"]}x{config["size"]} synthetic `{shape_name}` with deterministic Gaussian noise
- Model C params: `{config["model_c_params"]}`
- Model C anneal config: `{config["anneal_config"]}`
- Perona-Malik config: `{config["perona_malik_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

Baseline は noisy initial guide をそのまま表示する `initial_noisy.png` とした。Perona-Malik 型 diffusion は、この noisy initial から画像勾配ベースの conductance で明示的に反復更新する比較対象である。

## Metrics

| Output | MAD vs clean | PSNR vs clean | Global SSIM vs clean | Gradient MAD vs clean | Edge leakage | Decode time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{output_rows}
{soft_rows}
## Pair Weight / Conductance Summary

hard edge shape では `boundary_pair_mean` を true mask をまたぐ4近傍pair、`flat_pair_mean` をそれ以外のpairとして計算した。soft gradient では明確な境界がないため、boundary 系は `N/A` とし、全pair平均だけを保存した。

| Map | Flat pair mean | Boundary pair mean | Boundary / flat |
| --- | ---: | ---: | ---: |
{pair_rows}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Clean guide: `clean_guide.png`
- Initial noisy guide: `initial_noisy.png`
- Model C output: `model_c.png`
- Perona-Malik output: `perona_malik.png`
- Model C guide weight map: `model_c_weight_map.png`
- Perona-Malik initial conductance map: `pm_conductance_initial_map.png`
- Perona-Malik final conductance map: `pm_conductance_final_map.png`
- Difference maps: `diff_model_c_vs_clean.png`, `diff_pm_vs_clean.png`
- Comparison strip: `comparison.png`

## Images

![Comparison of clean, noisy initial, Model C, Perona-Malik, and difference maps](comparison.png)

![Model C guide-derived pair weights](model_c_weight_map.png)

![Perona-Malik initial conductance](pm_conductance_initial_map.png)

![Perona-Malik final conductance](pm_conductance_final_map.png)

## Result

`{shape_name}` について、Model C と Perona-Malik 型 diffusion の出力、差分、guide-derived weight、state-derived conductance を保存した。

## Interpretation

この形状でも、Model C のweightは入力guideから固定的に決まり、Perona-Malik 型のconductanceは初期状態とdiffusion後で変化するものとして確認できる。metricsの良し悪しは、この小規模synthetic条件での比較結果として扱い、一般的な優位性とは解釈しない。

## Limitations

- synthetic `{shape_name}` 1条件の小規模比較であり、一般画像品質を示さない。
- Perona-Malik 実装は最小の明示的diffusion loopであり、元論文の数値解析や安定性検討を網羅しない。
- Model C は stochastic relaxation なので、seed、sweeps、temperature schedule に依存する。
- この結果は compression、super-resolution、Model C の一般的優位性を示すものではない。

## Next

- Perona-Malik 型以外の guided filtering / anisotropic smoothing baseline と比較する場合は、低解像度guide条件と高解像度guidance条件を分ける。
- Model C freeze criteria として使う場合は、shape別の合格目安を別途定義する。
"""


def make_summary_notes(config: dict[str, object], summary: list[dict[str, object]]) -> str:
    rows = "\n".join(
        "| {shape} | {model_mad} | {pm_mad} | {model_edge} | {pm_edge} | {model_time} | {pm_time} |".format(
            shape=row["shape"],
            model_mad=fmt_float(row["model_c_mad_vs_clean"]),
            pm_mad=fmt_float(row["perona_malik_mad_vs_clean"]),
            model_edge=fmt_float(row["model_c_edge_leakage"]),
            pm_edge=fmt_float(row["perona_malik_edge_leakage"]),
            model_time=fmt_float(row["model_c_decode_time_seconds"]),
            pm_time=fmt_float(row["perona_malik_decode_time_seconds"]),
        )
        for row in summary
    )
    return f"""# Model C と Perona-Malik 型 diffusion の複数shape比較

## Question

Issue #40 の vertical edge 比較で確認した Model C と Perona-Malik 型 diffusion の類似点と相違点は、diagonal、circle、soft gradient でも同じように確認できるか。

## Hypothesis

Model C と Perona-Malik 型 diffusion は、どちらも大きな局所差をまたぐ混合を弱める点で似ている。ただし Model C は guide-derived fixed weight と data fidelity を持つ stochastic relaxation であり、Perona-Malik 型 diffusion は state-derived conductance による deterministic diffusion なので、形状を増やしても同等の方法とは扱わない。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_013_model_c_perona_malik_shapes.py`
- Date: {config["date"]}
- Experiment seed base: {config["experiment_seed_base"]}
- Decoder seed base: {config["decoder_seed_base"]}
- Input size: {config["size"]}x{config["size"]}
- Shapes: diagonal, circle, soft_gradient
- Model C params: `{config["model_c_params"]}`
- Model C anneal config: `{config["anneal_config"]}`
- Perona-Malik config: `{config["perona_malik_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

各shapeの baseline は noisy initial guide をそのまま表示する `initial_noisy.png` とした。Perona-Malik 型 diffusion は、この noisy initial から画像勾配ベースの conductance で明示的に反復更新する比較対象である。

## Result

| Shape | Model C MAD | Perona-Malik MAD | Model C edge leakage | Perona-Malik edge leakage | Model C seconds | Perona-Malik seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

soft gradient は明確な foreground/background 境界を持たないため、edge leakage は `N/A` とし、各shapeの `notes.md` に列平均の逆行数、slope error、二階差分を保存した。

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Per-shape artifacts: `<shape>/config.json`, `<shape>/metrics.json`, `<shape>/notes.md`, `<shape>/*.png`

## Images

各shapeの `notes.md` に主要PNGへのMarkdown画像参照を保存した。

## Interpretation

diagonal、circle、soft gradient へ広げても、Model C は guide から固定的に pair weight を決め、Perona-Malik 型 diffusion は現在画像状態から conductance を決めるという違いを図とmetricsで確認できる。両者は edge-aware な混合抑制という役割では似るが、係数決定元、更新過程、data fidelity の有無が異なるため、同じ方法または一般的な優劣として扱わない。

## Limitations

- synthetic 3形状の小規模比較であり、一般画像品質を示さない。
- Perona-Malik 実装は最小の明示的diffusion loopであり、元論文の数値解析や安定性検討を網羅しない。
- Model C は seed、sweeps、temperature schedule に依存する。
- soft gradient の代替指標は階調の基本確認であり、知覚品質を保証しない。
- この結果は compression、super-resolution、Model C の一般的優位性を示すものではない。

## Next

- shape別の合格目安が必要なら、Model C freeze criteria 側で別Issueとして整理する。
- Perona-Malik 型以外の guided filtering / anisotropic smoothing baseline と比較する場合は、低解像度guide条件と高解像度guidance条件を分ける。
"""


def run_shape(
    index: int,
    name: str,
    factory: Callable[[int], np.ndarray],
    params: ModelCParams,
    anneal_template: AnnealConfig,
    pm_config: dict[str, float | int],
) -> dict[str, object]:
    shape_dir = ensure_dir(RESULT_DIR / name)
    clean = factory(SIZE)
    initial = add_noise(clean, seed=EXPERIMENT_SEED + index, sigma=NOISE_SIGMA)
    mask = hard_edge_mask(name, clean)

    anneal_config = AnnealConfig(
        decoder_seed=DECODER_SEED_BASE + index,
        sweeps=anneal_template.sweeps,
        temp_start=anneal_template.temp_start,
        temp_end=anneal_template.temp_end,
        proposal_sigma=anneal_template.proposal_sigma,
    )

    start = time.perf_counter()
    model_c = model_c_decode(initial, params, anneal_config)
    model_c_time = time.perf_counter() - start

    start = time.perf_counter()
    pm = perona_malik_diffuse(
        initial,
        steps=int(pm_config["steps"]),
        dt=float(pm_config["dt"]),
        kappa=float(pm_config["kappa"]),
    )
    pm_time = time.perf_counter() - start

    model_h = horizontal_pair_values(initial, params.j_base, params.gamma)
    model_v = vertical_pair_values(initial, params.j_base, params.gamma)
    pm_initial_h = horizontal_conductance(initial, float(pm_config["kappa"]))
    pm_initial_v = vertical_conductance(initial, float(pm_config["kappa"]))
    pm_final_h = horizontal_conductance(pm, float(pm_config["kappa"]))
    pm_final_v = vertical_conductance(pm, float(pm_config["kappa"]))

    outputs = [
        metrics_for("initial_noisy", initial, clean, mask),
        metrics_for("model_c", model_c, clean, mask),
        metrics_for("perona_malik", pm, clean, mask),
    ]
    outputs[1]["decode_time_seconds"] = float(model_c_time)
    outputs[2]["decode_time_seconds"] = float(pm_time)
    pair_weights = [
        {"name": "model_c_guide_weight", **boundary_pair_stats(model_h, model_v, mask)},
        {"name": "pm_initial_conductance", **boundary_pair_stats(pm_initial_h, pm_initial_v, mask)},
        {"name": "pm_final_conductance", **boundary_pair_stats(pm_final_h, pm_final_v, mask)},
    ]

    config = {
        "date": DATE,
        "shape": name,
        "experiment_seed": EXPERIMENT_SEED + index,
        "decoder_seed": DECODER_SEED_BASE + index,
        "size": SIZE,
        "low_value": LOW_VALUE,
        "high_value": HIGH_VALUE,
        "noise_sigma": NOISE_SIGMA,
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
        "perona_malik_config": pm_config,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    metrics = {"outputs": outputs, "pair_weights": pair_weights}

    save_json(shape_dir / "config.json", config)
    save_json(shape_dir / "metrics.json", metrics)
    save_grayscale_png(shape_dir / "clean_guide.png", clean)
    save_grayscale_png(shape_dir / "initial_noisy.png", initial)
    save_grayscale_png(shape_dir / "model_c.png", model_c)
    save_grayscale_png(shape_dir / "perona_malik.png", pm)
    save_grayscale_png(shape_dir / "diff_model_c_vs_clean.png", np.abs(model_c - clean))
    save_grayscale_png(shape_dir / "diff_pm_vs_clean.png", np.abs(pm - clean))
    save_grayscale_png(shape_dir / "model_c_weight_map.png", pair_display(model_h / params.j_base, model_v / params.j_base))
    save_grayscale_png(shape_dir / "pm_conductance_initial_map.png", pair_display(pm_initial_h, pm_initial_v))
    save_grayscale_png(shape_dir / "pm_conductance_final_map.png", pair_display(pm_final_h, pm_final_v))
    save_comparison_png(
        shape_dir / "comparison.png",
        [clean, initial, model_c, pm, np.abs(model_c - clean), np.abs(pm - clean)],
    )
    (shape_dir / "notes.md").write_text(make_shape_notes(name, config, metrics), encoding="utf-8")

    model_c_row = outputs[1]
    pm_row = outputs[2]
    return {
        "shape": name,
        "model_c_mad_vs_clean": model_c_row["mad_vs_clean"],
        "perona_malik_mad_vs_clean": pm_row["mad_vs_clean"],
        "model_c_edge_leakage": model_c_row["edge_leakage"],
        "perona_malik_edge_leakage": pm_row["edge_leakage"],
        "model_c_decode_time_seconds": model_c_time,
        "perona_malik_decode_time_seconds": pm_time,
    }


def main() -> None:
    ensure_dir(RESULT_DIR)
    params = ModelCParams(j_base=2.0, lambda_data=5.0, gamma=40.0)
    anneal_template = AnnealConfig(
        decoder_seed=DECODER_SEED_BASE,
        sweeps=60,
        temp_start=0.45,
        temp_end=0.01,
        proposal_sigma=0.09,
    )
    pm_config = {"steps": 60, "dt": 0.18, "kappa": 0.11}
    shapes: list[tuple[str, Callable[[int], np.ndarray]]] = [
        ("diagonal", diagonal_step),
        ("circle", circle_step),
        ("soft_gradient", soft_gradient),
    ]
    summary = [
        run_shape(index, name, factory, params, anneal_template, pm_config)
        for index, (name, factory) in enumerate(shapes)
    ]
    config = {
        "date": DATE,
        "experiment_seed_base": EXPERIMENT_SEED,
        "decoder_seed_base": DECODER_SEED_BASE,
        "size": SIZE,
        "low_value": LOW_VALUE,
        "high_value": HIGH_VALUE,
        "noise_sigma": NOISE_SIGMA,
        "shapes": [name for name, _ in shapes],
        "model_c_params": {
            "j_base": params.j_base,
            "lambda_data": params.lambda_data,
            "gamma": params.gamma,
        },
        "anneal_config": {
            "sweeps": anneal_template.sweeps,
            "temp_start": anneal_template.temp_start,
            "temp_end": anneal_template.temp_end,
            "proposal_sigma": anneal_template.proposal_sigma,
        },
        "perona_malik_config": pm_config,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", {"shapes": summary})
    (RESULT_DIR / "notes.md").write_text(make_summary_notes(config, summary), encoding="utf-8")

    for row in summary:
        print(
            f"{row['shape']}: model_c_mad={row['model_c_mad_vs_clean']:.6f} "
            f"pm_mad={row['perona_malik_mad_vs_clean']:.6f} "
            f"model_c_edge={fmt_float(row['model_c_edge_leakage'])} "
            f"pm_edge={fmt_float(row['perona_malik_edge_leakage'])}"
        )


if __name__ == "__main__":
    main()
