"""Compare Model C guide-weighting with Perona-Malik style diffusion."""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np

from sidf_lab.anneal import AnnealConfig, model_c_decode
from sidf_lab.energy import ModelCParams
from sidf_lab.guides import add_noise
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json
from sidf_lab.metrics import edge_leakage, mad, psnr, region_summary, ssim_global


RESULT_DIR = Path("results/2026-05-23-model-c-perona-malik")
DATE = "2026-05-23"
EXPERIMENT_SEED = 20260523
DECODER_SEED = 6400
SIZE = 48
LEFT_VALUE = 0.08
RIGHT_VALUE = 0.62
NOISE_SIGMA = 0.045


def vertical_step(size: int) -> np.ndarray:
    """Return a synthetic vertical edge with two constant regions."""
    image = np.full((size, size), LEFT_VALUE, dtype=np.float64)
    image[:, size // 2 :] = RIGHT_VALUE
    return image


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


def horizontal_conductance(image: np.ndarray, kappa: float) -> np.ndarray:
    """Return Perona-Malik conductance for horizontal neighbor pairs."""
    values = np.asarray(image, dtype=np.float64)
    diff = values[:, 1:] - values[:, :-1]
    return np.exp(-((diff / kappa) ** 2))


def model_c_horizontal_weight(guide: np.ndarray, params: ModelCParams) -> np.ndarray:
    """Return Model C guide-derived weights for horizontal neighbor pairs."""
    values = np.asarray(guide, dtype=np.float64)
    diff = values[:, 1:] - values[:, :-1]
    return params.j_base * np.exp(-params.gamma * (diff**2))


def pad_pair_map(pair_map: np.ndarray) -> np.ndarray:
    """Pad an H x (W - 1) pair map to H x W for PNG output."""
    return np.pad(pair_map, ((0, 0), (0, 1)), mode="edge")


def perona_malik_diffuse(initial: np.ndarray, steps: int, dt: float, kappa: float) -> np.ndarray:
    """Run a small explicit Perona-Malik style anisotropic diffusion loop."""
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


def boundary_pair_mask(size: int) -> np.ndarray:
    """Mask horizontal pairs that cross the true vertical step boundary."""
    mask = np.zeros((size, size - 1), dtype=bool)
    mask[:, size // 2 - 1] = True
    return mask


def metrics_for(name: str, image: np.ndarray, clean: np.ndarray, foreground: np.ndarray) -> dict[str, float | str]:
    left = region_summary(image, ~foreground)
    right = region_summary(image, foreground)
    return {
        "name": name,
        "mad_vs_clean": mad(image, clean),
        "psnr_vs_clean": psnr(clean, image),
        "ssim_global_vs_clean": ssim_global(clean, image),
        "left_mean": left["mean"],
        "right_mean": right["mean"],
        "left_variance": left["variance"],
        "right_variance": right["variance"],
        "edge_leakage": edge_leakage(image, foreground, radius=2),
    }


def make_notes(config: dict[str, object], metrics: dict[str, object]) -> str:
    rows = "\n".join(
        "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {left:.6f} | {right:.6f} | {edge:.6f} |".format(
            name=row["name"],
            mad=row["mad_vs_clean"],
            psnr=row["psnr_vs_clean"],
            ssim=row["ssim_global_vs_clean"],
            left=row["left_mean"],
            right=row["right_mean"],
            edge=row["edge_leakage"],
        )
        for row in metrics["outputs"]
    )
    weight_rows = "\n".join(
        "| {name} | {flat:.6f} | {boundary:.6f} |".format(
            name=row["name"],
            flat=row["flat_pair_mean"],
            boundary=row["boundary_pair_mean"],
        )
        for row in metrics["pair_weights"]
    )
    return f"""# Model C と Perona-Malik 型 diffusion の最小比較

## Question

Model C の guide 差分ベース edge-aware weighting と、Perona-Malik 型の画像勾配ベース diffusion は、同じ synthetic vertical edge で係数決定元と出力の読み方がどう違うか。

## Hypothesis

Model C の近傍重みは guide `s` から固定的に決まり、decoder state のノイズには直接追従しない。一方、Perona-Malik 型 diffusion の conductance は現在の画像 `u` の勾配から各stepで決まり、初期ノイズや拡散後の状態に応じて変わる。そのため、両者は「エッジをまたぐ混合を弱める」という類似点を持つが、同等の処理ではない。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_009_model_c_perona_malik.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Decoder seed: {config["decoder_seed"]}
- Input: {config["size"]}x{config["size"]} synthetic vertical step with deterministic Gaussian noise
- Model C params: `{config["model_c_params"]}`
- Model C anneal config: `{config["anneal_config"]}`
- Perona-Malik config: `{config["perona_malik_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

Baseline は noisy initial guide をそのまま表示する `initial_noisy.png` とした。Perona-Malik 型 diffusion は、この noisy initial から画像勾配ベースの conductance で明示的に反復更新する比較対象である。

## Metrics

| Output | MAD vs clean | PSNR vs clean | Global SSIM vs clean | Left mean | Right mean | Edge leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

## Pair Weight Summary

`flat_pair_mean` は真の縦境界をまたがない水平近傍pairの平均係数、`boundary_pair_mean` は真の縦境界をまたぐ水平近傍pairの平均係数である。

| Weight map | Flat pair mean | Boundary pair mean |
| --- | ---: | ---: |
{weight_rows}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Clean guide: `clean_guide.png`
- Initial noisy guide: `initial_noisy.png`
- Model C output: `model_c.png`
- Perona-Malik output: `perona_malik.png`
- Model C guide weight map: `model_c_weight_h.png`
- Perona-Malik initial conductance map: `pm_conductance_initial_h.png`
- Perona-Malik final conductance map: `pm_conductance_final_h.png`
- Difference maps: `diff_model_c_vs_clean.png`, `diff_pm_vs_clean.png`
- Comparison strip: `comparison.png`

## Images

![Comparison of clean, noisy initial, Model C, Perona-Malik, and difference maps](comparison.png)

![Model C guide-derived horizontal weights](model_c_weight_h.png)

![Perona-Malik initial horizontal conductance](pm_conductance_initial_h.png)

![Perona-Malik final horizontal conductance](pm_conductance_final_h.png)

## Result

このrunでは、Model C と Perona-Malik 型 diffusion の両方を同じ noisy vertical step から比較できる成果物として保存した。Model C の重みは `guide` から計算した固定weightであり、Perona-Malik 型の conductance は初期画像と最終画像で異なる。

## Interpretation

両者は「大きな局所差のある近傍で混合を弱める」という点では類似している。ただし、Model C は guide `s` に基づく edge-aware interaction と data fidelity を持つ stochastic relaxation であり、Perona-Malik 型 diffusion は現在の画像勾配に基づく deterministic diffusion である。したがって、この比較からは「類似した目的を持つ部分がある」とは言えるが、「同等の効果」や「同じ方法」とは言わない。

## Limitations

- synthetic vertical edge 1条件だけの小規模比較であり、一般画像品質を示さない。
- Perona-Malik 実装は最小の明示的diffusion loopであり、元論文の数値解析や安定性検討を網羅しない。
- Model C は stochastic relaxation なので、seed、sweeps、temperature schedule に依存する。
- この結果は compression、super-resolution、Model Cの一般的優位性を示すものではない。

## Next

- 斜線、曲線、soft gradientで同じ比較を広げる場合は別Issueに分ける。
- Perona-Malik 型以外の guided filtering / anisotropic smoothing baseline と比較する場合は、低解像度guide条件と高解像度guidance条件を分ける。
"""


def main() -> None:
    ensure_dir(RESULT_DIR)
    clean = vertical_step(SIZE)
    initial = add_noise(clean, seed=EXPERIMENT_SEED, sigma=NOISE_SIGMA)
    foreground = clean > (LEFT_VALUE + RIGHT_VALUE) / 2.0

    model_c_params = ModelCParams(j_base=2.0, lambda_data=5.0, gamma=40.0)
    anneal_config = AnnealConfig(
        decoder_seed=DECODER_SEED,
        sweeps=60,
        temp_start=0.45,
        temp_end=0.01,
        proposal_sigma=0.09,
    )
    pm_steps = 60
    pm_dt = 0.18
    pm_kappa = 0.11

    start = time.perf_counter()
    model_c = model_c_decode(initial, model_c_params, anneal_config)
    model_c_time = time.perf_counter() - start

    start = time.perf_counter()
    pm = perona_malik_diffuse(initial, steps=pm_steps, dt=pm_dt, kappa=pm_kappa)
    pm_time = time.perf_counter() - start

    model_c_weights = model_c_horizontal_weight(initial, model_c_params)
    pm_initial_weights = horizontal_conductance(initial, pm_kappa)
    pm_final_weights = horizontal_conductance(pm, pm_kappa)
    boundary_mask = boundary_pair_mask(SIZE)
    flat_mask = ~boundary_mask

    output_metrics = [
        metrics_for("initial_noisy", initial, clean, foreground),
        metrics_for("model_c", model_c, clean, foreground),
        metrics_for("perona_malik", pm, clean, foreground),
    ]
    output_metrics[1]["decode_time_seconds"] = model_c_time
    output_metrics[2]["decode_time_seconds"] = pm_time

    pair_metrics = [
        {
            "name": "model_c_guide_weight",
            "flat_pair_mean": float(model_c_weights[flat_mask].mean()),
            "boundary_pair_mean": float(model_c_weights[boundary_mask].mean()),
        },
        {
            "name": "pm_initial_conductance",
            "flat_pair_mean": float(pm_initial_weights[flat_mask].mean()),
            "boundary_pair_mean": float(pm_initial_weights[boundary_mask].mean()),
        },
        {
            "name": "pm_final_conductance",
            "flat_pair_mean": float(pm_final_weights[flat_mask].mean()),
            "boundary_pair_mean": float(pm_final_weights[boundary_mask].mean()),
        },
    ]

    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "decoder_seed": DECODER_SEED,
        "size": SIZE,
        "left_value": LEFT_VALUE,
        "right_value": RIGHT_VALUE,
        "noise_sigma": NOISE_SIGMA,
        "model_c_params": {
            "j_base": model_c_params.j_base,
            "lambda_data": model_c_params.lambda_data,
            "gamma": model_c_params.gamma,
        },
        "anneal_config": {
            "sweeps": anneal_config.sweeps,
            "temp_start": anneal_config.temp_start,
            "temp_end": anneal_config.temp_end,
            "proposal_sigma": anneal_config.proposal_sigma,
        },
        "perona_malik_config": {"steps": pm_steps, "dt": pm_dt, "kappa": pm_kappa},
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    metrics = {"outputs": output_metrics, "pair_weights": pair_metrics}

    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", metrics)
    save_grayscale_png(RESULT_DIR / "clean_guide.png", clean)
    save_grayscale_png(RESULT_DIR / "initial_noisy.png", initial)
    save_grayscale_png(RESULT_DIR / "model_c.png", model_c)
    save_grayscale_png(RESULT_DIR / "perona_malik.png", pm)
    save_grayscale_png(RESULT_DIR / "diff_model_c_vs_clean.png", np.abs(model_c - clean))
    save_grayscale_png(RESULT_DIR / "diff_pm_vs_clean.png", np.abs(pm - clean))
    save_grayscale_png(RESULT_DIR / "model_c_weight_h.png", pad_pair_map(model_c_weights / model_c_params.j_base))
    save_grayscale_png(RESULT_DIR / "pm_conductance_initial_h.png", pad_pair_map(pm_initial_weights))
    save_grayscale_png(RESULT_DIR / "pm_conductance_final_h.png", pad_pair_map(pm_final_weights))
    save_comparison_png(
        RESULT_DIR / "comparison.png",
        [clean, initial, model_c, pm, np.abs(model_c - clean), np.abs(pm - clean)],
    )
    (RESULT_DIR / "notes.md").write_text(make_notes(config, metrics), encoding="utf-8")

    print("saved", RESULT_DIR)
    for row in output_metrics:
        print(f"{row['name']}: mad={row['mad_vs_clean']:.6f} edge={row['edge_leakage']:.6f}")
    for row in pair_metrics:
        print(f"{row['name']}: flat={row['flat_pair_mean']:.6f} boundary={row['boundary_pair_mean']:.6f}")


if __name__ == "__main__":
    main()
