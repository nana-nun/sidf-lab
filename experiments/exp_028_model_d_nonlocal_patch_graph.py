"""Compare low-guide-only non-local candidates against interpolation baselines.

Issue #130: measure whether low-guide-only self-similarity (a self-guided
Non-local Means baseline and a non-local patch graph decoder) improves
reference metrics over nearest / bilinear / bicubic and existing low-guide-only
guided filter / joint bilateral baselines.

Information rule: Ground Truth is used only for metrics. All patch similarity
and graph structure derive from the bilinear-upscaled low guide only, never
from Ground Truth or independent high-resolution guidance.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_005_model_d_shape_benchmark import (
    save_comparison_png,
    save_grayscale_png,
    upscale,
)
from experiments.exp_008_model_d_natural_patch import (
    downscale_block_average,
    natural_image_metrics,
)
from experiments.exp_015_guided_filter_baselines import (
    guided_filter,
    joint_bilateral_filter,
    synthetic_metrics,
    timed_call,
)
from sidf_lab.guides import circle, cross
from sidf_lab.io import ensure_dir, save_json
from sidf_lab.nonlocal_patch import (
    nonlocal_patch_graph_decode,
    self_guided_non_local_means,
)


RESULT_DIR = Path("results/2026-08-22-issue-130-model-d-nonlocal-patch-graph")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-08-22"
EXPERIMENT_SEED = 20260822

CROSS_LOW_SIZE = 16
CROSS_HIGH_SIZE = 64
CROSS_LOW_WIDTH = 2
CROSS_HIGH_WIDTH = 7
CIRCLE_LOW_SIZE = 16
CIRCLE_HIGH_SIZE = 64
CIRCLE_LOW_RADIUS = 4.0
CIRCLE_HIGH_RADIUS = 16.0
NATURAL_LOW_SIZE = 32
NATURAL_HIGH_SIZE = 128

GUIDED_FILTER_CONFIG = {
    "description": "self-guided filter on the bilinear-upscaled low guide",
    "radius": 3,
    "epsilon": 0.01**2,
    "high_resolution_guidance": False,
}
JOINT_BILATERAL_CONFIG = {
    "description": "nearest-upscaled values refined with the bilinear-upscaled low guide",
    "radius": 3,
    "sigma_spatial": 2.0,
    "sigma_range": 0.08,
    "high_resolution_guidance": False,
}
NLM_CONFIG = {
    "description": "self-guided Non-local Means over the bilinear-upscaled low guide",
    "patch_radius": 1,
    "search_radius": 5,
    "h": 0.08,
    "high_resolution_guidance": False,
}
PATCH_GRAPH_CONFIG = {
    "description": "non-local patch graph decoder on the bilinear-upscaled low guide",
    "lambda_data": 6.0,
    "j_base": 1.8,
    "gamma": 35.0,
    "j_nonlocal": 1.0,
    "patch_radius": 1,
    "search_radius": 7,
    "num_neighbors": 5,
    "local_exclude_radius": 1,
    "h": 0.08,
    "max_sweeps": 80,
    "tol": 1e-6,
    "high_resolution_guidance": False,
}

OUTPUT_NAMES = [
    "nearest",
    "bilinear",
    "bicubic",
    "guided_filter",
    "joint_bilateral",
    "self_guided_nlm",
    "nonlocal_patch_graph",
]
INTERPOLATION_NAMES = ["nearest", "bilinear", "bicubic"]


def make_outputs(low_guide: np.ndarray, high_size: int) -> tuple[
    dict[str, np.ndarray], dict[str, float], dict[str, Any]
]:
    """Build interpolation, edge-aware, NLM, and patch-graph outputs."""
    nearest, nearest_seconds = timed_call(upscale, low_guide, high_size, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, high_size, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, high_size, 3)

    guided, guided_seconds = timed_call(
        guided_filter,
        bilinear,
        bilinear,
        radius=int(GUIDED_FILTER_CONFIG["radius"]),
        epsilon=float(GUIDED_FILTER_CONFIG["epsilon"]),
    )
    joint, joint_seconds = timed_call(
        joint_bilateral_filter,
        nearest,
        bilinear,
        radius=int(JOINT_BILATERAL_CONFIG["radius"]),
        sigma_spatial=float(JOINT_BILATERAL_CONFIG["sigma_spatial"]),
        sigma_range=float(JOINT_BILATERAL_CONFIG["sigma_range"]),
    )
    nlm, nlm_seconds = timed_call(
        self_guided_non_local_means,
        bilinear,
        patch_radius=int(NLM_CONFIG["patch_radius"]),
        search_radius=int(NLM_CONFIG["search_radius"]),
        h=float(NLM_CONFIG["h"]),
    )
    (patch_graph, graph_info), graph_seconds = timed_call(
        nonlocal_patch_graph_decode,
        bilinear,
        lambda_data=float(PATCH_GRAPH_CONFIG["lambda_data"]),
        j_base=float(PATCH_GRAPH_CONFIG["j_base"]),
        gamma=float(PATCH_GRAPH_CONFIG["gamma"]),
        j_nonlocal=float(PATCH_GRAPH_CONFIG["j_nonlocal"]),
        patch_radius=int(PATCH_GRAPH_CONFIG["patch_radius"]),
        search_radius=int(PATCH_GRAPH_CONFIG["search_radius"]),
        num_neighbors=int(PATCH_GRAPH_CONFIG["num_neighbors"]),
        local_exclude_radius=int(PATCH_GRAPH_CONFIG["local_exclude_radius"]),
        h=float(PATCH_GRAPH_CONFIG["h"]),
        max_sweeps=int(PATCH_GRAPH_CONFIG["max_sweeps"]),
        tol=float(PATCH_GRAPH_CONFIG["tol"]),
    )
    outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
        "guided_filter": guided,
        "joint_bilateral": joint,
        "self_guided_nlm": nlm,
        "nonlocal_patch_graph": patch_graph,
    }
    timings = {
        "nearest": nearest_seconds,
        "bilinear": bilinear_seconds,
        "bicubic": bicubic_seconds,
        "guided_filter": guided_seconds,
        "joint_bilateral": joint_seconds,
        "self_guided_nlm": nlm_seconds,
        "nonlocal_patch_graph": graph_seconds,
    }
    return outputs, timings, graph_info


def save_case_artifacts(
    case_dir: Path,
    outputs: dict[str, np.ndarray],
    *,
    reference: np.ndarray,
    low_guide: np.ndarray,
) -> None:
    """Save the main PNGs and per-method reference differences."""
    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    for name, image in outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
        save_grayscale_png(case_dir / f"diff_{name}_vs_reference.png", np.abs(image - reference))
    guide_preview = upscale(low_guide, reference.shape[0], 0)
    save_comparison_png(
        case_dir / "comparison.png",
        [reference, guide_preview, *[outputs[name] for name in OUTPUT_NAMES]],
    )
    save_comparison_png(
        case_dir / "difference_comparison.png",
        [np.abs(outputs[name] - reference) for name in OUTPUT_NAMES],
    )


def run_synthetic_case(
    name: str,
    low_factory: Callable[[], np.ndarray],
    high_factory: Callable[[], np.ndarray],
    high_size: int,
) -> dict[str, Any]:
    """Run all outputs on a synthetic shape with a hard foreground mask."""
    case_dir = ensure_dir(RESULT_DIR / name)
    reference = high_factory()
    low_guide = low_factory()
    outputs, timings, graph_info = make_outputs(low_guide, high_size)
    foreground_mask = reference > 0.0
    metrics = {
        output_name: synthetic_metrics(image, reference, foreground_mask)
        for output_name, image in outputs.items()
    }
    save_case_artifacts(case_dir, outputs, reference=reference, low_guide=low_guide)
    return {"metrics": metrics, "timings": timings, "graph_info": graph_info}


def run_natural_case() -> dict[str, Any]:
    """Run all outputs on the public-domain natural patch."""
    case_dir = ensure_dir(RESULT_DIR / "natural_patch")
    reference = np.load(SOURCE_ASSET)
    low_guide = downscale_block_average(reference, NATURAL_LOW_SIZE)
    outputs, timings, graph_info = make_outputs(low_guide, NATURAL_HIGH_SIZE)
    metrics = {name: natural_image_metrics(image, reference) for name, image in outputs.items()}
    save_case_artifacts(case_dir, outputs, reference=reference, low_guide=low_guide)
    return {"metrics": metrics, "timings": timings, "graph_info": graph_info}


def best_by(metrics: dict[str, dict[str, float | None]], metric_name: str, names: list[str]) -> str:
    """Return the output name with the smallest non-null metric among ``names``."""
    return min(
        names,
        key=lambda name: float(metrics[name][metric_name])
        if metrics[name][metric_name] is not None
        else float("inf"),
    )


def format_synthetic_rows(metrics: dict[str, dict[str, float | None]], timings: dict[str, float]) -> str:
    """Format the synthetic shape metric table."""
    return "\n".join(
        "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {grad:.6f} | {edge:.6f} | {seconds:.6f} |".format(
            name=name,
            mad=metrics[name]["mad_vs_reference"],
            psnr=metrics[name]["psnr_vs_reference"],
            ssim=metrics[name]["ssim_global_vs_reference"],
            grad=metrics[name]["gradient_mad_vs_reference"],
            edge=metrics[name]["edge_leakage"],
            seconds=timings[name],
        )
        for name in OUTPUT_NAMES
    )


def format_natural_rows(metrics: dict[str, dict[str, float]], timings: dict[str, float]) -> str:
    """Format the natural patch metric table."""
    return "\n".join(
        "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {grad:.6f} | {edge:.6f} | {seconds:.6f} |".format(
            name=name,
            mad=metrics[name]["mad_vs_gt"],
            psnr=metrics[name]["psnr_vs_gt"],
            ssim=metrics[name]["ssim_global_vs_gt"],
            grad=metrics[name]["gradient_mad_vs_gt"],
            edge=metrics[name]["strong_edge_mad_vs_gt"],
            seconds=timings[name],
        )
        for name in OUTPUT_NAMES
    )


def graph_stats_rows(results: dict[str, Any]) -> str:
    """Format non-local patch graph statistics per case."""
    rows = []
    for case_name, case_results in results.items():
        stats = case_results["graph_info"]["graph_statistics"]
        solver = case_results["graph_info"]["solver"]
        rows.append(
            "| {case} | {degree:.3f} | {mean_dist:.6f} | {median_dist:.6f} | {edges:.0f} | {sweeps:.0f} | {obj:.4f} |".format(
                case=case_name,
                degree=stats["mean_nonlocal_degree"],
                mean_dist=stats["mean_patch_distance"],
                median_dist=stats["median_patch_distance"],
                edges=stats["nonlocal_edge_count"],
                sweeps=solver["sweeps_run"],
                obj=solver["final_objective"],
            )
        )
    return "\n".join(rows)


def candidate_comparison(metrics: dict[str, dict[str, float | None]], metric_name: str) -> str:
    """Return a data-driven sentence comparing candidates to interpolation."""
    best_interp = best_by(metrics, metric_name, INTERPOLATION_NAMES)
    best_interp_value = float(metrics[best_interp][metric_name])
    lines = []
    for candidate in ("self_guided_nlm", "nonlocal_patch_graph"):
        value = float(metrics[candidate][metric_name])
        verdict = "下回った（改善）" if value < best_interp_value else "上回った（悪化）または同等"
        lines.append(
            f"`{candidate}` の {metric_name} `{value:.6f}` は、補間baseline最小の "
            f"`{best_interp}` `{best_interp_value:.6f}` を{verdict}。"
        )
    return " ".join(lines)


def build_notes(config: dict[str, Any], results: dict[str, Any]) -> str:
    """Build the Japanese experiment note."""
    cross_rows = format_synthetic_rows(results["cross"]["metrics"], results["cross"]["timings"])
    circle_rows = format_synthetic_rows(results["circle"]["metrics"], results["circle"]["timings"])
    natural_rows = format_natural_rows(
        results["natural_patch"]["metrics"], results["natural_patch"]["timings"]
    )
    cross_best = best_by(results["cross"]["metrics"], "mad_vs_reference", OUTPUT_NAMES)
    circle_best = best_by(results["circle"]["metrics"], "mad_vs_reference", OUTPUT_NAMES)
    natural_best = best_by(results["natural_patch"]["metrics"], "mad_vs_gt", OUTPUT_NAMES)
    cross_compare = candidate_comparison(results["cross"]["metrics"], "mad_vs_reference")
    circle_compare = candidate_comparison(results["circle"]["metrics"], "mad_vs_reference")
    natural_compare = candidate_comparison(results["natural_patch"]["metrics"], "mad_vs_gt")
    stats_rows = graph_stats_rows(results)
    return f"""# Model D Non-local Patch Graph と Self-guided NLM Baseline の比較

## Question

low-guide-only 条件で、bilinear-upscaled low guide から作る self-guided Non-local Means baseline と non-local patch graph decoder 候補は、nearest / bilinear / bicubic および既存の low-guide-only guided filter / joint bilateral baseline と比べて、cross / circle / 自然画像patchの reference metrics を改善するか。

## Hypothesis

low guide に残った繰り返し構造や自己類似性は、局所4近傍平滑化より有用な拘束になる可能性がある。ただし、self-guided NLM も patch graph も high-resolution guidance や Ground Truth を使わないため、低解像度guideで失われた高周波構造を復元するとは仮定しない。#87 / #88 の負の結果を踏まえ、有限温度Metropolisとwhite-noise textureは使わず、決定論的Jacobi solverで解く。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_028_model_d_nonlocal_patch_graph.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]} (NLM と patch graph は決定論的で乱数を使わない)
- Cross: {config["cross_low_size"]}x{config["cross_low_size"]} guide to {config["cross_high_size"]}x{config["cross_high_size"]} output
- Circle: {config["circle_low_size"]}x{config["circle_low_size"]} guide to {config["circle_high_size"]}x{config["circle_high_size"]} output
- Natural patch: {config["natural_low_size"]}x{config["natural_low_size"]} guide to {config["natural_high_size"]}x{config["natural_high_size"]} output
- Natural patch source asset: `{config["source_asset"]}`
- Guidance policy: {config["guidance_policy"]}
- NLM config: `{config["nlm_config"]}`
- Patch graph config: `{config["patch_graph_config"]}`
- Guided filter config: `{config["guided_filter_config"]}`
- Joint bilateral config: `{config["joint_bilateral_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

補間baselineは nearest / bilinear / bicubic。既存 low-guide-only edge-aware baseline として guided filter と joint bilateral を含める。いずれも独立した高解像度guidanceを使わず、guidance は bilinear-upscaled low guide から作る。metrics の reference は synthetic shape の高解像度生成、または自然画像cropの128x128 grayscaleであり、Ground Truth は metrics 計算にのみ使う。

## What changed from #87 / #88

- Objective: 局所4近傍のquadratic pairwiseだけでなく、low-guide由来 patch descriptor で選んだ非局所edgeを加えた。
- Update: 有限温度Metropolis (#87) や Gaussian proposal greedy / ICM (#88) ではなく、決定論的Jacobi sweepでquadratic objectiveを解く。
- Texture: white-noise texture term (#37) は使わない。
- Confidence: gradient-based confidence map (#56/#61/#67) は使わず、data fidelity と patch-similarity 重みのみで構成した。
- Patch matching は bilinear-upscaled low guide からのみ計算し、Ground Truth や高解像度guidanceは使わない。

## Metrics

### Cross ({config["cross_high_size"]}x{config["cross_high_size"]})

| Output | MAD vs reference | PSNR | SSIM | Gradient MAD | Edge leakage | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{cross_rows}

### Circle ({config["circle_high_size"]}x{config["circle_high_size"]})

| Output | MAD vs reference | PSNR | SSIM | Gradient MAD | Edge leakage | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{circle_rows}

### Natural Patch ({config["natural_high_size"]}x{config["natural_high_size"]})

| Output | MAD vs GT | PSNR | SSIM | Gradient MAD | Strong-edge MAD | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{natural_rows}

### Non-local Patch Graph Statistics

| Case | Mean non-local degree | Mean patch distance | Median patch distance | Non-local edges | Solver sweeps | Final objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{stats_rows}

## Saved Artifacts

- Config: `config.json`
- Metrics and graph statistics: `metrics.json`
- Notes: `notes.md`
- Per-case artifacts: `cross/`, `circle/`, `natural_patch/`
- 各caseに reference、low guide、全出力、reference差分、comparison / difference_comparison PNGを保存した。

## Images

![Cross comparison](cross/comparison.png)

![Cross difference comparison](cross/difference_comparison.png)

![Circle comparison](circle/comparison.png)

![Natural patch comparison](natural_patch/comparison.png)

![Natural patch difference comparison](natural_patch/difference_comparison.png)

## Result

MADが最小だった出力は、crossで `{cross_best}`、circleで `{circle_best}`、natural patchで `{natural_best}` だった。

候補と補間baselineの比較:

- Cross: {cross_compare}
- Circle: {circle_compare}
- Natural patch: {natural_compare}

## Interpretation

この結果は、low-guide-only 条件で非局所自己類似性を追加した2候補の最小比較である。MADの最小値が補間baselineのままの場合、現在のparameterでは patch graph / NLM が single-pass補間より総合的に優れるとは言えない。改善が見られた指標がある場合も、cross / circle / 1枚のnatural patch という限定条件での初期測定であり、super-resolution / compression / 既存形式への優位性を示すものではない。

non-local patch graph は決定論的Jacobi solverでobjectiveを下げるが、#88 と同様に、objective低下と reference metrics 改善は別の評価軸として扱う。patch descriptor を bilinear-upscaled low guide から作るため、選ばれる非局所neighborが補間結果の滑らかさを再表現しているだけの可能性がある点にも注意する。

## Limitations

- cross / circle の synthetic shape と、1枚のpublic-domain自然画像patchだけの比較である。
- patch matching は bilinear-upscaled low guide からのみ計算しており、独立した高解像度guidanceやGround Truthは使っていない。したがって high-resolution guidance を使う手法の性能は測っていない。
- NLM と patch graph のparameter（patch radius、search radius、neighbor数、h、重み）は小規模な固定値で、網羅的探索はしていない。
- solver は決定論的Jacobiで、bit-perfect cross-environment再現性は未確認である。
- Global SSIM は依存なしの全画像SSIMであり、windowed SSIMではない。
- decode time はこの環境の小画像runに限る。
- この結果はsuper-resolutionやcompressionの成立を示さない。

## Next

- 改善が限定的な場合は、patch descriptor を bilinear ではなく別のlow-guide由来表現（例: nearest-upscaled guide や low-resolution guide 空間でのpatch matching）から作る条件を分けて比較する。
- high-resolution guidance を使う joint upsampling 条件は、low-guide-only 条件と明確に分けた別Issueで測る。
- patch graph の objective 低下と reference metrics の乖離が大きい場合は、quadratic penalty ではなく robust penalty や residual target 候補を別Issueで比較する。
"""


def main() -> None:
    """Run the experiment and save artifacts."""
    ensure_dir(RESULT_DIR)
    results = {
        "cross": run_synthetic_case(
            "cross",
            lambda: cross(size=CROSS_LOW_SIZE, width=CROSS_LOW_WIDTH, value=0.5),
            lambda: cross(size=CROSS_HIGH_SIZE, width=CROSS_HIGH_WIDTH, value=0.5),
            CROSS_HIGH_SIZE,
        ),
        "circle": run_synthetic_case(
            "circle",
            lambda: circle(size=CIRCLE_LOW_SIZE, radius=CIRCLE_LOW_RADIUS, value=0.5),
            lambda: circle(size=CIRCLE_HIGH_SIZE, radius=CIRCLE_HIGH_RADIUS, value=0.5),
            CIRCLE_HIGH_SIZE,
        ),
        "natural_patch": run_natural_case(),
    }
    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "cross_low_size": CROSS_LOW_SIZE,
        "cross_high_size": CROSS_HIGH_SIZE,
        "cross_low_width": CROSS_LOW_WIDTH,
        "cross_high_width": CROSS_HIGH_WIDTH,
        "circle_low_size": CIRCLE_LOW_SIZE,
        "circle_high_size": CIRCLE_HIGH_SIZE,
        "circle_low_radius": CIRCLE_LOW_RADIUS,
        "circle_high_radius": CIRCLE_HIGH_RADIUS,
        "natural_low_size": NATURAL_LOW_SIZE,
        "natural_high_size": NATURAL_HIGH_SIZE,
        "source_asset": SOURCE_ASSET.as_posix(),
        "source_page": "https://commons.wikimedia.org/wiki/File:Landscape.jpg",
        "source_title": "Meindert Hobbema: Landscape",
        "license_note": "Wikimedia Commons marks the faithful reproduction of this public-domain artwork as Public Domain / PD-Art.",
        "guidance_policy": "No independent high-resolution guidance; all patch similarity derives from the low guide only. Ground Truth is used only for metrics.",
        "nlm_config": NLM_CONFIG,
        "patch_graph_config": PATCH_GRAPH_CONFIG,
        "guided_filter_config": GUIDED_FILTER_CONFIG,
        "joint_bilateral_config": JOINT_BILATERAL_CONFIG,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    metrics_payload = {
        case_name: {
            "metrics": case_results["metrics"],
            "timings": case_results["timings"],
            "graph_info": case_results["graph_info"],
        }
        for case_name, case_results in results.items()
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", metrics_payload)
    (RESULT_DIR / "notes.md").write_text(build_notes(config, results), encoding="utf-8")

    for case_name, case_results in results.items():
        metric_name = "mad_vs_gt" if case_name == "natural_patch" else "mad_vs_reference"
        print(case_name)
        for name in OUTPUT_NAMES:
            values = case_results["metrics"][name]
            print(
                f"  {name}: {metric_name}={values[metric_name]:.6f} "
                f"time={case_results['timings'][name]:.6f}s"
            )


if __name__ == "__main__":
    main()
