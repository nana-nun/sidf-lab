"""Compare Model D acceptance modes and pixel update orders."""

from __future__ import annotations

import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_005_model_d_shape_benchmark import save_comparison_png, upscale
from experiments.exp_008_model_d_natural_patch import downscale_block_average
from experiments.exp_018_model_d_redesign_candidates import (
    cross_metrics,
    local_energy,
    natural_metrics,
)
from sidf_lab.energy import valid_neighbors
from sidf_lab.guides import cross
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json


RESULT_DIR = Path("results/2026-06-14-issue-87-model-d-update-procedure")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-06-14"
EXPERIMENT_SEED = 20260614
CROSS_DECODER_SEED = 8700
NATURAL_DECODER_SEED = 8701
CROSS_LOW_SIZE = 16
CROSS_HIGH_SIZE = 64
CROSS_LOW_WIDTH = 2
CROSS_HIGH_WIDTH = 7
NATURAL_LOW_SIZE = 32
NATURAL_HIGH_SIZE = 128

CONDITIONS = [
    {
        "id": "stochastic_random",
        "label": "Metropolis acceptance + random pixel order",
        "acceptance_mode": "stochastic",
        "update_order": "random",
    },
    {
        "id": "stochastic_fixed",
        "label": "Metropolis acceptance + fixed row-major order",
        "acceptance_mode": "stochastic",
        "update_order": "fixed",
    },
    {
        "id": "greedy_random",
        "label": "greedy acceptance + random pixel order",
        "acceptance_mode": "greedy",
        "update_order": "random",
    },
    {
        "id": "greedy_fixed",
        "label": "greedy acceptance + fixed row-major order",
        "acceptance_mode": "greedy",
        "update_order": "fixed",
    },
]

MODEL_PARAMS = {
    "j_base": 1.8,
    "lambda_data": 6.0,
    "gamma": 35.0,
    "pairwise_cap": 0.08,
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
    """Run a callable and return its result plus elapsed seconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def total_objective(
    state: np.ndarray,
    guide: np.ndarray,
    confidence: np.ndarray,
    *,
    j_base: float,
    lambda_data: float,
    gamma: float,
) -> float:
    """Return the data plus once-counted quadratic pairwise objective."""
    values = np.asarray(state, dtype=np.float64)
    guide_values = np.asarray(guide, dtype=np.float64)
    energy = lambda_data * float(np.sum(confidence * (values - guide_values) ** 2))
    height, width = values.shape
    for i in range(height):
        for j in range(width):
            for ni, nj in valid_neighbors(i, j, height, width):
                if (ni, nj) <= (i, j):
                    continue
                interaction = j_base * math.exp(
                    -gamma * (float(guide_values[i, j]) - float(guide_values[ni, nj])) ** 2
                )
                energy += interaction * (float(values[i, j]) - float(values[ni, nj])) ** 2
    return float(energy)


def decode(
    guide: np.ndarray,
    *,
    acceptance_mode: str,
    update_order: str,
    decoder_seed: int,
    sweeps: int,
    temp_start: float,
    temp_end: float,
    proposal_sigma: float,
    j_base: float,
    lambda_data: float,
    gamma: float,
    pairwise_cap: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Decode with selected acceptance and traversal policies."""
    if acceptance_mode not in {"stochastic", "greedy"}:
        raise ValueError(f"unknown acceptance mode: {acceptance_mode}")
    if update_order not in {"random", "fixed"}:
        raise ValueError(f"unknown update order: {update_order}")

    rng = np.random.default_rng(decoder_seed)
    state = np.asarray(guide, dtype=np.float64).copy()
    confidence = np.ones_like(state, dtype=np.float64)
    height, width = state.shape
    pixel_count = height * width
    fixed_order = np.arange(pixel_count)
    temperatures = np.geomspace(temp_start, temp_end, sweeps)
    initial_objective = total_objective(
        state,
        guide,
        confidence,
        j_base=j_base,
        lambda_data=lambda_data,
        gamma=gamma,
    )
    accepted = 0
    downhill_accepted = 0
    neutral_accepted = 0
    uphill_accepted = 0
    energy_tolerance = 1e-15
    energy_args = {
        "pairwise_mode": "quadratic",
        "j_base": j_base,
        "lambda_data": lambda_data,
        "gamma": gamma,
        "pairwise_cap": pairwise_cap,
    }

    for temperature in temperatures:
        indices = rng.permutation(pixel_count) if update_order == "random" else fixed_order
        for index in indices:
            i, j = divmod(int(index), width)
            old_value = float(state[i, j])
            new_value = float(np.clip(old_value + rng.normal(0.0, proposal_sigma), 0.0, 1.0))
            old_energy = local_energy(old_value, state, guide, confidence, i, j, **energy_args)
            new_energy = local_energy(new_value, state, guide, confidence, i, j, **energy_args)
            delta = new_energy - old_energy
            accept = delta < -energy_tolerance
            if not accept and acceptance_mode == "stochastic":
                accept = rng.random() < math.exp(-delta / float(temperature))
            if accept:
                state[i, j] = new_value
                accepted += 1
                if delta < -energy_tolerance:
                    downhill_accepted += 1
                elif delta > energy_tolerance:
                    uphill_accepted += 1
                else:
                    neutral_accepted += 1

    proposals = sweeps * pixel_count
    diagnostics: dict[str, float | int] = {
        "proposals": proposals,
        "accepted": accepted,
        "downhill_accepted": downhill_accepted,
        "neutral_accepted": neutral_accepted,
        "uphill_accepted": uphill_accepted,
        "acceptance_rate": accepted / proposals,
        "neutral_acceptance_rate": neutral_accepted / proposals,
        "uphill_acceptance_rate": uphill_accepted / proposals,
        "initial_objective": initial_objective,
        "final_objective": total_objective(
            state,
            guide,
            confidence,
            j_base=j_base,
            lambda_data=lambda_data,
            gamma=gamma,
        ),
    }
    return state, diagnostics


def run_case(
    case_name: str,
    reference: np.ndarray,
    low_guide: np.ndarray,
    *,
    decoder_seed: int,
    decode_config: dict[str, float | int],
    foreground_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run interpolation baselines and all update-procedure conditions."""
    case_dir = ensure_dir(RESULT_DIR / case_name)
    output_size = reference.shape[0]
    nearest, nearest_seconds = timed_call(upscale, low_guide, output_size, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, output_size, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, output_size, 3)
    outputs = {"nearest": nearest, "bilinear": bilinear, "bicubic": bicubic}
    timings = {
        "nearest": nearest_seconds,
        "bilinear": bilinear_seconds,
        "bicubic": bicubic_seconds,
    }
    diagnostics: dict[str, dict[str, float | int]] = {}

    for condition in CONDITIONS:
        condition_id = str(condition["id"])
        (rendered, condition_diagnostics), seconds = timed_call(
            decode,
            bilinear,
            acceptance_mode=str(condition["acceptance_mode"]),
            update_order=str(condition["update_order"]),
            decoder_seed=decoder_seed,
            **MODEL_PARAMS,
            **decode_config,
        )
        outputs[condition_id] = rendered
        timings[condition_id] = seconds
        diagnostics[condition_id] = condition_diagnostics

    if foreground_mask is None:
        metrics = {name: natural_metrics(image, reference) for name, image in outputs.items()}
    else:
        metrics = {
            name: cross_metrics(image, reference, foreground_mask)
            for name, image in outputs.items()
        }
    save_case_artifacts(
        case_dir,
        reference=reference,
        low_guide=low_guide,
        outputs=outputs,
        baseline_for_diff=bilinear,
    )
    return {"metrics": metrics, "timings": timings, "diagnostics": diagnostics}


def save_case_artifacts(
    case_dir: Path,
    *,
    reference: np.ndarray,
    low_guide: np.ndarray,
    outputs: dict[str, np.ndarray],
    baseline_for_diff: np.ndarray,
) -> None:
    """Save baseline, rendered, difference, and comparison images."""
    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    for name, image in outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
        if name not in {"nearest", "bilinear", "bicubic"}:
            save_grayscale_png(case_dir / f"diff_{name}_vs_bilinear.png", np.abs(image - baseline_for_diff))
            save_grayscale_png(case_dir / f"diff_{name}_vs_reference.png", np.abs(image - reference))
    condition_ids = [str(condition["id"]) for condition in CONDITIONS]
    save_comparison_png(
        case_dir / "comparison.png",
        [
            reference,
            upscale(low_guide, reference.shape[0], 0),
            outputs["nearest"],
            outputs["bilinear"],
            outputs["bicubic"],
            *[outputs[name] for name in condition_ids],
        ],
    )


def format_rows(
    case_name: str,
    results: dict[str, Any],
) -> str:
    """Format one case's metrics and diagnostics table."""
    metrics = results["metrics"]
    timings = results["timings"]
    diagnostics = results["diagnostics"]
    names = ["nearest", "bilinear", "bicubic", *[str(condition["id"]) for condition in CONDITIONS]]
    rows = []
    for name in names:
        if case_name == "cross":
            mad = metrics[name]["mad_vs_reference"]
            psnr = metrics[name]["psnr_vs_reference"]
            ssim = metrics[name]["ssim_global_vs_reference"]
        else:
            mad = metrics[name]["mad_vs_gt"]
            psnr = metrics[name]["psnr_vs_gt"]
            ssim = metrics[name]["ssim_global_vs_gt"]
        if name in diagnostics:
            acceptance = f"{diagnostics[name]['acceptance_rate']:.6f}"
            uphill = f"{diagnostics[name]['uphill_acceptance_rate']:.6f}"
            objective = f"{diagnostics[name]['final_objective']:.6f}"
        else:
            acceptance = uphill = objective = "N/A"
        rows.append(
            "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {gradient:.6f} | "
            "{acceptance} | {uphill} | {objective} | {seconds:.6f} |".format(
                name=name,
                mad=mad,
                psnr=psnr,
                ssim=ssim,
                gradient=metrics[name]["gradient_magnitude_mad"],
                acceptance=acceptance,
                uphill=uphill,
                objective=objective,
                seconds=timings[name],
            )
        )
    return "\n".join(rows)


def best_condition(results: dict[str, Any], case_name: str) -> tuple[str, float]:
    """Return the update condition with the smallest MAD."""
    metric_name = "mad_vs_reference" if case_name == "cross" else "mad_vs_gt"
    pairs = [
        (str(condition["id"]), float(results[case_name]["metrics"][str(condition["id"])][metric_name]))
        for condition in CONDITIONS
    ]
    return min(pairs, key=lambda pair: pair[1])


def build_notes(config: dict[str, Any], results: dict[str, Any]) -> str:
    """Build Japanese experiment notes from measured results."""
    cross_rows = format_rows("cross", results["cross"])
    natural_rows = format_rows("natural_patch", results["natural_patch"])
    cross_best, cross_best_mad = best_condition(results, "cross")
    natural_best, natural_best_mad = best_condition(results, "natural_patch")
    cross_stochastic = results["cross"]["diagnostics"]["stochastic_random"]
    cross_greedy = results["cross"]["diagnostics"]["greedy_random"]
    natural_stochastic = results["natural_patch"]["diagnostics"]["stochastic_random"]
    natural_greedy = results["natural_patch"]["diagnostics"]["greedy_random"]
    return f"""# Model D Acceptance / Update Order Isolation

## Question

Model Dのreference差分増加は、有限温度Metropolis acceptanceによるuphill moveと、pixel更新順序のどちらに強く関係するか。

## Hypothesis

greedy acceptanceはuphill moveを除くため、bilinear初期状態からの確率的driftを抑え、現行stochastic条件よりreference差分を減らす可能性がある。fixed orderの影響はacceptance modeより小さいと予想する。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_019_model_d_update_procedure.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Cross decoder seed: {config["cross_decoder_seed"]}
- Natural patch decoder seed: {config["natural_decoder_seed"]}
- Initial state: bilinear upscaled guide
- Confidence: uniform 1.0
- Texture: 0.0
- Pairwise: current quadratic interaction
- Conditions: `config.json` の `conditions`
- Model params: `{config["model_params"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

nearest、bilinear、bicubicを画像baselineとした。更新手順の対照は `stochastic_random` を現行相当とし、acceptanceだけをgreedyへ、更新順序だけをfixed row-majorへ切り替えた。

## Metrics

### Cross

| Output | MAD | PSNR | Global SSIM | Gradient magnitude MAD | Accept rate | Uphill rate | Final objective | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{cross_rows}

### Natural Patch

| Output | MAD | PSNR | Global SSIM | Gradient magnitude MAD | Accept rate | Uphill rate | Final objective | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{natural_rows}

## Saved Artifacts

- Config: `config.json`
- Metrics and diagnostics: `metrics.json`
- Notes: `notes.md`
- Cross artifacts: `cross/`
- Natural patch artifacts: `natural_patch/`
- 各caseにbaseline、各更新条件、reference/bilinear差分、`comparison.png`を保存した。

## Images

![Cross update procedure comparison](cross/comparison.png)

![Natural patch update procedure comparison](natural_patch/comparison.png)

## Result

更新条件内の最小MADは、crossでは `{cross_best}` の `{cross_best_mad:.6f}`、natural patchでは `{natural_best}` の `{natural_best_mad:.6f}` だった。

## Interpretation

stochastic条件はcrossでproposalの `{cross_stochastic["uphill_acceptance_rate"]:.3f}`、natural patchで `{natural_stochastic["uphill_acceptance_rate"]:.3f}` をuphill moveとして受理した。最終objectiveはcrossで初期 `{cross_stochastic["initial_objective"]:.3f}` から `{cross_stochastic["final_objective"]:.3f}`、natural patchで初期 `{natural_stochastic["initial_objective"]:.3f}` から `{natural_stochastic["final_objective"]:.3f}` へ増加した。

greedy条件はuphill moveを受理せず、最終objectiveをcross `{cross_greedy["final_objective"]:.3f}`、natural patch `{natural_greedy["final_objective"]:.3f}` まで低下させた。MADもstochastic条件から大きく改善し、自然画像ではbilinearに近い値へ戻った。一方、random / fixed order間のMAD差はcross・natural patchとも小さく、今回の設定では更新順序よりacceptance modeの影響が大きかった。

ただしgreedy条件もcrossではbilinear MAD `0.033143`、natural patchではbilinear MAD `0.044369` とbicubic MAD `0.042397` を上回らなかった。objective低下とreference metrics改善は一致せず、現行quadratic objective自体がreference品質を改善するとは確認できない。

## Limitations

- crossと1枚のpublic-domain自然画像patchだけの比較である。
- greedy条件もGaussian proposalを使うため、連続値objectiveの厳密な座標最適化ではない。
- random order条件とfixed order条件ではRNG消費順が異なり、pixelごとのproposal列は完全には一致しない。
- Global SSIMは依存なしの全画像SSIMであり、windowed SSIMではない。
- この結果はsuper-resolutionやcompressionの成立を示さない。

## Next

- Issue #88 で、Gaussian proposalに依存するgreedy更新と、quadratic objectiveの解析的な局所最小値を使うdeterministic ICM / coordinate descentを比較する。
- 目的はproposal samplingの非効率とobjective自体の限界を分けることであり、単純なtemperature調整は広げない。
"""


def main() -> None:
    """Run both cases and save reproducible artifacts."""
    ensure_dir(RESULT_DIR)
    cross_reference = cross(size=CROSS_HIGH_SIZE, width=CROSS_HIGH_WIDTH, value=0.5)
    cross_guide = cross(size=CROSS_LOW_SIZE, width=CROSS_LOW_WIDTH, value=0.5)
    natural_reference = np.load(SOURCE_ASSET)
    natural_guide = downscale_block_average(natural_reference, NATURAL_LOW_SIZE)
    results = {
        "cross": run_case(
            "cross",
            cross_reference,
            cross_guide,
            decoder_seed=CROSS_DECODER_SEED,
            decode_config=CROSS_DECODE_CONFIG,
            foreground_mask=cross_reference > 0.0,
        ),
        "natural_patch": run_case(
            "natural_patch",
            natural_reference,
            natural_guide,
            decoder_seed=NATURAL_DECODER_SEED,
            decode_config=NATURAL_DECODE_CONFIG,
        ),
    }
    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "cross_decoder_seed": CROSS_DECODER_SEED,
        "natural_decoder_seed": NATURAL_DECODER_SEED,
        "cross_low_size": CROSS_LOW_SIZE,
        "cross_high_size": CROSS_HIGH_SIZE,
        "natural_low_size": NATURAL_LOW_SIZE,
        "natural_high_size": NATURAL_HIGH_SIZE,
        "source_asset": SOURCE_ASSET.as_posix(),
        "conditions": CONDITIONS,
        "model_params": MODEL_PARAMS,
        "cross_decode_config": CROSS_DECODE_CONFIG,
        "natural_decode_config": NATURAL_DECODE_CONFIG,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", results)
    (RESULT_DIR / "notes.md").write_text(build_notes(config, results), encoding="utf-8")

    for case_name, case_results in results.items():
        metric_name = "mad_vs_reference" if case_name == "cross" else "mad_vs_gt"
        print(case_name)
        for name, values in case_results["metrics"].items():
            print(f"  {name}: {metric_name}={values[metric_name]:.6f}")


if __name__ == "__main__":
    main()
