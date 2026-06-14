"""Compare small Model D confidence-map and pairwise redesign candidates."""

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
from experiments.exp_008_model_d_natural_patch import downscale_block_average, natural_image_metrics
from sidf_lab.confidence import gradient_confidence
from sidf_lab.energy import valid_neighbors
from sidf_lab.guides import cross
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json
from sidf_lab.metrics import comparison_summary, perceptual_gradient_summary


RESULT_DIR = Path("results/2026-06-14-issue-67-model-d-redesign-candidates")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-06-14"
EXPERIMENT_SEED = 20260614
CROSS_DECODER_SEED = 6700
NATURAL_DECODER_SEED = 6701
CROSS_LOW_SIZE = 16
CROSS_HIGH_SIZE = 64
CROSS_LOW_WIDTH = 2
CROSS_HIGH_WIDTH = 7
NATURAL_LOW_SIZE = 32
NATURAL_HIGH_SIZE = 128

CONDITIONS = [
    {
        "id": "current_gradient_quadratic",
        "label": "current gradient confidence + quadratic pairwise",
        "confidence_mode": "current_gradient",
        "pairwise_mode": "quadratic",
    },
    {
        "id": "uniform_quadratic",
        "label": "uniform confidence + quadratic pairwise",
        "confidence_mode": "uniform",
        "pairwise_mode": "quadratic",
    },
    {
        "id": "flatter_quadratic",
        "label": "flatter gradient confidence + quadratic pairwise",
        "confidence_mode": "flatter",
        "pairwise_mode": "quadratic",
    },
    {
        "id": "edge_band_quadratic",
        "label": "edge-band confidence + quadratic pairwise",
        "confidence_mode": "edge_band",
        "pairwise_mode": "quadratic",
    },
    {
        "id": "uniform_clamped_pairwise",
        "label": "uniform confidence + clamped pairwise",
        "confidence_mode": "uniform",
        "pairwise_mode": "clamped",
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


def normalized_gradient(guide: np.ndarray) -> np.ndarray:
    """Return guide gradient magnitude normalized to [0, 1]."""
    gy, gx = np.gradient(np.asarray(guide, dtype=np.float64))
    magnitude = np.hypot(gx, gy)
    maximum = float(magnitude.max())
    if maximum == 0.0:
        return np.zeros_like(magnitude)
    return magnitude / maximum


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilate a boolean mask with four-neighbor connectivity."""
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return result


def make_confidence(guide: np.ndarray, mode: str) -> np.ndarray:
    """Build one of the confidence-map redesign candidates."""
    if mode == "uniform":
        return np.ones_like(guide, dtype=np.float64)
    if mode == "current_gradient":
        return gradient_confidence(guide, min_confidence=0.2, max_confidence=1.0, scale=4.0)

    gradient = normalized_gradient(guide)
    if mode == "flatter":
        return 0.65 + 0.35 * gradient
    if mode == "edge_band":
        positive = gradient[gradient > 0.0]
        if positive.size == 0:
            return np.full_like(guide, 0.65, dtype=np.float64)
        threshold = float(np.quantile(positive, 0.75))
        edge_band = dilate(gradient >= threshold, radius=1)
        return np.where(edge_band, 1.0, 0.65)
    raise ValueError(f"unknown confidence mode: {mode}")


def pairwise_penalty(difference: float, mode: str, cap: float) -> float:
    """Return quadratic or truncated-quadratic pairwise penalty."""
    squared = difference * difference
    if mode == "quadratic":
        return squared
    if mode == "clamped":
        return min(squared, cap * cap)
    raise ValueError(f"unknown pairwise mode: {mode}")


def local_energy(
    value: float,
    state: np.ndarray,
    guide: np.ndarray,
    confidence: np.ndarray,
    i: int,
    j: int,
    *,
    pairwise_mode: str,
    j_base: float,
    lambda_data: float,
    gamma: float,
    pairwise_cap: float,
) -> float:
    """Compute local Model D energy for one redesign condition."""
    height, width = guide.shape
    guide_value = float(guide[i, j])
    energy = lambda_data * float(confidence[i, j]) * (value - guide_value) ** 2
    for ni, nj in valid_neighbors(i, j, height, width):
        neighbor_guide = float(guide[ni, nj])
        interaction = j_base * math.exp(-gamma * (guide_value - neighbor_guide) ** 2)
        difference = value - float(state[ni, nj])
        energy += interaction * pairwise_penalty(difference, pairwise_mode, pairwise_cap)
    return energy


def decode(
    guide: np.ndarray,
    confidence: np.ndarray,
    *,
    pairwise_mode: str,
    decoder_seed: int,
    sweeps: int,
    temp_start: float,
    temp_end: float,
    proposal_sigma: float,
    j_base: float,
    lambda_data: float,
    gamma: float,
    pairwise_cap: float,
) -> np.ndarray:
    """Decode from the upscaled guide with a selected pairwise penalty."""
    rng = np.random.default_rng(decoder_seed)
    state = np.asarray(guide, dtype=np.float64).copy()
    height, width = state.shape
    temperatures = np.geomspace(temp_start, temp_end, sweeps)
    energy_args = {
        "pairwise_mode": pairwise_mode,
        "j_base": j_base,
        "lambda_data": lambda_data,
        "gamma": gamma,
        "pairwise_cap": pairwise_cap,
    }

    for temperature in temperatures:
        for index in rng.permutation(height * width):
            i, j = divmod(int(index), width)
            old_value = float(state[i, j])
            new_value = float(np.clip(old_value + rng.normal(0.0, proposal_sigma), 0.0, 1.0))
            old_energy = local_energy(old_value, state, guide, confidence, i, j, **energy_args)
            new_energy = local_energy(new_value, state, guide, confidence, i, j, **energy_args)
            delta = new_energy - old_energy
            if delta < 0.0 or rng.random() < math.exp(-delta / float(temperature)):
                state[i, j] = new_value
    return state


def cross_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    foreground_mask: np.ndarray,
) -> dict[str, float | None]:
    """Return reference, region, and perceptual-gradient metrics."""
    metrics = comparison_summary(candidate, reference=reference, foreground_mask=foreground_mask)
    metrics.update(perceptual_gradient_summary(reference, candidate))
    metrics["mean_error_vs_reference"] = float((candidate - reference).mean())
    return metrics


def natural_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    """Return natural-patch metrics plus unnormalized gradient metrics."""
    metrics: dict[str, float | None] = natural_image_metrics(candidate, reference)
    metrics.update(perceptual_gradient_summary(reference, candidate))
    return metrics


def run_case(
    case_name: str,
    reference: np.ndarray,
    low_guide: np.ndarray,
    *,
    decoder_seed: int,
    decode_config: dict[str, float | int],
    foreground_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run baselines and redesign candidates for one case."""
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
    confidences: dict[str, np.ndarray] = {}

    for condition in CONDITIONS:
        condition_id = str(condition["id"])
        confidence = make_confidence(bilinear, str(condition["confidence_mode"]))
        rendered, seconds = timed_call(
            decode,
            bilinear,
            confidence,
            pairwise_mode=str(condition["pairwise_mode"]),
            decoder_seed=decoder_seed,
            **MODEL_PARAMS,
            **decode_config,
        )
        outputs[condition_id] = rendered
        timings[condition_id] = seconds
        confidences[condition_id] = confidence

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
        confidences=confidences,
        baseline_for_diff=bilinear,
    )
    return {"metrics": metrics, "timings": timings}


def save_case_artifacts(
    case_dir: Path,
    *,
    reference: np.ndarray,
    low_guide: np.ndarray,
    outputs: dict[str, np.ndarray],
    confidences: dict[str, np.ndarray],
    baseline_for_diff: np.ndarray,
) -> None:
    """Save principal images and difference maps for one case."""
    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    for name, image in outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
        if name not in {"nearest", "bilinear", "bicubic"}:
            save_grayscale_png(case_dir / f"diff_{name}_vs_bilinear.png", np.abs(image - baseline_for_diff))
            save_grayscale_png(case_dir / f"diff_{name}_vs_reference.png", np.abs(image - reference))
    for name, confidence in confidences.items():
        save_grayscale_png(case_dir / f"confidence_{name}.png", confidence)

    guide_preview = upscale(low_guide, reference.shape[0], 0)
    condition_ids = [str(condition["id"]) for condition in CONDITIONS]
    save_comparison_png(
        case_dir / "comparison.png",
        [
            reference,
            guide_preview,
            outputs["nearest"],
            outputs["bilinear"],
            outputs["bicubic"],
            *[outputs[name] for name in condition_ids],
        ],
    )
    save_comparison_png(
        case_dir / "confidence_comparison.png",
        [confidences[name] for name in condition_ids],
    )


def format_cross_rows(metrics: dict[str, dict[str, float | None]], timings: dict[str, float]) -> str:
    """Format the cross comparison table."""
    names = ["nearest", "bilinear", "bicubic", *[str(condition["id"]) for condition in CONDITIONS]]
    return "\n".join(
        "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {gradient:.6f} | {edge:.6f} | {seconds:.6f} |".format(
            name=name,
            mad=metrics[name]["mad_vs_reference"],
            psnr=metrics[name]["psnr_vs_reference"],
            ssim=metrics[name]["ssim_global_vs_reference"],
            gradient=metrics[name]["gradient_magnitude_mad"],
            edge=metrics[name]["edge_leakage"],
            seconds=timings[name],
        )
        for name in names
    )


def format_natural_rows(metrics: dict[str, dict[str, float | None]], timings: dict[str, float]) -> str:
    """Format the natural-patch comparison table."""
    names = ["nearest", "bilinear", "bicubic", *[str(condition["id"]) for condition in CONDITIONS]]
    return "\n".join(
        "| {name} | {mad:.6f} | {psnr:.3f} | {ssim:.6f} | {gradient:.6f} | {edge:.6f} | {seconds:.6f} |".format(
            name=name,
            mad=metrics[name]["mad_vs_gt"],
            psnr=metrics[name]["psnr_vs_gt"],
            ssim=metrics[name]["ssim_global_vs_gt"],
            gradient=metrics[name]["gradient_magnitude_mad"],
            edge=metrics[name]["strong_edge_mad_vs_gt"],
            seconds=timings[name],
        )
        for name in names
    )


def best_condition(results: dict[str, Any], case_name: str, metric_name: str) -> tuple[str, float]:
    """Return the redesign condition with the smallest selected metric."""
    pairs = [
        (str(condition["id"]), float(results[case_name]["metrics"][str(condition["id"])][metric_name]))
        for condition in CONDITIONS
    ]
    return min(pairs, key=lambda pair: pair[1])


def build_notes(config: dict[str, Any], results: dict[str, Any]) -> str:
    """Build the Japanese experiment note from measured results."""
    cross_rows = format_cross_rows(results["cross"]["metrics"], results["cross"]["timings"])
    natural_rows = format_natural_rows(
        results["natural_patch"]["metrics"],
        results["natural_patch"]["timings"],
    )
    cross_best, cross_best_mad = best_condition(results, "cross", "mad_vs_reference")
    natural_best, natural_best_mad = best_condition(results, "natural_patch", "mad_vs_gt")
    return f"""# Model D Confidence / Pairwise Redesign Candidates

## Question

Issue #61 で現行gradient confidenceがuniform confidenceより悪かった結果を受け、confidence mapの形とpairwise penaltyの形を小さく変更すると、crossと自然画像patchのreference差分は改善するか。

## Hypothesis

現行gradient confidenceより空間変化を弱めた `flatter` またはedge近傍だけを強く拘束する `edge_band` は、低confidence領域でのdriftを減らす可能性がある。また、大きな画素差に対するpairwise penaltyをclampすると、境界をまたぐ過度な平滑化を抑える可能性がある。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_018_model_d_redesign_candidates.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Cross decoder seed: {config["cross_decoder_seed"]}
- Natural patch decoder seed: {config["natural_decoder_seed"]}
- Texture strength: 0.0（全条件）
- Cross: {config["cross_low_size"]}x{config["cross_low_size"]} guideから{config["cross_high_size"]}x{config["cross_high_size"]} output
- Natural patch: {config["natural_low_size"]}x{config["natural_low_size"]} guideから{config["natural_high_size"]}x{config["natural_high_size"]} output
- Conditions: `config.json` の `conditions`
- Model params: `{config["model_params"]}`
- Pairwise redesign: `min((v_i - v_j)^2, pairwise_cap^2)` をclamped条件で使用
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

nearest、bilinear、bicubicを共通baselineとした。`uniform_quadratic` はIssue #61の最良term条件に対応する対照、`current_gradient_quadratic` は現行confidence設計、`flatter_quadratic` と `edge_band_quadratic` はconfidence再設計、`uniform_clamped_pairwise` はpairwise再設計である。

## Metrics

### Cross

| Output | MAD vs reference | PSNR | Global SSIM | Gradient magnitude MAD | Edge leakage | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{cross_rows}

### Natural Patch

| Output | MAD vs GT | PSNR | Global SSIM | Gradient magnitude MAD | Strong-edge MAD | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{natural_rows}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Notes: `notes.md`
- Cross artifacts: `cross/`
- Natural patch artifacts: `natural_patch/`
- 各caseにbaseline、各候補、confidence map、reference/bilinear差分、`comparison.png`、`confidence_comparison.png` を保存した。

## Images

![Cross redesign comparison](cross/comparison.png)

![Cross confidence comparison](cross/confidence_comparison.png)

![Natural patch redesign comparison](natural_patch/comparison.png)

![Natural patch confidence comparison](natural_patch/confidence_comparison.png)

## Result

Model D候補内の最小MADは、crossでは `{cross_best}` の `{cross_best_mad:.6f}`、natural patchでは `{natural_best}` の `{natural_best_mad:.6f}` だった。

## Interpretation

`uniform_clamped_pairwise` はcrossで `uniform_quadratic` よりMADとedge leakageを改善したが、natural patchではMAD、SSIM、gradient magnitude MADが悪化した。`flatter_quadratic` と `edge_band_quadratic` は現行gradient confidenceより良かったものの、両caseで `uniform_quadratic` を一貫して上回らなかった。

さらに、crossではnearest、natural patchではbicubicがMAD、SSIM、gradient magnitude MADの主要値で全Model D候補より良かった。したがって、今回のconfidence 2案とclamped pairwiseをModel D draftへ採用する根拠は得られなかった。これは候補を小さく比較したnegative resultであり、confidence map一般、robust pairwise一般、super-resolution、compressionの可否を示すものではない。

## Limitations

- crossと1枚のpublic-domain自然画像patchだけの比較である。
- confidence候補のfloor、edge-band quantile、pairwise capは各1設定のみであり、広い探索ではない。
- clamped pairwiseはtruncated quadraticの実験候補で、正式な確率モデルや仕様ではない。
- Global SSIMは依存なしの全画像SSIMであり、windowed SSIMではない。
- decode timeはこの環境の小画像runに限る。
- この結果はsuper-resolutionやcompressionの成立を示さない。

## Next

- 今回の候補はModel D draftへ採用せず、negative evidenceとして残す。
- 次にModel Dを進める場合はconfidence floorやcapの小調整より、annealingによる確率的driftを含むrelaxation objectiveまたは更新手順を別Issueで切り分ける。
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
            print(
                f"  {name}: {metric_name}={values[metric_name]:.6f} "
                f"time={case_results['timings'][name]:.6f}s"
            )


if __name__ == "__main__":
    main()
