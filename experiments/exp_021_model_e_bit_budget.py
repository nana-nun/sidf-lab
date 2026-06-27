"""Compare Model E against fixed-feature INR baselines at similar side bits."""

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
from sidf_lab.guides import circle, diagonal
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json
from sidf_lab.metrics import (
    gradient_magnitude,
    gradient_magnitude_correlation,
    gradient_magnitude_mad,
    laplacian_mad,
    mad,
    psnr,
    ssim_global,
)
from sidf_lab.model_e import bilinear_resize, model_e_features


RESULT_DIR = Path("results/2026-06-27-issue-98-model-e-bit-budget")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-06-27"
EXPERIMENT_SEED = 20260627
HIGH_SIZE = 64
LOW_SIZE = 16
PARAMETER_BITS = 12
HEADER_BITS = 160
RIDGE_LAMBDA = 1e-4
RESIDUAL_LIMIT = 0.35
FOLLOW_UP_IMPL = "https://github.com/nana-nun/sidf-lab/issues/103"
FOLLOW_UP_EXP = "https://github.com/nana-nun/sidf-lab/issues/104"


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def natural_patch(index: int) -> np.ndarray:
    source = np.load(SOURCE_ASSET)
    offsets = [(0, 0), (0, 64), (64, 0), (64, 64)]
    row, col = offsets[index]
    return source[row : row + HIGH_SIZE, col : col + HIGH_SIZE]


def make_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "dev_diagonal",
            "split": "development",
            "kind": "synthetic",
            "reference": diagonal(HIGH_SIZE, width=4, value=0.55),
        },
        {
            "name": "dev_natural_tl",
            "split": "development",
            "kind": "natural",
            "reference": natural_patch(0),
        },
        {
            "name": "eval_circle",
            "split": "evaluation",
            "kind": "synthetic",
            "reference": circle(HIGH_SIZE, radius=16.0, value=0.55),
        },
        {
            "name": "eval_natural_br",
            "split": "evaluation",
            "kind": "natural",
            "reference": natural_patch(3),
        },
    ]


def make_candidate_specs() -> list[dict[str, Any]]:
    return [
        {"name": "fourier_low", "family": "fourier", "tier": "low", "order": 1},
        {"name": "fourier_mid", "family": "fourier", "tier": "mid", "order": 2},
        {"name": "rff_low", "family": "rff", "tier": "low", "features": 6, "seed": 1101},
        {"name": "rff_mid", "family": "rff", "tier": "mid", "features": 12, "seed": 1102},
        {"name": "siren_low", "family": "siren", "tier": "low", "features": 6, "seed": 1201},
        {"name": "siren_mid", "family": "siren", "tier": "mid", "features": 12, "seed": 1202},
        {
            "name": "model_e_single_low",
            "family": "model_e_single",
            "tier": "low",
            "depth": 6,
            "states": 1,
            "seed": 1301,
        },
        {
            "name": "model_e_single_mid",
            "family": "model_e_single",
            "tier": "mid",
            "depth": 12,
            "states": 1,
            "seed": 1302,
        },
        {
            "name": "model_e_coupled_low",
            "family": "model_e_coupled",
            "tier": "low",
            "depth": 3,
            "states": 3,
            "seed": 1401,
        },
        {
            "name": "model_e_coupled_mid",
            "family": "model_e_coupled",
            "tier": "mid",
            "depth": 6,
            "states": 3,
            "seed": 1402,
        },
    ]


def feature_matrix(spec: dict[str, Any], low_guide: np.ndarray, output_shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    features = model_e_features(low_guide, output_shape)
    x = features[..., 0]
    y = features[..., 1]
    base = features[..., 2]
    gx = features[..., 3]
    gy = features[..., 4]
    family = spec["family"]
    metadata: dict[str, Any] = {}
    if family == "fourier":
        columns = [np.ones_like(x)]
        for freq in range(1, int(spec["order"]) + 1):
            columns.extend(
                [
                    np.sin(math.pi * freq * x),
                    np.cos(math.pi * freq * x),
                    np.sin(math.pi * freq * y),
                    np.cos(math.pi * freq * y),
                    np.sin(math.pi * freq * (x + y)),
                    np.cos(math.pi * freq * (x - y)),
                ]
            )
        metadata["stored_feature_values"] = 0
        return np.stack(columns, axis=-1).reshape(-1, len(columns)), metadata
    if family in {"rff", "siren"}:
        rng = np.random.default_rng(int(spec["seed"]))
        count = int(spec["features"])
        inputs = np.stack([x, y, base, gx, gy], axis=-1)
        if family == "rff":
            weights = rng.normal(0.0, 3.0, size=(5, count))
            bias = rng.uniform(-math.pi, math.pi, size=count)
            projected = np.tensordot(inputs, weights, axes=([-1], [0])) + bias
            columns = np.concatenate([np.sin(projected), np.cos(projected)], axis=-1)
            metadata["stored_feature_values"] = int(weights.size + bias.size)
        else:
            weights = rng.normal(0.0, 2.0, size=(5, count))
            bias = rng.uniform(-1.0, 1.0, size=count)
            columns = np.sin(6.0 * (np.tensordot(inputs, weights, axes=([-1], [0])) + bias))
            metadata["stored_feature_values"] = int(weights.size + bias.size)
        return columns.reshape(-1, columns.shape[-1]), metadata
    if family in {"model_e_single", "model_e_coupled"}:
        states, stored_values = model_e_state_features(
            features,
            depth=int(spec["depth"]),
            states=int(spec["states"]),
            coupled=family == "model_e_coupled",
            seed=int(spec["seed"]),
        )
        metadata["stored_feature_values"] = stored_values
        return states.reshape(-1, states.shape[-1]), metadata
    raise ValueError(f"unknown family: {family}")


def model_e_state_features(
    features: np.ndarray,
    *,
    depth: int,
    states: int,
    coupled: bool,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    layers = rng.normal(0.0, 1.25, size=(depth, states, features.shape[-1]))
    state = np.zeros((*features.shape[:2], states), dtype=np.float64)
    state[..., 0] = 1.0
    if states > 1:
        state[..., 1:] = -1.0 / states
    for layer in layers:
        angles = np.tensordot(features, layer, axes=([-1], [-1]))
        rotated = np.sin(angles + state)
        if coupled:
            rotated = rotated + 0.25 * np.roll(rotated, 1, axis=-1) * np.roll(rotated, -1, axis=-1)
        state = np.tanh(rotated)
    return state, int(layers.size)


def fit_readout(matrix: np.ndarray, residual: np.ndarray) -> np.ndarray:
    target = np.clip(residual.reshape(-1), -RESIDUAL_LIMIT, RESIDUAL_LIMIT)
    gram = matrix.T @ matrix
    rhs = matrix.T @ target
    return np.linalg.solve(gram + RIDGE_LAMBDA * np.eye(gram.shape[0]), rhs)


def quantize(values: np.ndarray, bits: int = PARAMETER_BITS) -> np.ndarray:
    levels = (1 << bits) - 1
    clipped = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
    return np.rint((clipped + 1.0) * 0.5 * levels).astype(np.int64)


def dequantize(values: np.ndarray, bits: int = PARAMETER_BITS) -> np.ndarray:
    levels = (1 << bits) - 1
    return np.asarray(values, dtype=np.float64) / levels * 2.0 - 1.0


def side_bits(metadata: dict[str, Any], readout_count: int) -> int:
    return int(HEADER_BITS + (int(metadata["stored_feature_values"]) + readout_count) * PARAMETER_BITS)


def candidate_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    return {
        "mad_vs_gt": mad(candidate, reference),
        "psnr_vs_gt": psnr(reference, candidate),
        "ssim_global_vs_gt": ssim_global(reference, candidate),
        "gradient_magnitude_mad": gradient_magnitude_mad(reference, candidate),
        "gradient_magnitude_correlation": gradient_magnitude_correlation(reference, candidate),
        "laplacian_mad": laplacian_mad(reference, candidate),
    }


def artifact_summary(image: np.ndarray) -> dict[str, float]:
    values = np.asarray(image, dtype=np.float64)
    lap = (
        np.pad(values, 1, mode="edge")[:-2, 1:-1]
        + np.pad(values, 1, mode="edge")[2:, 1:-1]
        + np.pad(values, 1, mode="edge")[1:-1, :-2]
        + np.pad(values, 1, mode="edge")[1:-1, 2:]
        - 4.0 * values
    )
    grad = gradient_magnitude(values)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "gradient_magnitude_mean": float(grad.mean()),
        "gradient_magnitude_max": float(grad.max()),
        "laplacian_abs_mean": float(np.mean(np.abs(lap))),
    }


def run_candidate(spec: dict[str, Any], reference: np.ndarray, low_guide: np.ndarray) -> dict[str, Any]:
    base = bilinear_resize(low_guide, reference.shape)
    residual = reference - base
    start_fit = time.perf_counter()
    matrix, metadata = feature_matrix(spec, low_guide, reference.shape)
    readout = fit_readout(matrix, residual)
    fit_seconds = float(time.perf_counter() - start_fit)

    readout_q = quantize(readout)
    readout_restored = dequantize(readout_q)
    start_decode = time.perf_counter()
    quantized_residual = matrix @ readout_restored
    decoded = np.clip(base + quantized_residual.reshape(reference.shape), 0.0, 1.0)
    decode_seconds = float(time.perf_counter() - start_decode)

    float_decoded = np.clip(base + (matrix @ readout).reshape(reference.shape), 0.0, 1.0)
    quant_metrics = candidate_metrics(decoded, reference)
    float_metrics = candidate_metrics(float_decoded, reference)
    return {
        "spec": spec,
        "image": decoded,
        "float_image": float_decoded,
        "metrics": quant_metrics,
        "float_metrics": float_metrics,
        "fit_seconds": fit_seconds,
        "decode_seconds": decode_seconds,
        "serialized_bits": side_bits(metadata, readout.size),
        "bits_per_output_pixel": side_bits(metadata, readout.size) / float(reference.size),
        "parameter_count_for_bits": int(metadata["stored_feature_values"] + readout.size),
        "readout_count": int(readout.size),
        "stored_feature_values": int(metadata["stored_feature_values"]),
        "float_to_quantized_mad_delta": quant_metrics["mad_vs_gt"] - float_metrics["mad_vs_gt"],
        "readout_restored": readout_restored,
    }


def decode_parameterized(spec: dict[str, Any], readout: np.ndarray, low_guide: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    base = bilinear_resize(low_guide, output_shape)
    matrix, _metadata = feature_matrix(spec, low_guide, output_shape)
    residual = matrix @ readout
    return np.clip(base + residual.reshape(output_shape), 0.0, 1.0)


def run_case(case: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, Any]:
    case_dir = ensure_dir(RESULT_DIR / case["name"])
    reference = np.asarray(case["reference"], dtype=np.float64)
    low_guide = downscale_block_average(reference, LOW_SIZE)
    nearest, nearest_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 3)
    baseline_outputs = {
        "nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
    }
    baseline_metrics = {
        name: candidate_metrics(image, reference) for name, image in baseline_outputs.items()
    }
    baseline_timings = {
        "nearest": nearest_seconds,
        "bilinear": bilinear_seconds,
        "bicubic": bicubic_seconds,
    }
    candidate_results = {spec["name"]: run_candidate(spec, reference, low_guide) for spec in specs}

    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    for name, image in baseline_outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
    best_name = min(candidate_results, key=lambda name: candidate_results[name]["metrics"]["mad_vs_gt"])
    best_image = candidate_results[best_name]["image"]
    best_float = candidate_results[best_name]["float_image"]
    save_grayscale_png(case_dir / f"{best_name}.png", best_image)
    save_grayscale_png(case_dir / f"{best_name}_float.png", best_float)
    save_grayscale_png(case_dir / f"diff_{best_name}_vs_gt.png", np.abs(best_image - reference))
    save_grayscale_png(case_dir / "diff_bicubic_vs_gt.png", np.abs(bicubic - reference))
    save_comparison_png(
        case_dir / "comparison.png",
        [reference, nearest, bilinear, bicubic, best_image, np.abs(best_image - reference)],
    )
    extrapolated: dict[str, Any] = {}
    if case["name"] == "eval_natural_br":
        extra_shape = (HIGH_SIZE * 2, HIGH_SIZE * 2)
        extra_images = [upscale(low_guide, extra_shape[0], 3)]
        extra_names = ["bicubic"]
        for name in ["rff_mid", "model_e_single_mid", "model_e_coupled_mid"]:
            result = candidate_results[name]
            image = decode_parameterized(
                result["spec"],
                result["readout_restored"],
                low_guide,
                extra_shape,
            )
            save_grayscale_png(case_dir / f"extrapolated_{name}.png", image)
            extrapolated[name] = artifact_summary(image)
            extra_images.append(image)
            extra_names.append(name)
        save_grayscale_png(case_dir / "extrapolated_bicubic.png", extra_images[0])
        extrapolated["bicubic"] = artifact_summary(extra_images[0])
        save_comparison_png(case_dir / "extrapolated_comparison.png", extra_images)
    return {
        "name": case["name"],
        "split": case["split"],
        "kind": case["kind"],
        "baselines": {
            name: {
                "metrics": baseline_metrics[name],
                "decode_seconds": baseline_timings[name],
            }
            for name in baseline_outputs
        },
        "candidates": {
            name: {
                key: value
                for key, value in result.items()
                if key not in {"image", "float_image", "readout_restored"}
            }
            for name, result in candidate_results.items()
        },
        "best_candidate": best_name,
        "extrapolated": extrapolated,
    }


def aggregate_by_split(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ["development", "evaluation"]:
        selected = [case for case in case_results if case["split"] == split]
        names = sorted(selected[0]["candidates"]) if selected else []
        output[split] = {}
        for name in names:
            values = [case["candidates"][name]["metrics"]["mad_vs_gt"] for case in selected]
            bits = [case["candidates"][name]["serialized_bits"] for case in selected]
            output[split][name] = {
                "mean_mad_vs_gt": float(np.mean(values)),
                "mean_serialized_bits": float(np.mean(bits)),
            }
        for baseline in ["nearest", "bilinear", "bicubic"]:
            values = [case["baselines"][baseline]["metrics"]["mad_vs_gt"] for case in selected]
            output[split][baseline] = {"mean_mad_vs_gt": float(np.mean(values))}
    return output


def markdown_rows(case_results: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    rows = []
    for split in ["development", "evaluation"]:
        for name, values in sorted(aggregate[split].items()):
            bits = values.get("mean_serialized_bits", None)
            rows.append(
                "| {split} | {name} | {bits} | {mad:.6f} |".format(
                    split=split,
                    name=name,
                    bits="N/A" if bits is None else f"{bits:.0f}",
                    mad=values["mean_mad_vs_gt"],
                )
            )
    return "\n".join(rows)


def extrapolated_rows(case_results: list[dict[str, Any]]) -> str:
    selected = next(case for case in case_results if case["name"] == "eval_natural_br")
    rows = []
    for name, values in sorted(selected["extrapolated"].items()):
        rows.append(
            "| {name} | {grad_mean:.6f} | {grad_max:.6f} | {lap:.6f} | {minv:.6f} | {maxv:.6f} |".format(
                name=name,
                grad_mean=values["gradient_magnitude_mean"],
                grad_max=values["gradient_magnitude_max"],
                lap=values["laplacian_abs_mean"],
                minv=values["min"],
                maxv=values["max"],
            )
        )
    return "\n".join(rows)


def build_notes(config: dict[str, Any], case_results: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    best_eval = min(
        (item for item in aggregate["evaluation"].items() if "mean_serialized_bits" in item[1]),
        key=lambda item: item[1]["mean_mad_vs_gt"],
    )
    bicubic_eval = aggregate["evaluation"]["bicubic"]["mean_mad_vs_gt"]
    return f"""# Model E Bit-Budget INR Comparison

## Question

量子回路由来のModel E座標関数は、同程度の量子化後serialized side bitsを持つclassical implicit residual baselineより、低解像度guideから失われた画像残差を効率よく表現できるか。

## Hypothesis

Model E coupled-state候補は、single-state候補より2次元の交差構造を表現しやすく、Fourier / RFF / small SIREN residual baselineと同程度の保存bit数でMADを下げる可能性がある。ただし、量子回路由来であること自体は採用理由にならず、evaluation splitでclassical baselineに支配される場合はnegative resultとして扱う。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_021_model_e_bit_budget.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Output size: {config["high_size"]}x{config["high_size"]}
- Low guide size: {config["low_size"]}x{config["low_size"]}
- Low guide method: 4x4 block average from 64x64 reference
- Parameter quantization: signed uniform {config["parameter_bits"]}-bit values in `[-1, 1]`
- Header bits per parameterized model: {config["header_bits"]}
- Fit protocol: fixed feature dictionary plus ridge least-squares readout, then quantized readout decode
- Ridge lambda: {config["ridge_lambda"]}
- Residual clamp during fit: `{config["residual_limit"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

画像baselineはnearest、bilinear、bicubic。Parameterized residual baselineはFourier、RFF、small SIREN、Model E single-state、Model E coupled-state。すべて同じlow guide、同じreference、同じfixed-feature + least-squares readout protocolでfitした。

## Result

| Split | Output | Mean serialized side bits | Mean MAD vs GT |
| --- | --- | ---: | ---: |
{markdown_rows(case_results, aggregate)}

Evaluation splitで最小MADのparameterized候補は `{best_eval[0]}` で、mean MAD `{best_eval[1]["mean_mad_vs_gt"]:.6f}`、mean serialized side bits `{best_eval[1]["mean_serialized_bits"]:.0f}` だった。bicubic baselineのevaluation mean MADは `{bicubic_eval:.6f}` だった。

### Extrapolated Output Diagnostic

`eval_natural_br` の同じlow guideと保存parameterを128x128座標へ評価した。これは128x128 Ground Truthとの比較ではなく、周期artifactや局所高周波差を目視・統計確認するための診断である。

| Output | Gradient magnitude mean | Gradient magnitude max | Laplacian abs mean | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
{extrapolated_rows(case_results)}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Notes: `notes.md`
- Case directories: `dev_diagonal/`, `dev_natural_tl/`, `eval_circle/`, `eval_natural_br/`
- Per-case main images: `high_reference.png`, `low_guide.png`, `nearest.png`, `bilinear.png`, `bicubic.png`, best candidate PNG, `comparison.png`, difference maps
- Extrapolated output check: `eval_natural_br/extrapolated_bicubic.png`, `eval_natural_br/extrapolated_rff_mid.png`, `eval_natural_br/extrapolated_model_e_single_mid.png`, `eval_natural_br/extrapolated_model_e_coupled_mid.png`, `eval_natural_br/extrapolated_comparison.png`

## Images

![Development diagonal comparison](dev_diagonal/comparison.png)

![Development natural patch comparison](dev_natural_tl/comparison.png)

![Evaluation circle comparison](eval_circle/comparison.png)

![Evaluation natural patch comparison](eval_natural_br/comparison.png)

![Evaluation natural patch extrapolated outputs](eval_natural_br/extrapolated_comparison.png)

## Interpretation

このrunでは、Model E候補がevaluation splitでclassical INR baselineを一貫して上回るとは解釈しない。特に、fixed-feature + least-squares readout条件では、evaluationの最良parameterized候補とbicubic baselineの差を分けて読む必要がある。

今回の結果は、Model Eの最小候補をSIDF draft specificationへ採用する根拠ではない。一方で、quantized serialized side bits、float-to-quantized delta、fit time、decode timeを同じ形式で保存できたため、次の改善候補を比較する土台にはなる。

## Limitations

- 4ケースだけの小規模runであり、画像集合全体を代表しない。
- Model Eとclassical INRはいずれもfixed feature dictionary + linear readoutに制限した。全parameterを非線形最適化した結果ではない。
- small SIREN baselineは固定sine特徴 + linear readoutであり、通常のmulti-layer SIREN trainingではない。
- serialized bitsはparameter side informationの簡易見積もりであり、complete SIDF bitstream、guide bits、entropy coding、container overheadを含まない。
- extrapolated outputは同じlow guideと保存parameterを128x128座標へ評価した診断であり、128x128 Ground Truthに対する品質測定ではない。
- Global SSIMはwindowed SSIMではない。
- compression、super-resolution、quantum advantageは主張しない。

## Next

- Model Eを継続する場合は、全parameter optimizationまたはModel E特有のfrequency/angle parameterizationを改善し、同じこのprotocolで再比較する。
- classical baseline側は、fixed-feature SIRENではなく小型trainable SIREN/MLPを同じserialized bit accountingで追加する。
- datasetを増やす場合は、開発用と評価用のcrop由来が混ざらないようにsource image単位で分割する。
- Follow-up implementation: [#103]({FOLLOW_UP_IMPL})
- Follow-up experiment: [#104]({FOLLOW_UP_EXP})
"""


def main() -> None:
    ensure_dir(RESULT_DIR)
    specs = make_candidate_specs()
    case_results = [run_case(case, specs) for case in make_cases()]
    aggregate = aggregate_by_split(case_results)
    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "source_asset": SOURCE_ASSET.as_posix(),
        "high_size": HIGH_SIZE,
        "low_size": LOW_SIZE,
        "parameter_bits": PARAMETER_BITS,
        "header_bits": HEADER_BITS,
        "ridge_lambda": RIDGE_LAMBDA,
        "residual_limit": RESIDUAL_LIMIT,
        "candidate_specs": specs,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", {"cases": case_results, "aggregate": aggregate})
    (RESULT_DIR / "notes.md").write_text(build_notes(config, case_results, aggregate), encoding="utf-8")
    for split, rows in aggregate.items():
        print(split)
        for name, values in sorted(rows.items()):
            bits = values.get("mean_serialized_bits", None)
            bit_text = "N/A" if bits is None else f"{bits:.0f}"
            print(f"  {name}: bits={bit_text} mad={values['mean_mad_vs_gt']:.6f}")


if __name__ == "__main__":
    main()
