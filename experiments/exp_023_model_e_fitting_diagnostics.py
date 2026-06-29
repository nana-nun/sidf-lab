"""Diagnose Model E fitting convergence on the source-split patch fixture."""

from __future__ import annotations

import csv
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
from sidf_lab.inr_fit import (
    INRSpec,
    decode_inr,
    dequantize_vector,
    estimate_inr_bits,
    fit_inr,
    flatten_parameters,
    initialize_parameters,
    make_parameter_layout,
    quantize_vector,
    unflatten_parameters,
)
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json
from sidf_lab.metrics import (
    gradient_magnitude_correlation,
    gradient_magnitude_mad,
    laplacian_mad,
    mad,
    psnr,
    ssim_global,
)
from sidf_lab.model_e import QuantizationSpec
from sidf_lab.patch_fixtures import load_grayscale_patch, load_patch_manifest, list_patch_records


RESULT_DIR = Path("results/2026-06-29-issue-117-model-e-fitting-diagnostics")
DATE = "2026-06-29"
EXPERIMENT_SEED = 20260629
HIGH_SIZE = 64
LOW_SIZE = 16
RANDOM_STEPS = 24
ADAM_STEPS = 18
LBFGS_STEPS = 10
FINITE_DIFF_EPSILON = 1.0e-4
PARAMETER_BITS = 12
QUANTIZATION = QuantizationSpec(bits_per_value=PARAMETER_BITS, min_value=-1.0, max_value=1.0, header_bits=160)


def stable_case_offset(name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(name))


def make_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for record in list_patch_records():
        source = load_grayscale_patch(record["name"])
        for suffix, row, col in [("tl", 0, 0), ("br", 64, 64)]:
            cases.append(
                {
                    "name": f"{record['split']}_{record['source_id']}_{suffix}",
                    "split": record["split"],
                    "source_id": record["source_id"],
                    "patch_name": record["name"],
                    "crop": {"row": row, "col": col, "height": HIGH_SIZE, "width": HIGH_SIZE},
                    "reference": source[row : row + HIGH_SIZE, col : col + HIGH_SIZE],
                }
            )
    return cases


def baseline_specs() -> list[dict[str, Any]]:
    return [
        {"name": "rff_small", "spec": INRSpec("rff", feature_count=4), "seed": 5102},
        {"name": "siren_small", "spec": INRSpec("siren", feature_count=6), "seed": 5103},
        {"name": "mlp_small", "spec": INRSpec("mlp", feature_count=6), "seed": 5104},
    ]


def model_e_specs() -> list[dict[str, Any]]:
    return [
        {"name": "model_e_single", "spec": INRSpec("model_e_single", depth=6, states=1), "seed": 5105},
        {"name": "model_e_coupled", "spec": INRSpec("model_e_coupled", depth=3, states=3), "seed": 5106},
    ]


def image_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    return {
        "mad_vs_gt": mad(candidate, reference),
        "psnr_vs_gt": psnr(reference, candidate),
        "ssim_global_vs_gt": ssim_global(reference, candidate),
        "gradient_magnitude_mad": gradient_magnitude_mad(reference, candidate),
        "gradient_magnitude_correlation": gradient_magnitude_correlation(reference, candidate),
        "laplacian_mad": laplacian_mad(reference, candidate),
    }


def mse_for_params(spec: INRSpec, low_guide: np.ndarray, reference: np.ndarray, params: np.ndarray) -> float:
    decoded = decode_inr(spec, low_guide, reference.shape, params)
    diff = decoded - reference
    return float(np.mean(diff * diff))


def finite_difference_gradient(
    spec: INRSpec,
    low_guide: np.ndarray,
    reference: np.ndarray,
    params: np.ndarray,
    *,
    epsilon: float = FINITE_DIFF_EPSILON,
) -> tuple[float, np.ndarray]:
    base_loss = mse_for_params(spec, low_guide, reference, params)
    gradient = np.zeros_like(params)
    for index in range(params.size):
        shifted = params.copy()
        shifted[index] += epsilon
        gradient[index] = (mse_for_params(spec, low_guide, reference, shifted) - base_loss) / epsilon
    return base_loss, gradient


def make_initial_params(spec: INRSpec, seed: int, mode: str) -> np.ndarray:
    params = initialize_parameters(spec, seed=seed)
    if mode == "default":
        return params
    if mode != "small_layers":
        raise ValueError(f"unsupported initialization mode: {mode}")
    layout = make_parameter_layout(spec)
    named = unflatten_parameters(params, layout)
    named["layers"] = 0.45 * named["layers"]
    named["residual_scale"] = np.minimum(named["residual_scale"], 0.08)
    return flatten_parameters(named, layout)


def random_search_trace(
    spec: INRSpec,
    low_guide: np.ndarray,
    reference: np.ndarray,
    params: np.ndarray,
    *,
    seed: int,
    steps: int = RANDOM_STEPS,
    initial_step_scale: float = 0.035,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    current = params.copy()
    current_loss = mse_for_params(spec, low_guide, reference, current)
    trace: list[dict[str, float | int]] = [
        {"step": 0, "loss": current_loss, "gradient_norm": math.nan, "update_norm": 0.0, "accepted": 1}
    ]
    for step in range(1, steps + 1):
        decay = 1.0 - ((step - 1) / max(steps, 1))
        proposal = current + rng.normal(0.0, initial_step_scale * max(decay, 0.05), size=current.shape)
        proposal_loss = mse_for_params(spec, low_guide, reference, proposal)
        accepted = proposal_loss <= current_loss
        update_norm = 0.0
        if accepted:
            update_norm = float(np.linalg.norm(proposal - current))
            current = proposal
            current_loss = proposal_loss
        trace.append(
            {
                "step": step,
                "loss": current_loss,
                "gradient_norm": math.nan,
                "update_norm": update_norm,
                "accepted": int(accepted),
            }
        )
    return current, trace


def adam_trace(
    spec: INRSpec,
    low_guide: np.ndarray,
    reference: np.ndarray,
    params: np.ndarray,
    *,
    steps: int = ADAM_STEPS,
    learning_rate: float = 0.08,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    current = params.copy()
    first_moment = np.zeros_like(current)
    second_moment = np.zeros_like(current)
    beta1 = 0.9
    beta2 = 0.999
    trace: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        loss, gradient = finite_difference_gradient(spec, low_guide, reference, current)
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient * gradient)
        corrected_first = first_moment / (1.0 - beta1**step)
        corrected_second = second_moment / (1.0 - beta2**step)
        update = -learning_rate * corrected_first / (np.sqrt(corrected_second) + 1.0e-8)
        candidate = np.clip(current + update, -1.5, 1.5)
        current = candidate
        trace.append(
            {
                "step": step,
                "loss": loss,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "update_norm": float(np.linalg.norm(update)),
                "accepted": 1,
            }
        )
    final_loss, final_gradient = finite_difference_gradient(spec, low_guide, reference, current)
    trace.append(
        {
            "step": steps + 1,
            "loss": final_loss,
            "gradient_norm": float(np.linalg.norm(final_gradient)),
            "update_norm": 0.0,
            "accepted": 1,
        }
    )
    return current, trace


def lbfgs_like_trace(
    spec: INRSpec,
    low_guide: np.ndarray,
    reference: np.ndarray,
    params: np.ndarray,
    *,
    steps: int = LBFGS_STEPS,
    history_size: int = 5,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    current = params.copy()
    loss, gradient = finite_difference_gradient(spec, low_guide, reference, current)
    s_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    trace: list[dict[str, float | int]] = [
        {"step": 0, "loss": loss, "gradient_norm": float(np.linalg.norm(gradient)), "update_norm": 0.0, "accepted": 1}
    ]
    for step in range(1, steps + 1):
        direction = _lbfgs_direction(gradient, s_history, y_history)
        if float(np.dot(direction, gradient)) >= 0.0:
            direction = -gradient
        step_size = 0.6
        accepted = False
        next_params = current.copy()
        next_loss = loss
        for _ in range(8):
            candidate = np.clip(current + step_size * direction, -1.5, 1.5)
            candidate_loss = mse_for_params(spec, low_guide, reference, candidate)
            if candidate_loss <= loss:
                next_params = candidate
                next_loss = candidate_loss
                accepted = True
                break
            step_size *= 0.5
        next_loss, next_gradient = finite_difference_gradient(spec, low_guide, reference, next_params)
        s_vec = next_params - current
        y_vec = next_gradient - gradient
        if accepted and float(np.dot(s_vec, y_vec)) > 1.0e-12:
            s_history.append(s_vec)
            y_history.append(y_vec)
            s_history = s_history[-history_size:]
            y_history = y_history[-history_size:]
        current = next_params
        loss = next_loss
        gradient = next_gradient
        trace.append(
            {
                "step": step,
                "loss": loss,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "update_norm": float(np.linalg.norm(s_vec)),
                "accepted": int(accepted),
            }
        )
    return current, trace


def _lbfgs_direction(gradient: np.ndarray, s_history: list[np.ndarray], y_history: list[np.ndarray]) -> np.ndarray:
    q = gradient.copy()
    alphas: list[float] = []
    rhos: list[float] = []
    for s_vec, y_vec in reversed(list(zip(s_history, y_history))):
        rho = 1.0 / float(np.dot(y_vec, s_vec))
        alpha = rho * float(np.dot(s_vec, q))
        q = q - alpha * y_vec
        alphas.append(alpha)
        rhos.append(rho)
    if s_history:
        last_s = s_history[-1]
        last_y = y_history[-1]
        scale = float(np.dot(last_s, last_y) / max(float(np.dot(last_y, last_y)), 1.0e-12))
    else:
        scale = 1.0
    r = scale * q
    for s_vec, y_vec, alpha, rho in zip(s_history, y_history, reversed(alphas), reversed(rhos)):
        beta = rho * float(np.dot(y_vec, r))
        r = r + s_vec * (alpha - beta)
    return -r


def save_trace_csv(path: Path, trace: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "gradient_norm", "update_norm", "accepted"])
        writer.writeheader()
        writer.writerows(trace)


def save_loss_curve_png(path: Path, traces: dict[str, list[dict[str, float | int]]]) -> None:
    height = 80
    width = 180
    canvas = np.ones((height, width), dtype=np.float64)
    values = [float(row["loss"]) for trace in traces.values() for row in trace if np.isfinite(float(row["loss"]))]
    if not values:
        save_grayscale_png(path, canvas)
        return
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 1.0e-12)
    shades = [0.15, 0.35, 0.55, 0.75]
    for shade, trace in zip(shades, traces.values()):
        points = []
        for index, row in enumerate(trace):
            x = int(round(index * (width - 1) / max(len(trace) - 1, 1)))
            y = int(round((height - 1) - ((float(row["loss"]) - lo) / span) * (height - 1)))
            points.append((x, y))
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for t in range(steps + 1):
                x = int(round(x0 + (x1 - x0) * t / steps))
                y = int(round(y0 + (y1 - y0) * t / steps))
                canvas[max(0, min(height - 1, y)), max(0, min(width - 1, x))] = shade
    save_grayscale_png(path, canvas)


def finalize_candidate(
    spec: INRSpec,
    low_guide: np.ndarray,
    reference: np.ndarray,
    params: np.ndarray,
) -> dict[str, Any]:
    quantized = quantize_vector(params, QUANTIZATION)
    restored = dequantize_vector(quantized, QUANTIZATION)
    float_decoded = decode_inr(spec, low_guide, reference.shape, params)
    quantized_decoded = decode_inr(spec, low_guide, reference.shape, restored)
    bits = estimate_inr_bits(spec, QUANTIZATION)
    return {
        "parameters": params.copy(),
        "restored_parameters": restored.copy(),
        "float_decoded": float_decoded,
        "quantized_decoded": quantized_decoded,
        "float_metrics": image_metrics(float_decoded, reference),
        "quantized_metrics": image_metrics(quantized_decoded, reference),
        "bits": bits,
    }


def run_model_e_diagnostics(case_dir: Path, case: dict[str, Any], low_guide: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for record in model_e_specs():
        spec = record["spec"]
        seed = record["seed"] + stable_case_offset(case["name"])
        methods = {
            "random_default": ("default", lambda p: random_search_trace(spec, low_guide, reference, p, seed=seed)),
            "adam_default": ("default", lambda p: adam_trace(spec, low_guide, reference, p)),
            "adam_small_init": ("small_layers", lambda p: adam_trace(spec, low_guide, reference, p)),
            "lbfgs_like_default": ("default", lambda p: lbfgs_like_trace(spec, low_guide, reference, p)),
        }
        family_results: dict[str, Any] = {}
        curve_traces: dict[str, list[dict[str, float | int]]] = {}
        for method_name, (init_mode, optimizer) in methods.items():
            initial_params = make_initial_params(spec, seed, init_mode)
            start = time.perf_counter()
            fitted_params, trace = optimizer(initial_params)
            fit_seconds = float(time.perf_counter() - start)
            finalized = finalize_candidate(spec, low_guide, reference, fitted_params)
            output_name = f"{record['name']}_{method_name}"
            save_grayscale_png(case_dir / f"{output_name}_quantized.png", finalized["quantized_decoded"])
            save_grayscale_png(case_dir / f"diff_{output_name}_vs_gt.png", np.abs(finalized["quantized_decoded"] - reference))
            save_trace_csv(case_dir / f"{output_name}_trace.csv", trace)
            curve_traces[method_name] = trace
            family_results[method_name] = {
                "family": spec.family,
                "initialization": init_mode,
                "fit_seconds": fit_seconds,
                "final_loss": float(trace[-1]["loss"]),
                "initial_loss": float(trace[0]["loss"]),
                "final_gradient_norm": float(trace[-1]["gradient_norm"])
                if np.isfinite(float(trace[-1]["gradient_norm"]))
                else None,
                "mean_update_norm": float(np.mean([float(row["update_norm"]) for row in trace])),
                "accepted_steps": int(sum(int(row["accepted"]) for row in trace)),
                "serialized_bits": finalized["bits"]["incremental_side_bits"],
                "parameter_count": finalized["bits"]["parameter_count"],
                "float_metrics": finalized["float_metrics"],
                "quantized_metrics": finalized["quantized_metrics"],
                "float_to_quantized_mad_delta": finalized["quantized_metrics"]["mad_vs_gt"]
                - finalized["float_metrics"]["mad_vs_gt"],
                "image": finalized["quantized_decoded"],
            }
        save_loss_curve_png(case_dir / f"{record['name']}_loss_curves.png", curve_traces)
        diagnostics[record["name"]] = family_results
    return diagnostics


def run_baselines(case: dict[str, Any], low_guide: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for record in baseline_specs():
        result = fit_inr(
            record["spec"],
            low_guide,
            reference,
            seed=record["seed"] + stable_case_offset(case["name"]),
            steps=48,
            initial_step_scale=0.035,
            quantization=QUANTIZATION,
        )
        outputs[record["name"]] = {
            "family": result.spec.family,
            "fit_seconds": result.fit_seconds,
            "decode_seconds": result.decode_seconds,
            "serialized_bits": result.bits["incremental_side_bits"],
            "parameter_count": result.bits["parameter_count"],
            "float_metrics": image_metrics(result.float_decoded, reference),
            "quantized_metrics": image_metrics(result.quantized_decoded, reference),
        }
    return outputs


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    case_dir = ensure_dir(RESULT_DIR / case["name"])
    reference = np.asarray(case["reference"], dtype=np.float64)
    low_guide = downscale_block_average(reference, LOW_SIZE)
    nearest = upscale(low_guide, HIGH_SIZE, 0)
    bilinear = upscale(low_guide, HIGH_SIZE, 1)
    bicubic = upscale(low_guide, HIGH_SIZE, 3)
    baselines = {
        "nearest": {"metrics": image_metrics(nearest, reference)},
        "bilinear": {"metrics": image_metrics(bilinear, reference)},
        "bicubic": {"metrics": image_metrics(bicubic, reference)},
    }
    parameterized_baselines = run_baselines(case, low_guide, reference)
    diagnostics = run_model_e_diagnostics(case_dir, case, low_guide, reference)

    best_model_e_name, best_method, best_result = best_model_e_result(diagnostics)
    best_image = best_result["image"]
    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    save_grayscale_png(case_dir / "nearest.png", nearest)
    save_grayscale_png(case_dir / "bilinear.png", bilinear)
    save_grayscale_png(case_dir / "bicubic.png", bicubic)
    save_grayscale_png(case_dir / "best_model_e_quantized.png", best_image)
    save_grayscale_png(case_dir / "diff_best_model_e_vs_gt.png", np.abs(best_image - reference))
    save_comparison_png(
        case_dir / "comparison.png",
        [reference, nearest, bilinear, bicubic, best_image, np.abs(best_image - reference)],
    )

    return {
        "name": case["name"],
        "split": case["split"],
        "source_id": case["source_id"],
        "patch_name": case["patch_name"],
        "crop": case["crop"],
        "baselines": baselines,
        "parameterized_baselines": parameterized_baselines,
        "model_e_diagnostics": strip_images(diagnostics),
        "best_model_e": {"model": best_model_e_name, "method": best_method},
    }


def best_model_e_result(diagnostics: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    best: tuple[str, str, dict[str, Any]] | None = None
    for model_name, methods in diagnostics.items():
        for method_name, values in methods.items():
            if best is None or values["quantized_metrics"]["mad_vs_gt"] < best[2]["quantized_metrics"]["mad_vs_gt"]:
                best = (model_name, method_name, values)
    if best is None:
        raise ValueError("no diagnostics to select from")
    return best


def strip_images(diagnostics: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for model_name, methods in diagnostics.items():
        clean[model_name] = {}
        for method_name, values in methods.items():
            clean[model_name][method_name] = {key: value for key, value in values.items() if key != "image"}
    return clean


def aggregate(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["development", "evaluation"]:
        selected = [case for case in case_results if case["split"] == split]
        out[split] = {}
        if not selected:
            continue
        for name in ["nearest", "bilinear", "bicubic"]:
            values = [case["baselines"][name]["metrics"]["mad_vs_gt"] for case in selected]
            out[split][name] = {"family": "image", "mean_quantized_mad": float(np.mean(values))}
        baseline_names = selected[0]["parameterized_baselines"].keys()
        for name in baseline_names:
            values = [case["parameterized_baselines"][name]["quantized_metrics"]["mad_vs_gt"] for case in selected]
            float_values = [case["parameterized_baselines"][name]["float_metrics"]["mad_vs_gt"] for case in selected]
            out[split][name] = {
                "family": selected[0]["parameterized_baselines"][name]["family"],
                "mean_float_mad": float(np.mean(float_values)),
                "mean_quantized_mad": float(np.mean(values)),
                "mean_serialized_bits": float(np.mean([case["parameterized_baselines"][name]["serialized_bits"] for case in selected])),
                "mean_fit_seconds": float(np.mean([case["parameterized_baselines"][name]["fit_seconds"] for case in selected])),
            }
        for model_name in selected[0]["model_e_diagnostics"].keys():
            for method_name in selected[0]["model_e_diagnostics"][model_name].keys():
                key = f"{model_name}_{method_name}"
                records = [case["model_e_diagnostics"][model_name][method_name] for case in selected]
                out[split][key] = {
                    "family": records[0]["family"],
                    "method": method_name,
                    "initialization": records[0]["initialization"],
                    "mean_float_mad": float(np.mean([record["float_metrics"]["mad_vs_gt"] for record in records])),
                    "mean_quantized_mad": float(np.mean([record["quantized_metrics"]["mad_vs_gt"] for record in records])),
                    "mean_final_loss": float(np.mean([record["final_loss"] for record in records])),
                    "mean_initial_loss": float(np.mean([record["initial_loss"] for record in records])),
                    "mean_final_gradient_norm": mean_optional([record["final_gradient_norm"] for record in records]),
                    "mean_update_norm": float(np.mean([record["mean_update_norm"] for record in records])),
                    "mean_serialized_bits": float(np.mean([record["serialized_bits"] for record in records])),
                    "mean_fit_seconds": float(np.mean([record["fit_seconds"] for record in records])),
                }
    return out


def mean_optional(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return float(np.mean(finite))


def markdown_rows(aggregate_metrics: dict[str, Any]) -> str:
    rows = []
    for split in ["development", "evaluation"]:
        for name, values in sorted(aggregate_metrics[split].items()):
            rows.append(
                "| {split} | {name} | {family} | {bits} | {float_mad} | {quant_mad:.6f} | {loss} | {grad} | {fit} |".format(
                    split=split,
                    name=name,
                    family=values["family"],
                    bits="N/A" if "mean_serialized_bits" not in values else f"{values['mean_serialized_bits']:.0f}",
                    float_mad="N/A" if "mean_float_mad" not in values else f"{values['mean_float_mad']:.6f}",
                    quant_mad=values["mean_quantized_mad"],
                    loss="N/A" if "mean_final_loss" not in values else f"{values['mean_final_loss']:.6f}",
                    grad="N/A"
                    if values.get("mean_final_gradient_norm") is None
                    else f"{values['mean_final_gradient_norm']:.6f}",
                    fit="N/A" if "mean_fit_seconds" not in values else f"{values['mean_fit_seconds']:.4f}",
                )
            )
    return "\n".join(rows)


def build_notes(config: dict[str, Any], aggregate_metrics: dict[str, Any]) -> str:
    evaluation = aggregate_metrics["evaluation"]
    model_e_items = {name: values for name, values in evaluation.items() if str(values["family"]).startswith("model_e")}
    classical_items = {
        name: values
        for name, values in evaluation.items()
        if values["family"] not in {"image"} and not str(values["family"]).startswith("model_e")
    }
    best_model_e = min(model_e_items.items(), key=lambda item: item[1]["mean_quantized_mad"])
    best_classical = min(classical_items.items(), key=lambda item: item[1]["mean_quantized_mad"])
    return f"""# Model E Fitting Diagnostics

## Question

Issue #104 の負の結果は、現行 Model E single-state / coupled-state の構造限界なのか、最小random-search fitting protocol の不足なのか。

## Hypothesis

Adam系または L-BFGS 相当の勾配利用optimizerで loss curve、gradient norm、parameter update量が改善するなら、#104 の結果にはfitting protocol不足が含まれる可能性がある。一方、objectiveを下げても evaluation split の quantized MAD が classical INR baseline や bicubic baselineを上回らない場合、optimizer不足だけを理由に現行Model E候補を採用することはできない。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_023_model_e_fitting_diagnostics.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Output size: {config["high_size"]}x{config["high_size"]}
- Low guide size: {config["low_size"]}x{config["low_size"]}
- Fixture manifest: `{config["fixture_manifest"]}`
- Split policy: development/evaluation are separated by source image.
- Model E candidates: model_e_single, model_e_coupled
- Optimizers: random_search, finite-difference Adam, finite-difference L-BFGS-like
- Initialization candidates: default, small_layers
- Parameter quantization: signed uniform {config["parameter_bits"]}-bit values in `[-1, 1]`
- Finite difference epsilon: {config["finite_difference_epsilon"]}
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

画像baselineは nearest、bilinear、bicubic。Parameterized residual baselineは #104 と同じ RFF、SIREN、small MLP を `fit_inr` の最小random-searchでfitした。Model E single/coupled は random-search、Adam系、L-BFGS相当、small_layers初期化のAdamを比較した。

## Result

| Split | Output | Family | Mean serialized side bits | Mean float MAD | Mean quantized MAD | Mean final loss | Mean final grad norm | Mean fit seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{markdown_rows(aggregate_metrics)}

Evaluation splitの最良classical parameterized baselineは `{best_classical[0]}` で、mean quantized MAD `{best_classical[1]["mean_quantized_mad"]:.6f}` だった。最良Model E diagnostic conditionは `{best_model_e[0]}` で、mean quantized MAD `{best_model_e[1]["mean_quantized_mad"]:.6f}` だった。

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Notes: `notes.md`
- Case directories: `development_hobbema_landscape_tl/`, `development_hobbema_landscape_br/`, `evaluation_hokusai_wave_tl/`, `evaluation_hokusai_wave_br/`
- Per-case images: `high_reference.png`, `low_guide.png`, `nearest.png`, `bilinear.png`, `bicubic.png`, `best_model_e_quantized.png`, `comparison.png`, `diff_best_model_e_vs_gt.png`
- Per-case traces: `*_trace.csv`
- Per-case curve images: `model_e_single_loss_curves.png`, `model_e_coupled_loss_curves.png`

## Images

![Development Hobbema TL comparison](development_hobbema_landscape_tl/comparison.png)

![Development Hobbema BR comparison](development_hobbema_landscape_br/comparison.png)

![Evaluation Hokusai TL comparison](evaluation_hokusai_wave_tl/comparison.png)

![Evaluation Hokusai BR comparison](evaluation_hokusai_wave_br/comparison.png)

![Evaluation Hokusai BR single-state loss curves](evaluation_hokusai_wave_br/model_e_single_loss_curves.png)

![Evaluation Hokusai BR coupled-state loss curves](evaluation_hokusai_wave_br/model_e_coupled_loss_curves.png)

## Interpretation

このrunは、現行Model Eのoptimizer診断であり、Model E一般の採否や画像品質の一般結論ではない。finite-difference Adam / L-BFGS-like の結果でlossやgradient normが変化しても、それだけではcompression、super-resolution、quantum advantage、またはSIDF仕様採用の根拠にはならない。

評価では、optimizer不足と構造不足を混同しない。もしModel E条件が #104 のrandom-search条件より改善していても、classical INR baselineやbicubic baselineとの関係、量子化後MAD、serialized side bitsを分けて読む必要がある。

## Limitations

- Development source 1件、evaluation source 1件からの64x64 crop 2件ずつという小規模datasetであり、一般的な画像集合を代表しない。
- Adam / L-BFGS-like は依存追加を避けた有限差分診断であり、本格的なautograd optimizerや厳密なL-BFGS実装ではない。
- 有限差分は計算量を抑えるためforward differenceを使っており、gradient normは診断値である。
- `incremental_side_bits` はparameter side informationの簡易見積もりであり、guide bits、container overhead、entropy codingを含む `total_description_bits` ではない。
- compression、super-resolution、quantum advantageは主張しない。

## Next

- Model E parameterization候補の比較は #122 で扱う。
- より本格的なoptimizer比較を続ける場合は、autograd依存の導入可否と、classical baselineにも同じoptimizerを適用する方針を別Issueで決める。
"""


def main() -> None:
    ensure_dir(RESULT_DIR)
    manifest = load_patch_manifest()
    cases = make_cases()
    case_results = [run_case(case) for case in cases]
    aggregate_metrics = aggregate(case_results)
    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "fixture_manifest": "experiments/assets/source_split_grayscale/manifest.json",
        "fixture_sources": manifest["sources"],
        "high_size": HIGH_SIZE,
        "low_size": LOW_SIZE,
        "random_steps": RANDOM_STEPS,
        "adam_steps": ADAM_STEPS,
        "lbfgs_steps": LBFGS_STEPS,
        "finite_difference_epsilon": FINITE_DIFF_EPSILON,
        "parameter_bits": PARAMETER_BITS,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", {"cases": case_results, "aggregate": aggregate_metrics})
    (RESULT_DIR / "notes.md").write_text(build_notes(config, aggregate_metrics), encoding="utf-8")
    for split, values in aggregate_metrics.items():
        print(split)
        for name, row in sorted(values.items()):
            print(f"  {name}: mad={row['mean_quantized_mad']:.6f}")


if __name__ == "__main__":
    main()
