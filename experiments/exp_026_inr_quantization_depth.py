"""Compare INR parameter quantization depth on the #104 source-split fixture."""

from __future__ import annotations

import csv
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
    quantize_vector,
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


RESULT_DIR = Path("results/2026-07-05-issue-119-inr-quantization-depth")
DATE = "2026-07-05"
EXPERIMENT_SEED = 20260705
HIGH_SIZE = 64
LOW_SIZE = 16
FIT_STEPS = 48
FIT_STEP_SCALE = 0.035
BIT_DEPTHS = (8, 12, 16)
QUANTIZATION_MIN = -1.0
QUANTIZATION_MAX = 1.0
HEADER_BITS = 160


def make_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    crop_specs = [
        ("tl", 0, 0),
        ("br", 64, 64),
    ]
    for record in list_patch_records():
        source = load_grayscale_patch(record["name"])
        for suffix, row, col in crop_specs:
            reference = source[row : row + HIGH_SIZE, col : col + HIGH_SIZE]
            cases.append(
                {
                    "name": f"{record['split']}_{record['source_id']}_{suffix}",
                    "split": record["split"],
                    "source_id": record["source_id"],
                    "patch_name": record["name"],
                    "crop": {"row": row, "col": col, "height": HIGH_SIZE, "width": HIGH_SIZE},
                    "reference": reference,
                }
            )
    return cases


def make_specs() -> list[dict[str, Any]]:
    return [
        {"name": "rff_small", "spec": INRSpec("rff", feature_count=4), "seed": 4102},
        {"name": "siren_small", "spec": INRSpec("siren", feature_count=6), "seed": 4103},
        {"name": "mlp_small", "spec": INRSpec("mlp", feature_count=6), "seed": 4104},
        {"name": "model_e_single", "spec": INRSpec("model_e_single", depth=6, states=1), "seed": 4105},
        {"name": "model_e_coupled", "spec": INRSpec("model_e_coupled", depth=3, states=3), "seed": 4106},
    ]


def stable_case_offset(name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(name))


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def image_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    return {
        "mad_vs_gt": mad(candidate, reference),
        "psnr_vs_gt": psnr(reference, candidate),
        "ssim_global_vs_gt": ssim_global(reference, candidate),
        "gradient_magnitude_mad": gradient_magnitude_mad(reference, candidate),
        "gradient_magnitude_correlation": gradient_magnitude_correlation(reference, candidate),
        "laplacian_mad": laplacian_mad(reference, candidate),
    }


def quantization(bits: int) -> QuantizationSpec:
    return QuantizationSpec(
        bits_per_value=bits,
        min_value=QUANTIZATION_MIN,
        max_value=QUANTIZATION_MAX,
        header_bits=HEADER_BITS,
    )


def quantized_decode(
    spec: INRSpec,
    low_guide: np.ndarray,
    reference_shape: tuple[int, int],
    parameters: np.ndarray,
    bits: int,
) -> tuple[np.ndarray, dict[str, object]]:
    quant = quantization(bits)
    quantized = quantize_vector(parameters, quant)
    restored = dequantize_vector(quantized, quant)
    decoded = decode_inr(spec, low_guide, reference_shape, restored)
    return decoded, estimate_inr_bits(spec, quant)


def run_case(case: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, Any]:
    case_dir = ensure_dir(RESULT_DIR / case["name"])
    reference = np.asarray(case["reference"], dtype=np.float64)
    low_guide = downscale_block_average(reference, LOW_SIZE)
    nearest, nearest_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 3)

    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    save_grayscale_png(case_dir / "nearest.png", nearest)
    save_grayscale_png(case_dir / "bilinear.png", bilinear)
    save_grayscale_png(case_dir / "bicubic.png", bicubic)

    candidate_outputs: dict[str, Any] = {}
    comparison_images: dict[tuple[str, int], np.ndarray] = {}
    for spec_record in specs:
        fit = fit_inr(
            spec_record["spec"],
            low_guide,
            reference,
            seed=spec_record["seed"] + stable_case_offset(case["name"]),
            steps=FIT_STEPS,
            initial_step_scale=FIT_STEP_SCALE,
            quantization=quantization(12),
        )
        bit_outputs: dict[str, Any] = {}
        for bits in BIT_DEPTHS:
            decode_start = time.perf_counter()
            decoded, bit_summary = quantized_decode(
                spec_record["spec"],
                low_guide,
                reference.shape,
                fit.parameters,
                bits,
            )
            decode_seconds = float(time.perf_counter() - decode_start)
            save_grayscale_png(case_dir / f"{spec_record['name']}_{bits}bit.png", decoded)
            comparison_images[(spec_record["name"], bits)] = decoded
            bit_outputs[str(bits)] = {
                "metrics": image_metrics(decoded, reference),
                "serialized_bits": bit_summary["incremental_side_bits"],
                "bits_per_output_pixel": bit_summary["incremental_side_bits"] / float(reference.size),
                "decode_seconds": decode_seconds,
                "float_to_quantized_mad_delta": image_metrics(decoded, reference)["mad_vs_gt"]
                - image_metrics(fit.float_decoded, reference)["mad_vs_gt"],
            }
        candidate_outputs[spec_record["name"]] = {
            "family": fit.spec.family,
            "float_metrics": image_metrics(fit.float_decoded, reference),
            "fit_seconds": fit.fit_seconds,
            "parameter_count": fit.bits["parameter_count"],
            "bit_depths": bit_outputs,
        }

    eval_candidates = candidate_outputs
    best_classical = min(
        (name for name, result in eval_candidates.items() if not str(result["family"]).startswith("model_e")),
        key=lambda name: eval_candidates[name]["bit_depths"]["8"]["metrics"]["mad_vs_gt"],
    )
    best_model_e = min(
        (name for name, result in eval_candidates.items() if str(result["family"]).startswith("model_e")),
        key=lambda name: eval_candidates[name]["bit_depths"]["8"]["metrics"]["mad_vs_gt"],
    )
    save_comparison_png(
        case_dir / "comparison_8bit.png",
        [
            reference,
            bicubic,
            comparison_images[(best_classical, 8)],
            comparison_images[(best_model_e, 8)],
            np.abs(comparison_images[(best_classical, 8)] - reference),
            np.abs(comparison_images[(best_model_e, 8)] - reference),
        ],
    )

    return {
        "name": case["name"],
        "split": case["split"],
        "source_id": case["source_id"],
        "patch_name": case["patch_name"],
        "crop": case["crop"],
        "baselines": {
            "nearest": {"metrics": image_metrics(nearest, reference), "decode_seconds": nearest_seconds},
            "bilinear": {"metrics": image_metrics(bilinear, reference), "decode_seconds": bilinear_seconds},
            "bicubic": {"metrics": image_metrics(bicubic, reference), "decode_seconds": bicubic_seconds},
        },
        "candidates": candidate_outputs,
        "comparison_8bit": {
            "best_classical": best_classical,
            "best_model_e": best_model_e,
        },
    }


def aggregate_by_split(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["development", "evaluation"]:
        selected = [case for case in case_results if case["split"] == split]
        out[split] = {"baselines": {}, "candidates": {}}
        if not selected:
            continue
        for baseline in ["nearest", "bilinear", "bicubic"]:
            metrics = [case["baselines"][baseline]["metrics"] for case in selected]
            out[split]["baselines"][baseline] = {
                "mean_mad_vs_gt": mean_metric(metrics, "mad_vs_gt"),
                "mean_psnr_vs_gt": mean_metric(metrics, "psnr_vs_gt"),
                "mean_ssim_global_vs_gt": mean_metric(metrics, "ssim_global_vs_gt"),
            }
        for name in selected[0]["candidates"]:
            records = [case["candidates"][name] for case in selected]
            bit_depths: dict[str, Any] = {}
            for bits in BIT_DEPTHS:
                key = str(bits)
                bit_records = [record["bit_depths"][key] for record in records]
                metrics = [record["metrics"] for record in bit_records]
                bit_depths[key] = {
                    "mean_serialized_bits": float(np.mean([record["serialized_bits"] for record in bit_records])),
                    "mean_bits_per_output_pixel": float(
                        np.mean([record["bits_per_output_pixel"] for record in bit_records])
                    ),
                    "mean_quantized_mad_vs_gt": mean_metric(metrics, "mad_vs_gt"),
                    "mean_quantized_psnr_vs_gt": mean_metric(metrics, "psnr_vs_gt"),
                    "mean_quantized_ssim_global_vs_gt": mean_metric(metrics, "ssim_global_vs_gt"),
                    "mean_float_to_quantized_mad_delta": float(
                        np.mean([record["float_to_quantized_mad_delta"] for record in bit_records])
                    ),
                    "mean_decode_seconds": float(np.mean([record["decode_seconds"] for record in bit_records])),
                }
            float_metrics = [record["float_metrics"] for record in records]
            out[split]["candidates"][name] = {
                "family": records[0]["family"],
                "parameter_count": records[0]["parameter_count"],
                "mean_float_mad_vs_gt": mean_metric(float_metrics, "mad_vs_gt"),
                "mean_float_psnr_vs_gt": mean_metric(float_metrics, "psnr_vs_gt"),
                "mean_fit_seconds": float(np.mean([record["fit_seconds"] for record in records])),
                "bit_depths": bit_depths,
            }
    return out


def mean_metric(metrics: list[dict[str, float | None]], key: str) -> float | None:
    values = [metric[key] for metric in metrics if metric[key] is not None]
    if not values:
        return None
    return float(np.mean(values))


def write_rate_distortion_csv(path: Path, aggregate: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "candidate",
                "family",
                "bit_depth",
                "mean_serialized_bits",
                "mean_bits_per_output_pixel",
                "mean_float_mad_vs_gt",
                "mean_quantized_mad_vs_gt",
                "mean_float_to_quantized_mad_delta",
                "mean_quantized_psnr_vs_gt",
            ],
        )
        writer.writeheader()
        for split in ["development", "evaluation"]:
            for name, record in sorted(aggregate[split]["candidates"].items()):
                for bits in BIT_DEPTHS:
                    bit_record = record["bit_depths"][str(bits)]
                    writer.writerow(
                        {
                            "split": split,
                            "candidate": name,
                            "family": record["family"],
                            "bit_depth": bits,
                            "mean_serialized_bits": bit_record["mean_serialized_bits"],
                            "mean_bits_per_output_pixel": bit_record["mean_bits_per_output_pixel"],
                            "mean_float_mad_vs_gt": record["mean_float_mad_vs_gt"],
                            "mean_quantized_mad_vs_gt": bit_record["mean_quantized_mad_vs_gt"],
                            "mean_float_to_quantized_mad_delta": bit_record[
                                "mean_float_to_quantized_mad_delta"
                            ],
                            "mean_quantized_psnr_vs_gt": bit_record["mean_quantized_psnr_vs_gt"],
                        }
                    )


def markdown_rate_rows(aggregate: dict[str, Any]) -> str:
    rows = []
    for split in ["development", "evaluation"]:
        for name, record in sorted(aggregate[split]["candidates"].items()):
            for bits in BIT_DEPTHS:
                bit_record = record["bit_depths"][str(bits)]
                rows.append(
                    "| {split} | {name} | {family} | {bits} | {side:.0f} | {float_mad:.6f} | {quant_mad:.6f} | {delta:+.6f} | {psnr:.3f} |".format(
                        split=split,
                        name=name,
                        family=record["family"],
                        bits=bits,
                        side=bit_record["mean_serialized_bits"],
                        float_mad=record["mean_float_mad_vs_gt"],
                        quant_mad=bit_record["mean_quantized_mad_vs_gt"],
                        delta=bit_record["mean_float_to_quantized_mad_delta"],
                        psnr=bit_record["mean_quantized_psnr_vs_gt"],
                    )
                )
    return "\n".join(rows)


def best_by_family(aggregate: dict[str, Any], split: str, bits: int, model_e: bool) -> tuple[str, dict[str, Any]]:
    candidates = aggregate[split]["candidates"]
    eligible = {
        name: record
        for name, record in candidates.items()
        if str(record["family"]).startswith("model_e") == model_e
    }
    return min(
        eligible.items(),
        key=lambda item: item[1]["bit_depths"][str(bits)]["mean_quantized_mad_vs_gt"],
    )


def build_notes(config: dict[str, Any], aggregate: dict[str, Any]) -> str:
    comparisons = []
    for bits in BIT_DEPTHS:
        classical = best_by_family(aggregate, "evaluation", bits, model_e=False)
        model_e = best_by_family(aggregate, "evaluation", bits, model_e=True)
        comparisons.append(
            "- {bits}-bit: best classical `{classical}` MAD {classical_mad:.6f}; best Model E `{model_e}` MAD {model_e_mad:.6f}.".format(
                bits=bits,
                classical=classical[0],
                classical_mad=classical[1]["bit_depths"][str(bits)]["mean_quantized_mad_vs_gt"],
                model_e=model_e[0],
                model_e_mad=model_e[1]["bit_depths"][str(bits)]["mean_quantized_mad_vs_gt"],
            )
        )
    return f"""# INR Parameter Quantization Depth Comparison

## Question

Model E single/coupled と classical INR baseline は、8-bit、12-bit、16-bit のparameter量子化でどちらが劣化しにくいか。

## Hypothesis

Model Eの回転角・状態更新に由来するparameterizationは、低bit量子化でも品質低下が小さい可能性がある。ただし #104 の結果では現行Model E候補は classical INR baselineを上回っていないため、この実験は採用判断ではなく量子化耐性の切り分けとして扱う。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_026_inr_quantization_depth.py`
- Date: {config["date"]}
- Issue: #{config["issue"]}
- Experiment seed: {config["experiment_seed"]}
- Output size: {config["high_size"]}x{config["high_size"]}
- Low guide size: {config["low_size"]}x{config["low_size"]}
- Low guide method: block average from 64x64 reference crop
- Fixture manifest: `{config["fixture_manifest"]}`
- Split policy: development/evaluation are separated by source image.
- Fit steps: {config["fit_steps"]}
- Initial step scale: {config["fit_step_scale"]}
- Parameter quantization depths: {", ".join(str(bits) for bits in config["bit_depths"])} bits in `[-1, 1]`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

画像baselineは nearest、bilinear、bicubic。Parameterized residual candidates は RFF、SIREN、small MLP、Model E single-state、Model E coupled-state。各candidateは一度float parameterをfitし、その同じparameterを8/12/16-bitに再量子化した。

## Metrics

- MAD、PSNR、global SSIM、gradient magnitude MAD、gradient magnitude correlation、Laplacian MAD
- incremental side bits と bits per output pixel
- float-to-quantized MAD delta
- fit time、decode time

## Result

| Split | Candidate | Family | Bit depth | Mean side bits | Mean float MAD | Mean quantized MAD | Float-to-quantized MAD delta | Mean PSNR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{markdown_rate_rows(aggregate)}

Evaluation split summary:

{chr(10).join(comparisons)}

## Saved Artifacts

- `config.json`
- `metrics.json`
- `rate_distortion.csv`
- `notes.md`
- Per-case directories with `high_reference.png`, `low_guide.png`, nearest/bilinear/bicubic baselines, `*_8bit.png`, `*_12bit.png`, `*_16bit.png`, and `comparison_8bit.png`

## Images

![Development Hobbema TL 8-bit comparison](development_hobbema_landscape_tl/comparison_8bit.png)

![Development Hobbema BR 8-bit comparison](development_hobbema_landscape_br/comparison_8bit.png)

![Evaluation Hokusai TL 8-bit comparison](evaluation_hokusai_wave_tl/comparison_8bit.png)

![Evaluation Hokusai BR 8-bit comparison](evaluation_hokusai_wave_br/comparison_8bit.png)

## Interpretation

このrunでは、8/12/16-bitのいずれでも best Model E 候補が best classical INR 候補をevaluation splitのmean quantized MADで上回ったとは解釈しない。Model E single は8-bitでfloat-to-quantized deltaが小さいが、float時点のMADがclassical候補より高く、低bit耐性だけで採用根拠にはならない。

## Limitations

- Development source 1件、evaluation source 1件からの64x64 crop 2件ずつという小規模datasetである。
- `incremental_side_bits` はparameter side informationだけで、guide bits、container overhead、entropy codingを含まない。
- 同じrandom-search fit後の再量子化比較であり、bit depthごとに再fitしていない。
- 8-bit PNG保存画像は可視化artifactであり、metricsはNumPy配列上で計算した。
- 実用圧縮、super-resolution、量子優位は主張しない。

## Next

- Model E系列の継続/保留判断では、bit-depth耐性だけでなく #98 / #104 / #117 / #122 / #119 の負の結果をまとめて扱う。
- 追加実験をする場合は、bit-depthごとの再fitまたはoptional autograd backendでの同条件比較を別Issueに分ける。
"""


def main() -> None:
    ensure_dir(RESULT_DIR)
    manifest = load_patch_manifest()
    specs = make_specs()
    cases = make_cases()
    case_results = [run_case(case, specs) for case in cases]
    aggregate = aggregate_by_split(case_results)
    config = {
        "date": DATE,
        "issue": 119,
        "experiment_seed": EXPERIMENT_SEED,
        "fixture_manifest": "experiments/assets/source_split_grayscale/manifest.json",
        "fixture_sources": manifest["sources"],
        "high_size": HIGH_SIZE,
        "low_size": LOW_SIZE,
        "fit_steps": FIT_STEPS,
        "fit_step_scale": FIT_STEP_SCALE,
        "bit_depths": list(BIT_DEPTHS),
        "quantization_min": QUANTIZATION_MIN,
        "quantization_max": QUANTIZATION_MAX,
        "candidate_specs": [
            {
                "name": record["name"],
                "family": record["spec"].family,
                "order": record["spec"].order,
                "feature_count": record["spec"].feature_count,
                "depth": record["spec"].depth,
                "states": record["spec"].states,
                "seed": record["seed"],
            }
            for record in specs
        ],
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", {"cases": case_results, "aggregate": aggregate})
    write_rate_distortion_csv(RESULT_DIR / "rate_distortion.csv", aggregate)
    (RESULT_DIR / "notes.md").write_text(build_notes(config, aggregate), encoding="utf-8")

    for bits in BIT_DEPTHS:
        classical = best_by_family(aggregate, "evaluation", bits, model_e=False)
        model_e = best_by_family(aggregate, "evaluation", bits, model_e=True)
        print(
            f"{bits}-bit evaluation: best_classical={classical[0]} "
            f"mad={classical[1]['bit_depths'][str(bits)]['mean_quantized_mad_vs_gt']:.6f}; "
            f"best_model_e={model_e[0]} "
            f"mad={model_e[1]['bit_depths'][str(bits)]['mean_quantized_mad_vs_gt']:.6f}"
        )


if __name__ == "__main__":
    main()
