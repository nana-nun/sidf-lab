"""Compare small nonlinear Model E coupling variants on the source-split fixture."""

from __future__ import annotations

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
from sidf_lab.inr_fit import INRSpec, fit_inr
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
from sidf_lab.model_e import QuantizationSpec
from sidf_lab.patch_fixtures import load_grayscale_patch, load_patch_manifest, list_patch_records


RESULT_DIR = Path("results/2026-07-27-issue-118-model-e-coupling-variants")
DATE = "2026-07-27"
EXPERIMENT_SEED = 20260727
HIGH_SIZE = 64
LOW_SIZE = 16
FIT_STEPS = 48
FIT_STEP_SCALE = 0.035
PARAMETER_BITS = 12
QUANTIZATION = QuantizationSpec(bits_per_value=PARAMETER_BITS, min_value=-1.0, max_value=1.0, header_bits=160)


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


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
        {
            "name": "rff_small",
            "role": "classical_baseline",
            "spec": INRSpec("rff", feature_count=4),
            "seed": 4102,
        },
        {
            "name": "mlp_small",
            "role": "classical_baseline",
            "spec": INRSpec("mlp", feature_count=6),
            "seed": 4104,
        },
        {
            "name": "model_e_single",
            "role": "single_state_baseline",
            "spec": INRSpec("model_e_single", depth=6, states=1),
            "seed": 4105,
        },
        {
            "name": "model_e_coupled_current",
            "role": "current_coupled_baseline",
            "spec": INRSpec("model_e_coupled", depth=3, states=3),
            "seed": 4106,
        },
        {
            "name": "model_e_controlled_rotation",
            "role": "coupling_variant",
            "spec": INRSpec("model_e_controlled_rotation", depth=3, states=3),
            "seed": 4118,
        },
        {
            "name": "model_e_gated_coupled",
            "role": "coupling_variant",
            "spec": INRSpec("model_e_gated_coupled", depth=3, states=3),
            "seed": 4119,
        },
    ]


def stable_case_offset(name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(name))


def image_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
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
    grad = gradient_magnitude(values)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "gradient_magnitude_mean": float(grad.mean()),
        "gradient_magnitude_max": float(grad.max()),
    }


def run_case(case: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, Any]:
    case_dir = ensure_dir(RESULT_DIR / case["name"])
    reference = np.asarray(case["reference"], dtype=np.float64)
    low_guide = downscale_block_average(reference, LOW_SIZE)
    nearest, nearest_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 3)
    baseline_outputs = {"nearest": nearest, "bilinear": bilinear, "bicubic": bicubic}

    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    for name, image in baseline_outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)

    candidate_outputs: dict[str, dict[str, Any]] = {}
    current_coupled_bits: int | None = None
    for spec_record in specs:
        result = fit_inr(
            spec_record["spec"],
            low_guide,
            reference,
            seed=spec_record["seed"] + stable_case_offset(case["name"]),
            steps=FIT_STEPS,
            initial_step_scale=FIT_STEP_SCALE,
            quantization=QUANTIZATION,
        )
        save_grayscale_png(case_dir / f"{spec_record['name']}_quantized.png", result.quantized_decoded)
        if spec_record["name"] == "model_e_coupled_current":
            current_coupled_bits = int(result.bits["incremental_side_bits"])
        candidate_outputs[spec_record["name"]] = {
            "family": result.spec.family,
            "role": spec_record["role"],
            "metrics": image_metrics(result.quantized_decoded, reference),
            "float_metrics": image_metrics(result.float_decoded, reference),
            "fit_seconds": result.fit_seconds,
            "decode_seconds": result.decode_seconds,
            "serialized_bits": result.bits["incremental_side_bits"],
            "bits_per_output_pixel": result.bits["incremental_side_bits"] / float(reference.size),
            "parameter_count": result.bits["parameter_count"],
            "parameter_group_bits": result.bits["parameter_group_bits"],
            "float_to_quantized_mad_delta": image_metrics(result.quantized_decoded, reference)["mad_vs_gt"]
            - image_metrics(result.float_decoded, reference)["mad_vs_gt"],
            "artifact_summary": artifact_summary(result.quantized_decoded),
            "image": result.quantized_decoded,
        }

    if current_coupled_bits is not None:
        for output in candidate_outputs.values():
            if output["role"] == "coupling_variant":
                output["coupling_overhead_bits_vs_current"] = int(output["serialized_bits"] - current_coupled_bits)

    best_current_or_variant = min(
        (
            name
            for name, output in candidate_outputs.items()
            if output["role"] in {"current_coupled_baseline", "coupling_variant"}
        ),
        key=lambda name: candidate_outputs[name]["metrics"]["mad_vs_gt"],
    )
    best_model_e = min(
        (name for name, output in candidate_outputs.items() if str(output["family"]).startswith("model_e")),
        key=lambda name: candidate_outputs[name]["metrics"]["mad_vs_gt"],
    )
    best_classical = min(
        (name for name, output in candidate_outputs.items() if output["role"] == "classical_baseline"),
        key=lambda name: candidate_outputs[name]["metrics"]["mad_vs_gt"],
    )
    save_comparison_png(
        case_dir / "comparison.png",
        [
            reference,
            bicubic,
            candidate_outputs[best_classical]["image"],
            candidate_outputs["model_e_coupled_current"]["image"],
            candidate_outputs[best_current_or_variant]["image"],
            np.abs(candidate_outputs[best_current_or_variant]["image"] - reference),
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
        "candidates": {
            name: {key: value for key, value in output.items() if key != "image"}
            for name, output in candidate_outputs.items()
        },
        "best_classical": best_classical,
        "best_model_e": best_model_e,
        "best_current_or_variant": best_current_or_variant,
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
            metrics = [record["metrics"] for record in records]
            float_metrics = [record["float_metrics"] for record in records]
            overhead_values = [
                record["coupling_overhead_bits_vs_current"]
                for record in records
                if "coupling_overhead_bits_vs_current" in record
            ]
            out[split]["candidates"][name] = {
                "family": records[0]["family"],
                "role": records[0]["role"],
                "mean_mad_vs_gt": mean_metric(metrics, "mad_vs_gt"),
                "mean_psnr_vs_gt": mean_metric(metrics, "psnr_vs_gt"),
                "mean_ssim_global_vs_gt": mean_metric(metrics, "ssim_global_vs_gt"),
                "mean_gradient_magnitude_mad": mean_metric(metrics, "gradient_magnitude_mad"),
                "mean_laplacian_mad": mean_metric(metrics, "laplacian_mad"),
                "mean_float_mad_vs_gt": mean_metric(float_metrics, "mad_vs_gt"),
                "mean_serialized_bits": float(np.mean([record["serialized_bits"] for record in records])),
                "mean_fit_seconds": float(np.mean([record["fit_seconds"] for record in records])),
                "mean_decode_seconds": float(np.mean([record["decode_seconds"] for record in records])),
                "mean_float_to_quantized_mad_delta": float(
                    np.mean([record["float_to_quantized_mad_delta"] for record in records])
                ),
                "mean_coupling_overhead_bits_vs_current": (
                    float(np.mean(overhead_values)) if overhead_values else None
                ),
            }
    return out


def mean_metric(metrics: list[dict[str, float | None]], key: str) -> float | None:
    values = [metric[key] for metric in metrics if metric[key] is not None]
    if not values:
        return None
    return float(np.mean(values))


def markdown_rows(aggregate: dict[str, Any]) -> str:
    rows = []
    for split in ["development", "evaluation"]:
        for name, values in sorted(aggregate[split]["candidates"].items()):
            overhead = values["mean_coupling_overhead_bits_vs_current"]
            rows.append(
                "| {split} | {name} | {role} | {bits:.0f} | {overhead} | {float_mad:.6f} | {mad:.6f} | {grad:.6f} | {decode:.5f} |".format(
                    split=split,
                    name=name,
                    role=values["role"],
                    bits=values["mean_serialized_bits"],
                    overhead="N/A" if overhead is None else f"{overhead:.0f}",
                    float_mad=values["mean_float_mad_vs_gt"],
                    mad=values["mean_mad_vs_gt"],
                    grad=values["mean_gradient_magnitude_mad"],
                    decode=values["mean_decode_seconds"],
                )
            )
    return "\n".join(rows)


def best_candidate(
    aggregate: dict[str, Any],
    split: str,
    roles: set[str],
) -> tuple[str, dict[str, Any]]:
    candidates = aggregate[split]["candidates"]
    eligible = {name: record for name, record in candidates.items() if record["role"] in roles}
    return min(eligible.items(), key=lambda item: item[1]["mean_mad_vs_gt"])


def build_notes(config: dict[str, Any], aggregate: dict[str, Any]) -> str:
    best_classical = best_candidate(aggregate, "evaluation", {"classical_baseline"})
    best_current_or_variant = best_candidate(
        aggregate,
        "evaluation",
        {"current_coupled_baseline", "coupling_variant"},
    )
    best_model_e = best_candidate(
        aggregate,
        "evaluation",
        {"single_state_baseline", "current_coupled_baseline", "coupling_variant"},
    )
    current = aggregate["evaluation"]["candidates"]["model_e_coupled_current"]
    controlled = aggregate["evaluation"]["candidates"]["model_e_controlled_rotation"]
    gated = aggregate["evaluation"]["candidates"]["model_e_gated_coupled"]
    return f"""# Model E Coupling Variants

## Question

Model E の coupled-state 更新を、controlled-rotation風またはgated interaction風に変えると、single-state、現行coupled、classical INR baselineに対して評価splitで改善するか。

## Hypothesis

現行coupled-stateは単純な隣接state相互作用だけを使うため、入力featureから制御されるcouplingを追加すると一部の残差構造を表現しやすくなる可能性がある。一方で、追加parameterによるside-bit overheadが増えるため、改善がなければ採用候補ではなく負の切り分け結果として扱う。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_027_model_e_coupling_variants.py`
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
- Parameter quantization: signed uniform {config["parameter_bits"]}-bit values in `[-1, 1]`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

画像baselineは nearest、bilinear、bicubic。Parameterized residual baselineは `rff_small` と `mlp_small`。Model E側は `model_e_single`、現行 `model_e_coupled_current`、新候補 `model_e_controlled_rotation`、`model_e_gated_coupled` を同じ `fit_inr` interface、同じstep数、同じ12-bit量子化で比較した。

## Result

| Split | Candidate | Role | Mean side bits | Coupling overhead bits | Mean float MAD | Mean quantized MAD | Mean gradient MAD | Mean decode seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{markdown_rows(aggregate)}

Evaluation splitで最良classical baselineは `{best_classical[0]}` の mean quantized MAD `{best_classical[1]["mean_mad_vs_gt"]:.6f}`、最良Model E系候補は `{best_model_e[0]}` の `{best_model_e[1]["mean_mad_vs_gt"]:.6f}` だった。

現行coupledと新coupling候補の比較では、現行coupled `model_e_coupled_current` が `{current["mean_mad_vs_gt"]:.6f}`、controlled rotation が `{controlled["mean_mad_vs_gt"]:.6f}`、gated coupled が `{gated["mean_mad_vs_gt"]:.6f}` だった。新候補のcoupling overheadは現行coupled比で平均 `{controlled["mean_coupling_overhead_bits_vs_current"]:.0f}` bits だった。

## Saved Artifacts

- `config.json`
- `metrics.json`
- `notes.md`
- Per-case directories with `high_reference.png`, `low_guide.png`, nearest/bilinear/bicubic baselines, `*_quantized.png`, and `comparison.png`

## Images

![Development Hobbema TL comparison](development_hobbema_landscape_tl/comparison.png)

![Development Hobbema BR comparison](development_hobbema_landscape_br/comparison.png)

![Evaluation Hokusai TL comparison](evaluation_hokusai_wave_tl/comparison.png)

![Evaluation Hokusai BR comparison](evaluation_hokusai_wave_br/comparison.png)

## Interpretation

このrunでは、`model_e_gated_coupled` は現行coupled-stateよりevaluation splitのmean quantized MADを改善した。一方で、best classical INRの `mlp_small` には届かず、現行coupledに対して平均540 bitsのcoupling overheadも増えた。したがって、今回のfixtureとrandom-search fit条件では、gated interaction候補を採用候補へ戻す根拠としては不足している。`model_e_controlled_rotation` は現行coupledを改善しなかった。

## Limitations

- Development source 1件、evaluation source 1件からの64x64 crop 2件ずつという小規模datasetである。
- dependency-free random searchであり、各candidateの到達可能品質を保証しない。
- `incremental_side_bits` はparameter side informationだけで、guide bits、container overhead、entropy codingを含まない。
- 新coupling候補は2案だけであり、coupling設計一般の否定ではない。
- 実用圧縮、super-resolution、量子優位は主張しない。

## Next

- #118 の結果を Model E 系列の継続/保留判断へ反映する。
- Model Eをさらに続ける場合は、新しいcoupling式の追加より先に optional autograd optimizer の同条件実測、またはModel E系列の一時保留判断を分けて扱う。
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
        "issue": 118,
        "experiment_seed": EXPERIMENT_SEED,
        "fixture_manifest": "experiments/assets/source_split_grayscale/manifest.json",
        "fixture_sources": manifest["sources"],
        "high_size": HIGH_SIZE,
        "low_size": LOW_SIZE,
        "fit_steps": FIT_STEPS,
        "fit_step_scale": FIT_STEP_SCALE,
        "parameter_bits": PARAMETER_BITS,
        "candidate_specs": [
            {
                "name": record["name"],
                "role": record["role"],
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
    (RESULT_DIR / "notes.md").write_text(build_notes(config, aggregate), encoding="utf-8")

    for split in ["development", "evaluation"]:
        print(split)
        for name, values in sorted(aggregate[split]["candidates"].items()):
            print(
                f"  {name}: bits={values['mean_serialized_bits']:.0f} "
                f"mad={values['mean_mad_vs_gt']:.6f}"
            )


if __name__ == "__main__":
    main()
