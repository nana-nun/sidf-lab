"""Compare trainable Model E and classical INR baselines on source-split patches."""

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
from sidf_lab.inr_fit import INRSpec, decode_inr, fit_inr
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


RESULT_DIR = Path("results/2026-06-28-issue-104-trainable-inr-source-split")
DATE = "2026-06-28"
EXPERIMENT_SEED = 20260628
HIGH_SIZE = 64
LOW_SIZE = 16
FIT_STEPS = 48
FIT_STEP_SCALE = 0.035
PARAMETER_BITS = 12
QUANTIZATION = QuantizationSpec(bits_per_value=PARAMETER_BITS, min_value=-1.0, max_value=1.0, header_bits=160)
EXTRAPOLATE_SIZE = 128


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
        {"name": "fourier_mid", "spec": INRSpec("fourier", order=2), "seed": 4101},
        {"name": "rff_small", "spec": INRSpec("rff", feature_count=4), "seed": 4102},
        {"name": "siren_small", "spec": INRSpec("siren", feature_count=6), "seed": 4103},
        {"name": "mlp_small", "spec": INRSpec("mlp", feature_count=6), "seed": 4104},
        {"name": "model_e_single", "spec": INRSpec("model_e_single", depth=6, states=1), "seed": 4105},
        {"name": "model_e_coupled", "spec": INRSpec("model_e_coupled", depth=3, states=3), "seed": 4106},
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


def artifact_summary(image: np.ndarray) -> dict[str, float]:
    values = np.asarray(image, dtype=np.float64)
    grad = gradient_magnitude(values)
    lap = (
        np.pad(values, 1, mode="edge")[:-2, 1:-1]
        + np.pad(values, 1, mode="edge")[2:, 1:-1]
        + np.pad(values, 1, mode="edge")[1:-1, :-2]
        + np.pad(values, 1, mode="edge")[1:-1, 2:]
        - 4.0 * values
    )
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "gradient_magnitude_mean": float(grad.mean()),
        "gradient_magnitude_max": float(grad.max()),
        "laplacian_abs_mean": float(np.mean(np.abs(lap))),
    }


def run_case(case: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, Any]:
    case_dir = ensure_dir(RESULT_DIR / case["name"])
    reference = np.asarray(case["reference"], dtype=np.float64)
    low_guide = downscale_block_average(reference, LOW_SIZE)
    nearest, nearest_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, HIGH_SIZE, 3)
    baseline_outputs = {"nearest": nearest, "bilinear": bilinear, "bicubic": bicubic}

    candidate_outputs: dict[str, dict[str, Any]] = {}
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
        candidate_outputs[spec_record["name"]] = {
            "family": result.spec.family,
            "metrics": image_metrics(result.quantized_decoded, reference),
            "float_metrics": image_metrics(result.float_decoded, reference),
            "fit_seconds": result.fit_seconds,
            "decode_seconds": result.decode_seconds,
            "serialized_bits": result.bits["incremental_side_bits"],
            "bits_per_output_pixel": result.bits["incremental_side_bits"] / float(reference.size),
            "parameter_count": result.bits["parameter_count"],
            "float_to_quantized_mad_delta": image_metrics(result.quantized_decoded, reference)["mad_vs_gt"]
            - image_metrics(result.float_decoded, reference)["mad_vs_gt"],
            "parameters": result.restored_parameters,
            "image": result.quantized_decoded,
            "float_image": result.float_decoded,
        }

    best_name = min(candidate_outputs, key=lambda name: candidate_outputs[name]["metrics"]["mad_vs_gt"])
    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    for name, image in baseline_outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
    save_grayscale_png(case_dir / f"{best_name}_float.png", candidate_outputs[best_name]["float_image"])
    save_grayscale_png(case_dir / f"diff_{best_name}_vs_gt.png", np.abs(candidate_outputs[best_name]["image"] - reference))
    save_grayscale_png(case_dir / "diff_bicubic_vs_gt.png", np.abs(bicubic - reference))
    save_comparison_png(
        case_dir / "comparison.png",
        [
            reference,
            nearest,
            bilinear,
            bicubic,
            candidate_outputs[best_name]["image"],
            np.abs(candidate_outputs[best_name]["image"] - reference),
        ],
    )

    extrapolated: dict[str, Any] = {}
    if case["split"] == "evaluation" and case["name"].endswith("_br"):
        bicubic_extra = upscale(low_guide, EXTRAPOLATE_SIZE, 3)
        save_grayscale_png(case_dir / "extrapolated_bicubic.png", bicubic_extra)
        extrapolated["bicubic"] = artifact_summary(bicubic_extra)
        for name in ["rff_small", "siren_small", "mlp_small", "model_e_coupled", best_name]:
            if name not in candidate_outputs:
                continue
            spec = next(record["spec"] for record in specs if record["name"] == name)
            image = decode_inr(spec, low_guide, (EXTRAPOLATE_SIZE, EXTRAPOLATE_SIZE), candidate_outputs[name]["parameters"])
            save_grayscale_png(case_dir / f"extrapolated_{name}.png", image)
            extrapolated[name] = artifact_summary(image)
        save_comparison_png(
            case_dir / "extrapolated_comparison.png",
            [bicubic_extra]
            + [
                decode_inr(
                    next(record["spec"] for record in specs if record["name"] == name),
                    low_guide,
                    (EXTRAPOLATE_SIZE, EXTRAPOLATE_SIZE),
                    candidate_outputs[name]["parameters"],
                )
                for name in ["rff_small", "siren_small", "mlp_small", "model_e_coupled"]
            ],
        )

    return {
        "name": case["name"],
        "split": case["split"],
        "source_id": case["source_id"],
        "patch_name": case["patch_name"],
        "crop": case["crop"],
        "baselines": {
            name: {
                "metrics": image_metrics(image, reference),
                "decode_seconds": {"nearest": nearest_seconds, "bilinear": bilinear_seconds, "bicubic": bicubic_seconds}[name],
            }
            for name, image in baseline_outputs.items()
        },
        "candidates": {
            name: {key: value for key, value in result.items() if key not in {"image", "float_image", "parameters"}}
            for name, result in candidate_outputs.items()
        },
        "best_candidate": best_name,
        "extrapolated": extrapolated,
    }


def stable_case_offset(name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(name))


def aggregate_by_split(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["development", "evaluation"]:
        selected = [case for case in case_results if case["split"] == split]
        out[split] = {}
        if not selected:
            continue
        for baseline in ["nearest", "bilinear", "bicubic"]:
            values = [case["baselines"][baseline]["metrics"]["mad_vs_gt"] for case in selected]
            out[split][baseline] = {"mean_mad_vs_gt": float(np.mean(values))}
        for name in selected[0]["candidates"]:
            mad_values = [case["candidates"][name]["metrics"]["mad_vs_gt"] for case in selected]
            float_mad = [case["candidates"][name]["float_metrics"]["mad_vs_gt"] for case in selected]
            bits = [case["candidates"][name]["serialized_bits"] for case in selected]
            fit_seconds = [case["candidates"][name]["fit_seconds"] for case in selected]
            decode_seconds = [case["candidates"][name]["decode_seconds"] for case in selected]
            out[split][name] = {
                "family": selected[0]["candidates"][name]["family"],
                "mean_mad_vs_gt": float(np.mean(mad_values)),
                "mean_float_mad_vs_gt": float(np.mean(float_mad)),
                "mean_serialized_bits": float(np.mean(bits)),
                "mean_fit_seconds": float(np.mean(fit_seconds)),
                "mean_decode_seconds": float(np.mean(decode_seconds)),
            }
    return out


def markdown_rows(aggregate: dict[str, Any]) -> str:
    rows = []
    for split in ["development", "evaluation"]:
        for name, values in sorted(aggregate[split].items()):
            bits = values.get("mean_serialized_bits")
            float_mad = values.get("mean_float_mad_vs_gt")
            rows.append(
                "| {split} | {name} | {family} | {bits} | {float_mad} | {mad:.6f} | {fit} |".format(
                    split=split,
                    name=name,
                    family=values.get("family", "image"),
                    bits="N/A" if bits is None else f"{bits:.0f}",
                    float_mad="N/A" if float_mad is None else f"{float_mad:.6f}",
                    mad=values["mean_mad_vs_gt"],
                    fit="N/A" if "mean_fit_seconds" not in values else f"{values['mean_fit_seconds']:.4f}",
                )
            )
    return "\n".join(rows)


def extrapolated_rows(case_results: list[dict[str, Any]]) -> str:
    rows = []
    for case in case_results:
        for name, values in sorted(case["extrapolated"].items()):
            rows.append(
                "| {case} | {name} | {grad:.6f} | {gmax:.6f} | {lap:.6f} | {minv:.6f} | {maxv:.6f} |".format(
                    case=case["name"],
                    name=name,
                    grad=values["gradient_magnitude_mean"],
                    gmax=values["gradient_magnitude_max"],
                    lap=values["laplacian_abs_mean"],
                    minv=values["min"],
                    maxv=values["max"],
                )
            )
    return "\n".join(rows)


def build_notes(config: dict[str, Any], case_results: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    eval_parameterized = {
        name: values
        for name, values in aggregate["evaluation"].items()
        if values.get("family") not in {None, "image"}
    }
    best_eval = min(eval_parameterized.items(), key=lambda item: item[1]["mean_mad_vs_gt"])
    best_model_e = min(
        ((name, values) for name, values in eval_parameterized.items() if str(values["family"]).startswith("model_e")),
        key=lambda item: item[1]["mean_mad_vs_gt"],
    )
    best_classical = min(
        ((name, values) for name, values in eval_parameterized.items() if not str(values["family"]).startswith("model_e")),
        key=lambda item: item[1]["mean_mad_vs_gt"],
    )
    return f"""# Model E Trainable INR Source-Split Comparison

## Question

Issue #98 の fixed-feature 制限を外し、source image単位で development / evaluation を分けた小規模datasetで、Model E single/coupled が trainable classical INR baseline より低い量子化後MADを示すか。

## Hypothesis

全parameter fittingにより Model E の固定feature制限は緩和される可能性がある。一方で、同じ serialized side-bit accounting のもとで RFF / SIREN / small MLP が同等または優位なら、現行Model E候補は再設計または不採用候補として扱う。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_022_model_e_trainable_inr_source_split.py`
- Date: {config["date"]}
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
- Base dependency: #103 fitting helper. This PR is stacked on #103 until #112 is merged.

## Baseline

画像baselineは nearest、bilinear、bicubic。Parameterized residual candidates は Fourier、RFF、SIREN、small MLP、Model E single-state、Model E coupled-state。全candidateは同じ `fit_inr` interface、同じstep数、同じ量子化規則でfitした。

## Result

| Split | Output | Family | Mean serialized side bits | Mean float MAD | Mean quantized MAD | Mean fit seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
{markdown_rows(aggregate)}

Evaluation splitで最小MADのparameterized候補は `{best_eval[0]}` で、mean quantized MAD `{best_eval[1]["mean_mad_vs_gt"]:.6f}`、mean serialized side bits `{best_eval[1]["mean_serialized_bits"]:.0f}` だった。

最良classical INR候補は `{best_classical[0]}` の mean quantized MAD `{best_classical[1]["mean_mad_vs_gt"]:.6f}`、最良Model E候補は `{best_model_e[0]}` の mean quantized MAD `{best_model_e[1]["mean_mad_vs_gt"]:.6f}` だった。

### Extrapolated Output Diagnostic

Evaluation sourceのbottom-right cropで、fit済みparameterを128x128座標へ評価した。これは128x128 Ground Truth品質の測定ではなく、periodic artifactやaliasing傾向を見る診断である。

| Case | Output | Gradient magnitude mean | Gradient magnitude max | Laplacian abs mean | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{extrapolated_rows(case_results)}

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Notes: `notes.md`
- Case directories: `development_hobbema_landscape_tl/`, `development_hobbema_landscape_br/`, `evaluation_hokusai_wave_tl/`, `evaluation_hokusai_wave_br/`
- Per-case images: `high_reference.png`, `low_guide.png`, `nearest.png`, `bilinear.png`, `bicubic.png`, `*_quantized.png`, `comparison.png`, difference maps
- Extrapolated diagnostic: `evaluation_hokusai_wave_br/extrapolated_comparison.png`

## Images

![Development Hobbema TL comparison](development_hobbema_landscape_tl/comparison.png)

![Development Hobbema BR comparison](development_hobbema_landscape_br/comparison.png)

![Evaluation Hokusai TL comparison](evaluation_hokusai_wave_tl/comparison.png)

![Evaluation Hokusai BR comparison](evaluation_hokusai_wave_br/comparison.png)

![Evaluation extrapolated outputs](evaluation_hokusai_wave_br/extrapolated_comparison.png)

## Interpretation

このrunでは、最良Model E候補が最良classical INR候補をevaluation splitで上回るとは解釈しない。結果は small source-split fixture と最小random-search optimizerに限定されるが、fixed-feature制限を外しても現行Model E候補を採用する根拠は得られなかった。

これは量子インスパイアード表現一般の否定ではなく、今回の Model E parameterization、optimizer、dataset、bit accounting 条件での負の結果である。

## Limitations

- Development source 1件、evaluation source 1件からの64x64 crop 2件ずつという小規模datasetであり、一般的な画像集合を代表しない。
- #103のoptimizerはdependency-freeな最小random-searchであり、SIREN/MLP/Model Eの到達可能品質を保証しない。
- `incremental_side_bits` はparameter side informationの簡易見積もりであり、guide bits、container overhead、entropy codingを含む `total_description_bits` ではない。
- extrapolated outputはartifact診断であり、Ground Truth比較ではない。
- compression、super-resolution、quantum advantageは主張しない。

## Next

- 現行Model E候補は、今回の条件では採用しない候補として `docs/model-decision-map.md` または関連docsへ反映する。
- Model Eを再設計する場合は、random-search改善だけでなく、angle/frequency parameterizationそのものを見直す。
- Classical INR baselineについては、より適切なoptimizerを入れる場合も、同じsource-split fixtureとbit accountingで再比較する。
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
    (RESULT_DIR / "notes.md").write_text(build_notes(config, case_results, aggregate), encoding="utf-8")

    for split, outputs in aggregate.items():
        print(split)
        for name, values in sorted(outputs.items()):
            bits = values.get("mean_serialized_bits")
            bit_text = "N/A" if bits is None else f"{bits:.0f}"
            print(f"  {name}: bits={bit_text} mad={values['mean_mad_vs_gt']:.6f}")


if __name__ == "__main__":
    main()
