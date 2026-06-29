"""Compare Model E parameterization candidates on source-split patches."""

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


RESULT_DIR = Path("results/2026-06-29-issue-122-model-e-parameterization-redesign")
DATE = "2026-06-29"
EXPERIMENT_SEED = 20260629
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


def make_specs() -> list[dict[str, Any]]:
    return [
        {"name": "fourier_mid", "group": "classical", "spec": INRSpec("fourier", order=2), "seed": 6101},
        {"name": "rff_small", "group": "classical", "spec": INRSpec("rff", feature_count=4), "seed": 6102},
        {"name": "siren_small", "group": "classical", "spec": INRSpec("siren", feature_count=6), "seed": 6103},
        {"name": "mlp_small", "group": "classical", "spec": INRSpec("mlp", feature_count=6), "seed": 6104},
        {"name": "model_e_single", "group": "current_model_e", "spec": INRSpec("model_e_single", depth=6, states=1), "seed": 6105},
        {"name": "model_e_coupled", "group": "current_model_e", "spec": INRSpec("model_e_coupled", depth=3, states=3), "seed": 6106},
        {"name": "candidate_a_ladder", "group": "candidate", "spec": INRSpec("model_e_ladder", depth=1, states=2), "seed": 6107},
        {
            "name": "candidate_b_frequency_table",
            "group": "candidate",
            "spec": INRSpec("model_e_frequency_table", depth=2, states=2, feature_count=4),
            "seed": 6108,
        },
        {"name": "candidate_c_modulated", "group": "candidate", "spec": INRSpec("model_e_modulated", depth=3, states=2), "seed": 6109},
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
    padded = np.pad(values, 1, mode="edge")
    lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * values
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "gradient_magnitude_mean": float(grad.mean()),
        "gradient_magnitude_max": float(grad.max()),
        "laplacian_abs_mean": float(np.mean(np.abs(lap))),
    }


def clipping_summary(parameters: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(parameters, dtype=np.float64)
    clipped = (values <= QUANTIZATION.min_value) | (values >= QUANTIZATION.max_value)
    return {
        "clipped_count": int(np.count_nonzero(clipped)),
        "parameter_count": int(values.size),
        "clipped_ratio": float(np.count_nonzero(clipped) / max(values.size, 1)),
        "min_parameter": float(values.min()) if values.size else 0.0,
        "max_parameter": float(values.max()) if values.size else 0.0,
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
        save_grayscale_png(case_dir / f"{spec_record['name']}_float.png", result.float_decoded)
        save_grayscale_png(case_dir / f"{spec_record['name']}_quantized.png", result.quantized_decoded)
        save_grayscale_png(case_dir / f"diff_{spec_record['name']}_vs_gt.png", np.abs(result.quantized_decoded - reference))
        candidate_outputs[spec_record["name"]] = {
            "group": spec_record["group"],
            "family": result.spec.family,
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
            "clipping": clipping_summary(result.parameters),
            "parameters": result.restored_parameters,
            "image": result.quantized_decoded,
            "float_image": result.float_decoded,
        }

    best_overall = min(candidate_outputs, key=lambda name: candidate_outputs[name]["metrics"]["mad_vs_gt"])
    best_classical = min(
        (name for name, output in candidate_outputs.items() if output["group"] == "classical"),
        key=lambda name: candidate_outputs[name]["metrics"]["mad_vs_gt"],
    )
    best_current = min(
        (name for name, output in candidate_outputs.items() if output["group"] == "current_model_e"),
        key=lambda name: candidate_outputs[name]["metrics"]["mad_vs_gt"],
    )
    best_candidate = min(
        (name for name, output in candidate_outputs.items() if output["group"] == "candidate"),
        key=lambda name: candidate_outputs[name]["metrics"]["mad_vs_gt"],
    )

    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    for name, image in baseline_outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
    save_grayscale_png(case_dir / "diff_bicubic_vs_gt.png", np.abs(bicubic - reference))
    save_comparison_png(
        case_dir / "comparison.png",
        [
            reference,
            bicubic,
            candidate_outputs[best_classical]["image"],
            candidate_outputs[best_current]["image"],
            candidate_outputs[best_candidate]["image"],
            np.abs(candidate_outputs[best_candidate]["image"] - reference),
        ],
    )

    extrapolated: dict[str, Any] = {}
    if case["split"] == "evaluation" and case["name"].endswith("_br"):
        extra_names = ["bicubic", best_classical, best_current, best_candidate]
        bicubic_extra = upscale(low_guide, EXTRAPOLATE_SIZE, 3)
        save_grayscale_png(case_dir / "extrapolated_bicubic.png", bicubic_extra)
        extrapolated["bicubic"] = artifact_summary(bicubic_extra)
        extra_images = [bicubic_extra]
        for name in extra_names[1:]:
            spec = next(record["spec"] for record in specs if record["name"] == name)
            image = decode_inr(spec, low_guide, (EXTRAPOLATE_SIZE, EXTRAPOLATE_SIZE), candidate_outputs[name]["parameters"])
            save_grayscale_png(case_dir / f"extrapolated_{name}.png", image)
            extrapolated[name] = artifact_summary(image)
            extra_images.append(image)
        save_comparison_png(case_dir / "extrapolated_comparison.png", extra_images)

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
        "best_overall": best_overall,
        "best_classical": best_classical,
        "best_current_model_e": best_current,
        "best_new_candidate": best_candidate,
        "extrapolated": extrapolated,
    }


def aggregate_by_split(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["development", "evaluation"]:
        selected = [case for case in case_results if case["split"] == split]
        out[split] = {}
        if not selected:
            continue
        for baseline in ["nearest", "bilinear", "bicubic"]:
            values = [case["baselines"][baseline]["metrics"]["mad_vs_gt"] for case in selected]
            out[split][baseline] = {"group": "image", "family": "image", "mean_quantized_mad": float(np.mean(values))}
        for name in selected[0]["candidates"]:
            records = [case["candidates"][name] for case in selected]
            out[split][name] = {
                "group": records[0]["group"],
                "family": records[0]["family"],
                "mean_float_mad": float(np.mean([record["float_metrics"]["mad_vs_gt"] for record in records])),
                "mean_quantized_mad": float(np.mean([record["metrics"]["mad_vs_gt"] for record in records])),
                "mean_float_to_quantized_mad_delta": float(np.mean([record["float_to_quantized_mad_delta"] for record in records])),
                "mean_serialized_bits": float(np.mean([record["serialized_bits"] for record in records])),
                "mean_fit_seconds": float(np.mean([record["fit_seconds"] for record in records])),
                "mean_decode_seconds": float(np.mean([record["decode_seconds"] for record in records])),
                "mean_clipped_ratio": float(np.mean([record["clipping"]["clipped_ratio"] for record in records])),
                "parameter_group_bits": records[0]["parameter_group_bits"],
            }
    return out


def markdown_rows(aggregate: dict[str, Any]) -> str:
    rows = []
    for split in ["development", "evaluation"]:
        for name, values in sorted(aggregate[split].items()):
            rows.append(
                "| {split} | {name} | {group} | {family} | {bits} | {float_mad} | {quant_mad:.6f} | {delta} | {clip} | {fit} |".format(
                    split=split,
                    name=name,
                    group=values["group"],
                    family=values["family"],
                    bits="N/A" if "mean_serialized_bits" not in values else f"{values['mean_serialized_bits']:.0f}",
                    float_mad="N/A" if "mean_float_mad" not in values else f"{values['mean_float_mad']:.6f}",
                    quant_mad=values["mean_quantized_mad"],
                    delta="N/A"
                    if "mean_float_to_quantized_mad_delta" not in values
                    else f"{values['mean_float_to_quantized_mad_delta']:.6f}",
                    clip="N/A" if "mean_clipped_ratio" not in values else f"{values['mean_clipped_ratio']:.4f}",
                    fit="N/A" if "mean_fit_seconds" not in values else f"{values['mean_fit_seconds']:.4f}",
                )
            )
    return "\n".join(rows)


def group_bits_rows(aggregate: dict[str, Any]) -> str:
    rows = []
    for name, values in sorted(aggregate["evaluation"].items()):
        if "parameter_group_bits" not in values:
            continue
        group_text = ", ".join(f"{key}={value}" for key, value in sorted(values["parameter_group_bits"].items()))
        rows.append(f"| {name} | {values['family']} | {values['mean_serialized_bits']:.0f} | {group_text} |")
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


def best_by_group(aggregate: dict[str, Any], group: str) -> tuple[str, dict[str, Any]]:
    items = [(name, values) for name, values in aggregate["evaluation"].items() if values["group"] == group]
    return min(items, key=lambda item: item[1]["mean_quantized_mad"])


def build_notes(config: dict[str, Any], case_results: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    best_classical = best_by_group(aggregate, "classical")
    best_current = best_by_group(aggregate, "current_model_e")
    best_candidate = best_by_group(aggregate, "candidate")
    return f"""# Model E Parameterization Redesign Source-Split Comparison

## Question

Issue #116 で整理し、Issue #121 で実装した Model E Candidate A / B / C は、#104 と同じ source-split fixture、baseline、12-bit quantization、incremental side-bit accounting の条件で、現行 Model E single/coupled より続ける価値を示すか。

## Hypothesis

Candidate A の fixed frequency ladder、Candidate B の compact frequency table、Candidate C の coordinate frequency + guide modulation は、現行Model Eで混ざっていた frequency placement と guide modulation を分けるため、現行 Model E より低い量子化後MADを示す可能性がある。一方、同じ条件で classical INR baseline や bicubic baselineを上回らない場合、このrunだけで採用候補とは扱わない。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_024_model_e_parameterization_redesign.py`
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

## Baseline

画像baselineは nearest、bilinear、bicubic。Classical parameterized baseline は Fourier、RFF、SIREN、small MLP。現行Model Eは single-state / coupled-state。新候補は Candidate A `model_e_ladder`、Candidate B `model_e_frequency_table`、Candidate C `model_e_modulated`。全candidateは同じ `fit_inr` interface、同じrandom-search step数、同じ12-bit量子化規則でfitした。

## Result

| Split | Output | Group | Family | Mean serialized side bits | Mean float MAD | Mean quantized MAD | Mean quantized-float MAD delta | Mean clipped ratio | Mean fit seconds |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{markdown_rows(aggregate)}

Evaluation splitの最良classical INR候補は `{best_classical[0]}` で、mean quantized MAD `{best_classical[1]["mean_quantized_mad"]:.6f}`、mean serialized side bits `{best_classical[1]["mean_serialized_bits"]:.0f}` だった。

Evaluation splitの最良current Model E候補は `{best_current[0]}` で、mean quantized MAD `{best_current[1]["mean_quantized_mad"]:.6f}`、mean serialized side bits `{best_current[1]["mean_serialized_bits"]:.0f}` だった。

Evaluation splitの最良new candidateは `{best_candidate[0]}` で、mean quantized MAD `{best_candidate[1]["mean_quantized_mad"]:.6f}`、mean serialized side bits `{best_candidate[1]["mean_serialized_bits"]:.0f}` だった。

### Evaluation Side-Bit Groups

| Output | Family | Mean serialized side bits | Parameter group bits |
| --- | --- | ---: | --- |
{group_bits_rows(aggregate)}

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
- Per-case images: `high_reference.png`, `low_guide.png`, `nearest.png`, `bilinear.png`, `bicubic.png`, `*_float.png`, `*_quantized.png`, `diff_*_vs_gt.png`, `comparison.png`
- Extrapolated diagnostic: `evaluation_hokusai_wave_br/extrapolated_comparison.png`

## Images

![Development Hobbema TL comparison](development_hobbema_landscape_tl/comparison.png)

![Development Hobbema BR comparison](development_hobbema_landscape_br/comparison.png)

![Evaluation Hokusai TL comparison](evaluation_hokusai_wave_tl/comparison.png)

![Evaluation Hokusai BR comparison](evaluation_hokusai_wave_br/comparison.png)

![Evaluation extrapolated outputs](evaluation_hokusai_wave_br/extrapolated_comparison.png)

## Interpretation

このrunは、Model E parameterization候補の小規模比較であり、Model E一般の採否や画像品質の一般結論ではない。評価では、float結果、12-bit量子化後結果、serialized side bits、clipping ratioを分けて読む。

新candidateが現行Model Eより改善しても、classical INR baselineやbicubic baselineとの関係を見ずに採用根拠とはしない。逆に、この小規模runで改善しない場合も、autograd optimizer、別bit depth、別candidate設計、より広いdatasetの否定ではない。

今回のevaluation splitでは、最良new candidateの `{best_candidate[0]}` は最良current Model Eの `{best_current[0]}` より量子化後MADを下げた。一方で、最良classical INRの `{best_classical[0]}`、nearest baseline、bicubic baselineを上回らなかった。そのため、Candidate A/B/Cのcompact設定は「現行Model Eよりは一部改善したが、採用候補へ戻す根拠には不足」と扱う。

## Limitations

- Development source 1件、evaluation source 1件からの64x64 crop 2件ずつという小規模datasetであり、一般的な画像集合を代表しない。
- optimizerは #104 と同じ dependency-free random search であり、各parameterizationの到達可能品質を保証しない。
- Candidate A/B/Cは compact な最小設定であり、周波数数、depth、states、parameter group別quantizationは未探索である。
- `incremental_side_bits` はparameter side informationの簡易見積もりであり、guide bits、container overhead、entropy codingを含む `total_description_bits` ではない。
- extrapolated outputはartifact診断であり、Ground Truth比較ではない。
- compression、super-resolution、quantum advantageは主張しない。

## Next

- この結果を `docs/model-decision-map.md` と `docs/research-state.md` に反映する。
- Candidate A/C の改善がcompact設定とrandom-search条件に限定されるかを調べる場合は、candidate size、bit-depth耐性、optimizer依存を別Issueへ分ける。
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
                "group": record["group"],
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
            print(f"  {name}: bits={bit_text} mad={values['mean_quantized_mad']:.6f}")


if __name__ == "__main__":
    main()
