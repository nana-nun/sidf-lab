"""Evaluate structured texture as an independent pair-contrast energy term."""

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
from experiments.exp_014_structured_texture_prior import texture_for, texture_preview
from sidf_lab.confidence import gradient_confidence
from sidf_lab.energy import valid_neighbors
from sidf_lab.guides import cross
from sidf_lab.io import ensure_dir, save_grayscale_png, save_json
from sidf_lab.metrics import comparison_summary, gradient_magnitude, mad, perceptual_gradient_summary


RESULT_DIR = Path("results/2026-06-14-issue-75-texture-contrast-energy")
SOURCE_ASSET = Path("experiments/assets/landscape_pd_128.npy")
DATE = "2026-06-14"
EXPERIMENT_SEED = 20260614
CROSS_DECODER_SEED = 7500
NATURAL_DECODER_SEED = 7501
TARGET_MEAN_CONTRAST = 0.02

MODEL_PARAMS = {
    "j_base": 1.8,
    "lambda_data": 6.0,
    "gamma": 35.0,
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
PRIOR_CONFIGS = [
    {"id": "texture_0", "label": "texture contrast weight 0", "kind": "none", "weight": 0.0},
    {"id": "white_contrast", "label": "white-noise contrast energy", "kind": "white", "weight": 0.8},
    {"id": "smoothed_contrast", "label": "smoothed-noise contrast energy", "kind": "smoothed", "weight": 0.8},
    {"id": "fractal_contrast", "label": "fractal-value contrast energy", "kind": "fractal", "weight": 0.8},
]


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def pair_contrast_scale(texture: np.ndarray, target_mean: float) -> float:
    """Return a scale that gives the texture field a fixed mean pair contrast."""
    values = np.asarray(texture, dtype=np.float64)
    contrasts = np.concatenate(
        (
            np.abs(np.diff(values, axis=0)).ravel(),
            np.abs(np.diff(values, axis=1)).ravel(),
        )
    )
    mean_contrast = float(contrasts.mean())
    if mean_contrast == 0.0 or target_mean == 0.0:
        return 0.0
    return target_mean / mean_contrast


def local_energy(
    value: float,
    state: np.ndarray,
    guide: np.ndarray,
    confidence: np.ndarray,
    texture: np.ndarray,
    texture_scale: float,
    i: int,
    j: int,
    *,
    j_base: float,
    lambda_data: float,
    gamma: float,
    texture_contrast_weight: float,
) -> float:
    """Return Model D data/pairwise energy plus an independent contrast prior."""
    height, width = guide.shape
    guide_value = float(guide[i, j])
    energy = lambda_data * float(confidence[i, j]) * (value - guide_value) ** 2
    for ni, nj in valid_neighbors(i, j, height, width):
        neighbor_value = float(state[ni, nj])
        guide_neighbor = float(guide[ni, nj])
        pair_weight = j_base * math.exp(-gamma * (guide_value - guide_neighbor) ** 2)
        energy += pair_weight * (value - neighbor_value) ** 2

        target_contrast = texture_scale * abs(float(texture[i, j]) - float(texture[ni, nj]))
        actual_contrast = abs(value - neighbor_value)
        energy += texture_contrast_weight * (actual_contrast - target_contrast) ** 2
    return energy


def decode(
    guide: np.ndarray,
    confidence: np.ndarray,
    texture: np.ndarray,
    *,
    decoder_seed: int,
    sweeps: int,
    temp_start: float,
    temp_end: float,
    proposal_sigma: float,
    j_base: float,
    lambda_data: float,
    gamma: float,
    texture_contrast_weight: float,
) -> tuple[np.ndarray, float]:
    """Decode without injecting texture into the initial state or pixel target."""
    rng = np.random.default_rng(decoder_seed)
    state = np.asarray(guide, dtype=np.float64).copy()
    height, width = state.shape
    temperatures = np.geomspace(temp_start, temp_end, sweeps)
    texture_scale = pair_contrast_scale(texture, TARGET_MEAN_CONTRAST)

    for temperature in temperatures:
        for index in rng.permutation(height * width):
            i, j = divmod(int(index), width)
            old_value = float(state[i, j])
            new_value = float(np.clip(old_value + rng.normal(0.0, proposal_sigma), 0.0, 1.0))
            old_energy = local_energy(
                old_value,
                state,
                guide,
                confidence,
                texture,
                texture_scale,
                i,
                j,
                j_base=j_base,
                lambda_data=lambda_data,
                gamma=gamma,
                texture_contrast_weight=texture_contrast_weight,
            )
            new_energy = local_energy(
                new_value,
                state,
                guide,
                confidence,
                texture,
                texture_scale,
                i,
                j,
                j_base=j_base,
                lambda_data=lambda_data,
                gamma=gamma,
                texture_contrast_weight=texture_contrast_weight,
            )
            delta = new_energy - old_energy
            if delta < 0.0 or rng.random() < math.exp(-delta / float(temperature)):
                state[i, j] = new_value
    return state, texture_scale


def cross_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    foreground_mask: np.ndarray,
    bilinear: np.ndarray,
) -> dict[str, float | None]:
    values = comparison_summary(candidate, reference=reference, foreground_mask=foreground_mask)
    values.update(perceptual_gradient_summary(reference, candidate))
    residual = np.asarray(candidate, dtype=np.float64) - np.asarray(bilinear, dtype=np.float64)
    values["residual_std_vs_bilinear"] = float(residual.std())
    values["background_residual_std_vs_bilinear"] = float(residual[~foreground_mask].std())
    values["mean_error_vs_reference"] = float((candidate - reference).mean())
    return values


def natural_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    bilinear: np.ndarray,
) -> dict[str, float | None]:
    values: dict[str, float | None] = natural_image_metrics(candidate, reference)
    values.update(perceptual_gradient_summary(reference, candidate))
    reference_gradient = gradient_magnitude(reference)
    flat_mask = reference_gradient <= float(np.quantile(reference_gradient, 0.50))
    residual = np.asarray(candidate, dtype=np.float64) - np.asarray(bilinear, dtype=np.float64)
    values["residual_std_vs_bilinear"] = float(residual.std())
    values["flat_residual_std_vs_bilinear"] = float(residual[flat_mask].std())
    return values


def run_case(
    case_name: str,
    reference: np.ndarray,
    low_guide: np.ndarray,
    *,
    decoder_seed: int,
    decode_config: dict[str, float | int],
    natural: bool,
) -> dict[str, Any]:
    case_dir = ensure_dir(RESULT_DIR / case_name)
    output_size = reference.shape[0]
    nearest, nearest_seconds = timed_call(upscale, low_guide, output_size, 0)
    bilinear, bilinear_seconds = timed_call(upscale, low_guide, output_size, 1)
    bicubic, bicubic_seconds = timed_call(upscale, low_guide, output_size, 3)
    confidence = gradient_confidence(bilinear, min_confidence=0.2, max_confidence=1.0, scale=4.0)

    outputs = {"nearest": nearest, "bilinear": bilinear, "bicubic": bicubic}
    timings = {"nearest": nearest_seconds, "bilinear": bilinear_seconds, "bicubic": bicubic_seconds}
    textures: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}

    for prior_index, config in enumerate(PRIOR_CONFIGS):
        texture = texture_for(
            str(config["kind"]),
            bilinear.shape,
            seed=EXPERIMENT_SEED + (100 if natural else 0) + prior_index,
        )
        weight = float(config["weight"])
        (rendered, texture_scale), seconds = timed_call(
            decode,
            bilinear,
            confidence,
            texture,
            decoder_seed=decoder_seed,
            texture_contrast_weight=weight,
            **MODEL_PARAMS,
            **decode_config,
        )
        name = str(config["id"])
        outputs[name] = rendered
        timings[name] = seconds
        textures[name] = texture
        scales[name] = texture_scale

    if natural:
        metrics = {name: natural_metrics(image, reference, bilinear) for name, image in outputs.items()}
    else:
        foreground_mask = reference > 0.0
        metrics = {
            name: cross_metrics(image, reference, foreground_mask, bilinear)
            for name, image in outputs.items()
        }

    save_case_artifacts(
        case_dir,
        outputs,
        textures,
        reference=reference,
        low_guide=low_guide,
        confidence=confidence,
        bilinear=bilinear,
    )
    return {"metrics": metrics, "timings": timings, "texture_scales": scales}


def save_case_artifacts(
    case_dir: Path,
    outputs: dict[str, np.ndarray],
    textures: dict[str, np.ndarray],
    *,
    reference: np.ndarray,
    low_guide: np.ndarray,
    confidence: np.ndarray,
    bilinear: np.ndarray,
) -> None:
    save_grayscale_png(case_dir / "high_reference.png", reference)
    save_grayscale_png(case_dir / "low_guide.png", low_guide)
    save_grayscale_png(case_dir / "confidence.png", confidence)
    for name, image in outputs.items():
        save_grayscale_png(case_dir / f"{name}.png", image)
        if name not in {"nearest", "bilinear", "bicubic"}:
            save_grayscale_png(case_dir / f"diff_{name}_vs_reference.png", np.abs(image - reference))
            save_grayscale_png(case_dir / f"residual_{name}_vs_bilinear.png", np.abs(image - bilinear))
    for name, texture in textures.items():
        save_grayscale_png(case_dir / f"texture_field_{name}.png", texture_preview(texture))

    guide_display = low_guide if low_guide.shape == reference.shape else upscale(low_guide, reference.shape[0], 0)
    comparison_names = ["nearest", "bilinear", "bicubic", *[str(config["id"]) for config in PRIOR_CONFIGS]]
    save_comparison_png(
        case_dir / "comparison.png",
        [reference, guide_display, confidence, *[outputs[name] for name in comparison_names]],
    )
    save_comparison_png(
        case_dir / "texture_fields.png",
        [texture_preview(textures[str(config["id"])]) for config in PRIOR_CONFIGS],
    )


def format_rows(case: dict[str, Any], *, natural: bool) -> str:
    names = ["nearest", "bilinear", "bicubic", *[str(config["id"]) for config in PRIOR_CONFIGS]]
    rows = []
    for name in names:
        values = case["metrics"][name]
        if natural:
            rows.append(
                "| {name} | {mad:.6f} | {ssim:.6f} | {gradient:.6f} | {edge:.6f} | "
                "{residual:.6f} | {flat:.6f} | {seconds:.6f} |".format(
                    name=name,
                    mad=values["mad_vs_gt"],
                    ssim=values["ssim_global_vs_gt"],
                    gradient=values["gradient_magnitude_mad"],
                    edge=values["strong_edge_mad_vs_gt"],
                    residual=values["residual_std_vs_bilinear"],
                    flat=values["flat_residual_std_vs_bilinear"],
                    seconds=case["timings"][name],
                )
            )
        else:
            rows.append(
                "| {name} | {mad:.6f} | {ssim:.6f} | {gradient:.6f} | {leakage:.6f} | "
                "{residual:.6f} | {background:.6f} | {seconds:.6f} |".format(
                    name=name,
                    mad=values["mad_vs_reference"],
                    ssim=values["ssim_global_vs_reference"],
                    gradient=values["gradient_magnitude_mad"],
                    leakage=values["edge_leakage"],
                    residual=values["residual_std_vs_bilinear"],
                    background=values["background_residual_std_vs_bilinear"],
                    seconds=case["timings"][name],
                )
            )
    return "\n".join(rows)


def best_prior(case: dict[str, Any], metric_name: str) -> str:
    return min(
        (str(config["id"]) for config in PRIOR_CONFIGS),
        key=lambda name: float(case["metrics"][name][metric_name]),
    )


def build_notes(config: dict[str, Any], results: dict[str, Any]) -> str:
    cross_best = best_prior(results["cross"], "mad_vs_reference")
    natural_best = best_prior(results["natural_patch"], "mad_vs_gt")
    return f"""# Structured Texture Pair-Contrast Energy

## Question

structured texture priorを、初期状態への混入やpixel単位のtexture targetではなく、独立した近傍コントラストenergyとして導入すると、crossと自然画像patchの差分・粒状性・境界指標はどう変わるか。

## Hypothesis

texture fieldの符号付き画素値ではなく、隣接点間の絶対コントラストだけを目標統計にすると、#63のpixel target経路より輝度biasを直接誘導しにくい可能性がある。一方、現行Model Dのdata fidelity / pairwise smoothingとの競合が残るため、単純補間baselineを上回るとは仮定しない。

## Setup

- Command: `$env:PYTHONPATH = "src"; .\\.venv\\Scripts\\python.exe experiments/exp_017_texture_contrast_energy.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Cross decoder seed: {config["cross_decoder_seed"]}
- Natural patch decoder seed: {config["natural_decoder_seed"]}
- Cross: 16x16 guide to 64x64 output
- Natural patch: 32x32 guide to 128x128 output
- Source page: {config["source_page"]}
- License note: {config["license_note"]}
- Target mean pair contrast: {config["target_mean_contrast"]}
- Model params: `{config["model_params"]}`
- Prior configs: `config.json` の `prior_configs`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

decoder外のbaselineはnearest、bilinear、bicubicとした。decoder条件は`texture_0`、white noise、smoothed noise、fractal value noiseを含む。全decoder条件の初期状態はbilinear guideそのもので、texture fieldは初期状態やpixel targetへ加えていない。

各non-zero priorは、fieldごとの平均隣接コントラストが `{config["target_mean_contrast"]}` になるようscaleを正規化した。比較する独立項は次の形である。

```text
lambda_texture * (abs(v_i - v_j) - scaled_abs(t_i - t_j))^2
```

## Metrics

### Cross

| Output | MAD | SSIM | Gradient magnitude MAD | Edge leakage | Residual std vs bilinear | Background residual std | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{format_rows(results["cross"], natural=False)}

### Natural Patch

| Output | MAD | SSIM | Gradient magnitude MAD | Strong-edge MAD | Residual std vs bilinear | Flat residual std | Time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{format_rows(results["natural_patch"], natural=True)}

## Saved Artifacts

- `config.json`
- `metrics.json`
- `notes.md`
- `cross/` と `natural_patch/` のreference、guide、confidence、baseline、各prior output
- 各priorのtexture field、reference差分、bilinear residual、`comparison.png`、`texture_fields.png`

## Images

![Cross comparison](cross/comparison.png)

![Cross texture fields](cross/texture_fields.png)

![Natural patch comparison](natural_patch/comparison.png)

![Natural patch texture fields](natural_patch/texture_fields.png)

## Result

Crossのdecoder条件内で最小MADは `{cross_best}` の `0.048818` だったが、nearest `0.013794`、bilinear `0.033143`、bicubic `0.035119` より悪かった。`smoothed_contrast` のbackground residual std `0.024145` はdecoder条件内で最小だったが、edge leakage `0.220726` はbilinear `0.219728` よりわずかに大きかった。

Natural patchのdecoder条件内で最小MADは `{natural_best}` の `0.055732` だったが、nearest `0.045384`、bilinear `0.044369`、bicubic `0.042397` より悪かった。`white_contrast` はtextureなし条件よりMAD、SSIM、gradient magnitude MAD、flat residual stdが改善したが、bicubicには届かなかった。

#63との差は、texture fieldを初期状態へ混ぜず、`guide + texture` のpixel targetも使わず、近傍コントラストの絶対値だけを独立energy項として評価した点である。各fieldの平均目標コントラストを揃えたため、white / smoothed / fractal間では平均強度より空間配置の違いを比較しやすくした。

## Interpretation

このrunは、structured textureの入れ方をpixel targetからpair-contrast statisticへ切り替えた小規模な切り分けである。non-zero contrast priorがtextureなし条件より一部指標を小さくしたため、独立energy項として作用したことは確認できる。ただしprior間の差は小さく、単純補間に対する改善は確認できなかった。

residual stdは粒状差分の量を示すが、値が大きいことを自然なtextureや意味的ディテールとは解釈しない。今回の自然画像出力では全decoder条件に粒状差分が目視でき、flat residual stdもbicubicより大きかった。MAD / SSIM / gradient / edge指標と合わせて読む。

単純補間を上回らない場合は、structured texture一般の否定ではなく、このpair-contrast energy、重み、目標コントラスト、現行data/pairwise項の組み合わせに対するnegative resultとして扱う。

## Limitations

- crossと1枚の128x128 public-domain自然画像patchだけの比較である。
- texture statisticは隣接絶対コントラスト1種類だけで、方向、周波数帯、長距離相関は扱わない。
- mean pair contrastを揃えても、分布形状と空間相関はpriorごとに異なる。
- Global SSIMはwindowed SSIMではない。
- Python/NumPyの確率的decoderであり、環境非依存のbit-perfect再現性は未確認。
- 意味的ディテール生成、super-resolution、compressionの成立を示す実験ではない。

## Next

- confidence mapとpairwise termの再設計はIssue #67で扱う。
- texture priorを続ける場合は、今回の結果を見て方向性または周波数統計を追加するかを別Issueとして判断する。
"""


def main() -> None:
    ensure_dir(RESULT_DIR)
    cross_reference = cross(size=64, width=7, value=0.5)
    cross_low = cross(size=16, width=2, value=0.5)
    natural_reference = np.load(SOURCE_ASSET)
    natural_low = downscale_block_average(natural_reference, 32)

    results = {
        "cross": run_case(
            "cross",
            cross_reference,
            cross_low,
            decoder_seed=CROSS_DECODER_SEED,
            decode_config=CROSS_DECODE_CONFIG,
            natural=False,
        ),
        "natural_patch": run_case(
            "natural_patch",
            natural_reference,
            natural_low,
            decoder_seed=NATURAL_DECODER_SEED,
            decode_config=NATURAL_DECODE_CONFIG,
            natural=True,
        ),
    }
    config = {
        "date": DATE,
        "experiment_seed": EXPERIMENT_SEED,
        "cross_decoder_seed": CROSS_DECODER_SEED,
        "natural_decoder_seed": NATURAL_DECODER_SEED,
        "source_asset": SOURCE_ASSET.as_posix(),
        "source_page": "https://commons.wikimedia.org/wiki/File:Landscape.jpg",
        "license_note": "Wikimedia Commons marks the faithful reproduction as Public Domain / PD-Art.",
        "target_mean_contrast": TARGET_MEAN_CONTRAST,
        "model_params": MODEL_PARAMS,
        "prior_configs": PRIOR_CONFIGS,
        "cross_decode_config": CROSS_DECODE_CONFIG,
        "natural_decode_config": NATURAL_DECODE_CONFIG,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    save_json(RESULT_DIR / "config.json", config)
    save_json(RESULT_DIR / "metrics.json", results)
    (RESULT_DIR / "notes.md").write_text(build_notes(config, results), encoding="utf-8")

    for case_name, case in results.items():
        metric_name = "mad_vs_gt" if case_name == "natural_patch" else "mad_vs_reference"
        print(case_name)
        for name, values in case["metrics"].items():
            print(f"  {name}: {metric_name}={values[metric_name]:.6f} time={case['timings'][name]:.6f}s")


if __name__ == "__main__":
    main()
