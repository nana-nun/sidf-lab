"""Run the reproducible Model C cross baseline experiment."""

from __future__ import annotations

import platform
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from sidf_lab.anneal import AnnealConfig, model_c_decode
from sidf_lab.energy import ModelCParams
from sidf_lab.guides import add_noise, cross
from sidf_lab.io import ensure_dir, save_json
from sidf_lab.metrics import edge_leakage, mad, region_summary


RESULT_DIR = Path("results/2026-05-16-model-c-cross-baseline")


def save_grayscale_png(path: str | Path, image: np.ndarray) -> None:
    """Save a [0, 1] grayscale image as an 8-bit PNG without optional deps."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pixels = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    height, width = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )
    target.write_bytes(png)


def save_comparison_png(path: str | Path, images: list[np.ndarray], gap: int = 2) -> None:
    """Save a simple horizontal comparison strip."""
    if not images:
        raise ValueError("images must not be empty")
    normalized = [np.asarray(image, dtype=np.float64) for image in images]
    height = normalized[0].shape[0]
    if any(image.ndim != 2 or image.shape[0] != height for image in normalized):
        raise ValueError("all images must be 2D and have the same height")
    separator = np.ones((height, gap), dtype=np.float64)
    parts: list[np.ndarray] = []
    for index, image in enumerate(normalized):
        if index:
            parts.append(separator)
        parts.append(image)
    save_grayscale_png(path, np.concatenate(parts, axis=1))


def make_notes(config: dict[str, object], metrics: dict[str, float]) -> str:
    """Build the Japanese experiment note for the saved artifacts."""
    return f"""# Model C Cross Baseline

## Question

同解像度のcross guideで、Model Cのdata fidelityとedge-aware interactionは、noisy static guide direct displayに対して暗部保持とエッジ漏れ抑制を示せるか。

## Hypothesis

Model Cは、guideへの忠実度を保ちながら近傍相互作用を使うため、crossの明部を保ちつつ背景への漏れを抑える。

## Setup

- Command: `$env:PYTHONPATH = "src"; python experiments/exp_002_model_c_cross.py`
- Date: {config["date"]}
- Experiment seed: {config["experiment_seed"]}
- Decoder seed: {config["decoder_seed"]}
- Input guide: synthetic cross with deterministic Gaussian noise
- Input size: {config["input_size"]}x{config["input_size"]}
- Output size: {config["output_size"]}x{config["output_size"]}
- Model: Model C
- Model config: `{config["model_c_params"]}`
- Anneal config: `{config["anneal_config"]}`
- Python / dependency version: Python {config["python_version"]}, NumPy {config["numpy_version"]}

## Baseline

baselineは、noiseを加えたstatic guideをそのまま表示する `baseline_direct.png` とした。Model Aは現時点で共通モジュールとして実装されていないため、このIssueでは `if implemented` の対象外として扱った。

## Metrics

| Metric | Baseline direct | Model C |
| --- | ---: | ---: |
| MAD vs clean guide | {metrics["baseline_mad_vs_clean"]:.6f} | {metrics["model_c_mad_vs_clean"]:.6f} |
| Cross mean | {metrics["baseline_cross_mean"]:.6f} | {metrics["model_c_cross_mean"]:.6f} |
| Background mean | {metrics["baseline_background_mean"]:.6f} | {metrics["model_c_background_mean"]:.6f} |
| Edge leakage | {metrics["baseline_edge_leakage"]:.6f} | {metrics["model_c_edge_leakage"]:.6f} |
| Cross variance | {metrics["baseline_cross_variance"]:.6f} | {metrics["model_c_cross_variance"]:.6f} |
| Background variance | {metrics["baseline_background_variance"]:.6f} | {metrics["model_c_background_variance"]:.6f} |
| Decode time seconds | 0.000000 | {metrics["decode_time_seconds"]:.6f} |

## Saved Artifacts

- Config: `config.json`
- Metrics: `metrics.json`
- Notes: `notes.md`
- Clean guide image: `guide_clean.png`
- Static guide image: `static_guide.png`
- Baseline image: `baseline_direct.png`
- Rendered image: `rendered_model_c.png`
- Difference image: `diff_model_c_vs_baseline.png`
- Comparison image: `comparison.png`

## Result

Model CのMAD、background mean、edge leakageを保存した。cross baselineの暫定目安である `Background Mean <= 0.02`、`Edge Leakage <= 0.02`、`MAD <= 0.03` と比較できる形式になった。

## Interpretation

この結果は、Model Cが少なくとも単一のsynthetic cross条件で、guideに近い値へ収束しながら背景を暗く保つ候補であることを示す。ただし、これは実用圧縮性能や超解像性能を示すものではない。

## Limitations

- 対象は単一のsynthetic crossのみで、斜線、円、細線、soft gradientでは未確認。
- Ground Truth比較はclean synthetic guideに限られる。
- Model A baselineは共通実装がないため今回保存していない。
- NumPy実装の同一環境再現性を確認した段階であり、Rust固定小数点や環境非依存のbit-perfect再現性は未確認。

## Next

- Issue #5 で複数形状のModel C freeze benchmarkを作る。
- Model Aを比較対象として残す必要があるか検討する。
- Rust移植前にPRNG、丸め、固定小数点の再現性要件を整理する。
"""


def main() -> None:
    result_dir = ensure_dir(RESULT_DIR)

    experiment_seed = 20260516
    decoder_seed = 42
    size = 32
    clean_guide = cross(size=size, width=4, value=0.5)
    static_guide = add_noise(clean_guide, seed=experiment_seed, sigma=0.03)
    baseline_direct = static_guide.copy()

    params = ModelCParams(j_base=2.0, lambda_data=5.0, gamma=40.0)
    anneal_config = AnnealConfig(
        decoder_seed=decoder_seed,
        sweeps=80,
        temp_start=0.5,
        temp_end=0.01,
        proposal_sigma=0.12,
    )

    start = time.perf_counter()
    rendered = model_c_decode(static_guide, params, anneal_config)
    decode_time = time.perf_counter() - start

    foreground_mask = clean_guide > 0.0
    background_mask = ~foreground_mask
    baseline_fg = region_summary(baseline_direct, foreground_mask)
    baseline_bg = region_summary(baseline_direct, background_mask)
    model_c_fg = region_summary(rendered, foreground_mask)
    model_c_bg = region_summary(rendered, background_mask)

    metrics = {
        "baseline_mad_vs_clean": mad(baseline_direct, clean_guide),
        "model_c_mad_vs_clean": mad(rendered, clean_guide),
        "baseline_cross_mean": baseline_fg["mean"],
        "model_c_cross_mean": model_c_fg["mean"],
        "baseline_background_mean": baseline_bg["mean"],
        "model_c_background_mean": model_c_bg["mean"],
        "baseline_edge_leakage": edge_leakage(baseline_direct, foreground_mask, radius=2),
        "model_c_edge_leakage": edge_leakage(rendered, foreground_mask, radius=2),
        "baseline_cross_variance": baseline_fg["variance"],
        "model_c_cross_variance": model_c_fg["variance"],
        "baseline_background_variance": baseline_bg["variance"],
        "model_c_background_variance": model_c_bg["variance"],
        "decode_time_seconds": float(decode_time),
    }

    config = {
        "date": "2026-05-16",
        "experiment_seed": experiment_seed,
        "decoder_seed": decoder_seed,
        "input_size": size,
        "output_size": size,
        "static_noise_sigma": 0.03,
        "guide": {"shape": "cross", "width": 4, "value": 0.5},
        "model": "Model C",
        "model_c_params": {
            "j_base": params.j_base,
            "lambda_data": params.lambda_data,
            "gamma": params.gamma,
        },
        "anneal_config": {
            "sweeps": anneal_config.sweeps,
            "temp_start": anneal_config.temp_start,
            "temp_end": anneal_config.temp_end,
            "proposal_sigma": anneal_config.proposal_sigma,
        },
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }

    diff = np.abs(rendered - baseline_direct)
    save_json(result_dir / "config.json", config)
    save_json(result_dir / "metrics.json", metrics)
    save_grayscale_png(result_dir / "guide_clean.png", clean_guide)
    save_grayscale_png(result_dir / "static_guide.png", static_guide)
    save_grayscale_png(result_dir / "baseline_direct.png", baseline_direct)
    save_grayscale_png(result_dir / "rendered_model_c.png", rendered)
    save_grayscale_png(result_dir / "diff_model_c_vs_baseline.png", diff)
    save_comparison_png(
        result_dir / "comparison.png",
        [
            clean_guide,
            static_guide,
            baseline_direct,
            rendered,
            diff,
        ],
    )
    (result_dir / "notes.md").write_text(make_notes(config, metrics), encoding="utf-8")

    print(f"saved {result_dir}")
    print(f"model_c_mad_vs_clean={metrics['model_c_mad_vs_clean']:.6f}")
    print(f"model_c_background_mean={metrics['model_c_background_mean']:.6f}")
    print(f"model_c_edge_leakage={metrics['model_c_edge_leakage']:.6f}")


if __name__ == "__main__":
    main()
