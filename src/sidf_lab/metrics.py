"""Metrics for SIDF reconstruction outputs."""

from __future__ import annotations

import math

import numpy as np


def mad(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference."""
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def region_summary(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Return mean and variance for a masked region."""
    values = np.asarray(image, dtype=np.float64)[mask]
    if values.size == 0:
        raise ValueError("mask selects no pixels")
    return {"mean": float(values.mean()), "variance": float(values.var())}


def edge_leakage(image: np.ndarray, foreground_mask: np.ndarray, radius: int = 2) -> float:
    """Mean value in a simple dilation ring around the foreground mask."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    mask = np.asarray(foreground_mask, dtype=bool)
    dilated = mask.copy()
    for _ in range(radius):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        dilated = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    ring = dilated & ~mask
    if not np.any(ring):
        return 0.0
    return float(np.asarray(image, dtype=np.float64)[ring].mean())


def psnr(reference: np.ndarray, candidate: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio against a reference image.

    Use only when a meaningful reference or Ground Truth image exists. For
    SIDF guide-only comparisons, report this as comparison-to-reference, not
    as proof of super-resolution quality.
    """
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate must have the same shape")
    mse = float(np.mean((ref - cand) ** 2))
    if mse == 0.0:
        return math.inf
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def ssim_global(reference: np.ndarray, candidate: np.ndarray, data_range: float = 1.0) -> float:
    """Global grayscale SSIM against a reference image.

    This is a small dependency-free SSIM helper over the whole image, not a
    windowed perceptual SSIM implementation. Use it for controlled experiment
    comparisons with a reference image, and keep Ground Truth limitations
    explicit in notes.
    """
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate must have the same shape")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ref_mean = float(ref.mean())
    cand_mean = float(cand.mean())
    ref_var = float(ref.var())
    cand_var = float(cand.var())
    covariance = float(np.mean((ref - ref_mean) * (cand - cand_mean)))
    numerator = (2.0 * ref_mean * cand_mean + c1) * (2.0 * covariance + c2)
    denominator = (ref_mean**2 + cand_mean**2 + c1) * (ref_var + cand_var + c2)
    return float(numerator / denominator)


def edge_width(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    low_fraction: float = 0.1,
    high_fraction: float = 0.9,
    max_radius: int = 8,
) -> float | None:
    """Approximate edge transition width in pixels.

    The helper normalizes image values between the foreground and background
    means, then estimates how many boundary-band pixels sit between
    ``low_fraction`` and ``high_fraction``. It is intended for synthetic masks
    used by Model D comparisons; for soft gradients or masks with no clear
    foreground/background split, return ``None`` or interpret cautiously.
    """
    if not 0.0 <= low_fraction < high_fraction <= 1.0:
        raise ValueError("fractions must satisfy 0 <= low < high <= 1")
    if max_radius <= 0:
        raise ValueError("max_radius must be positive")

    values = np.asarray(image, dtype=np.float64)
    mask = np.asarray(foreground_mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("image and foreground_mask must have the same shape")
    if not np.any(mask) or np.all(mask):
        return None

    foreground_mean = float(values[mask].mean())
    background_mean = float(values[~mask].mean())
    dynamic_range = foreground_mean - background_mean
    if abs(dynamic_range) < 1e-12:
        return None

    normalized = (values - background_mean) / dynamic_range
    if dynamic_range < 0.0:
        normalized = 1.0 - normalized

    boundary_count = _boundary_count(mask)
    if boundary_count == 0:
        return None

    band = _dilate(mask, max_radius) & _dilate(~mask, max_radius)
    transition = band & (normalized >= low_fraction) & (normalized <= high_fraction)
    return float(np.count_nonzero(transition) / boundary_count)


def comparison_summary(
    candidate: np.ndarray,
    reference: np.ndarray | None = None,
    foreground_mask: np.ndarray | None = None,
    edge_radius: int = 2,
    edge_width_radius: int = 8,
    data_range: float = 1.0,
) -> dict[str, float | None]:
    """Return common metrics for baseline and Model D comparison experiments."""
    values = np.asarray(candidate, dtype=np.float64)
    metrics: dict[str, float | None] = {}

    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        metrics["mad_vs_reference"] = mad(values, ref)
        metrics["psnr_vs_reference"] = psnr(ref, values, data_range=data_range)
        metrics["ssim_global_vs_reference"] = ssim_global(ref, values, data_range=data_range)

    if foreground_mask is not None:
        mask = np.asarray(foreground_mask, dtype=bool)
        if values.shape != mask.shape:
            raise ValueError("candidate and foreground_mask must have the same shape")
        foreground = region_summary(values, mask)
        background = region_summary(values, ~mask)
        metrics["foreground_mean"] = foreground["mean"]
        metrics["foreground_variance"] = foreground["variance"]
        metrics["background_mean"] = background["mean"]
        metrics["background_variance"] = background["variance"]
        metrics["edge_leakage"] = edge_leakage(values, mask, radius=edge_radius)
        metrics["edge_width_pixels"] = edge_width(values, mask, max_radius=edge_width_radius)

    return metrics


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    dilated = np.asarray(mask, dtype=bool).copy()
    for _ in range(radius):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        dilated = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return dilated


def _boundary_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    return int(np.count_nonzero(boundary))

