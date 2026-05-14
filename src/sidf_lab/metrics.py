"""Metrics for SIDF reconstruction outputs."""

from __future__ import annotations

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

