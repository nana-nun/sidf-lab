"""Confidence map helpers."""

from __future__ import annotations

import numpy as np


def gradient_confidence(
    guide: np.ndarray,
    min_confidence: float = 0.1,
    max_confidence: float = 1.0,
    scale: float = 5.0,
) -> np.ndarray:
    """Build a confidence map from guide gradient magnitude."""
    if not 0.0 <= min_confidence <= max_confidence:
        raise ValueError("expected 0 <= min_confidence <= max_confidence")
    guide = np.asarray(guide, dtype=np.float64)
    dy, dx = np.gradient(guide)
    magnitude = np.sqrt(dx * dx + dy * dy)
    max_mag = float(magnitude.max())
    if max_mag == 0.0:
        return np.full_like(guide, min_confidence, dtype=np.float64)
    normalized = np.clip((magnitude / max_mag) * scale, 0.0, 1.0)
    return min_confidence + normalized * (max_confidence - min_confidence)

