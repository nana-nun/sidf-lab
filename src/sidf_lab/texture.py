"""Deterministic texture priors for SIDF experiments."""

from __future__ import annotations

import numpy as np


def white_noise(shape: tuple[int, int], seed: int, sigma: float = 1.0) -> np.ndarray:
    """Return deterministic zero-mean white noise."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, shape).astype(np.float64)

