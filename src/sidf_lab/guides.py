"""Synthetic guide image helpers for SIDF experiments."""

from __future__ import annotations

import numpy as np


def cross(size: int, width: int | None = None, value: float = 0.5) -> np.ndarray:
    """Return a square grayscale cross guide in range [0, 1]."""
    if size <= 0:
        raise ValueError("size must be positive")
    if width is None:
        width = max(1, size // 8)
    if width <= 0 or width > size:
        raise ValueError("width must be in 1..size")

    guide = np.zeros((size, size), dtype=np.float64)
    start = (size - width) // 2
    end = start + width
    guide[start:end, :] = value
    guide[:, start:end] = value
    return np.clip(guide, 0.0, 1.0)


def diagonal(size: int, width: int = 1, value: float = 0.5) -> np.ndarray:
    """Return a square diagonal-line guide."""
    if size <= 0:
        raise ValueError("size must be positive")
    if width <= 0:
        raise ValueError("width must be positive")

    guide = np.zeros((size, size), dtype=np.float64)
    rows, cols = np.indices((size, size))
    guide[np.abs(rows - cols) < width] = value
    return np.clip(guide, 0.0, 1.0)


def circle(size: int, radius: float | None = None, value: float = 0.5) -> np.ndarray:
    """Return a filled circular guide."""
    if size <= 0:
        raise ValueError("size must be positive")
    if radius is None:
        radius = size * 0.25
    if radius <= 0:
        raise ValueError("radius must be positive")

    center = (size - 1) / 2.0
    rows, cols = np.indices((size, size))
    dist = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)
    guide = np.zeros((size, size), dtype=np.float64)
    guide[dist <= radius] = value
    return np.clip(guide, 0.0, 1.0)


def horizontal_gradient(size: int) -> np.ndarray:
    """Return a left-to-right grayscale gradient."""
    if size <= 0:
        raise ValueError("size must be positive")
    row = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return np.tile(row, (size, 1))


def add_noise(image: np.ndarray, seed: int, sigma: float) -> np.ndarray:
    """Add deterministic Gaussian noise and clip to [0, 1]."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = np.random.default_rng(seed)
    noisy = np.asarray(image, dtype=np.float64) + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0.0, 1.0)

