"""Deterministic texture priors for SIDF experiments."""

from __future__ import annotations

import numpy as np


def _validate_shape(shape: tuple[int, int]) -> None:
    if len(shape) != 2:
        raise ValueError("shape must be two-dimensional")
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape dimensions must be positive")


def _normalize_zero_mean(field: np.ndarray, sigma: float) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    centered = field - float(field.mean())
    std = float(centered.std())
    if std == 0.0 or sigma == 0.0:
        return np.zeros_like(centered, dtype=np.float64)
    return (centered / std * sigma).astype(np.float64)


def white_noise(shape: tuple[int, int], seed: int, sigma: float = 1.0) -> np.ndarray:
    """Return deterministic zero-mean white noise."""
    _validate_shape(shape)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, shape).astype(np.float64)


def smoothed_noise(
    shape: tuple[int, int],
    seed: int,
    sigma: float = 1.0,
    radius: int = 2,
) -> np.ndarray:
    """Return deterministic low-pass noise for structured-prior experiments.

    This candidate is a baseline for later texture ablation, not evidence that
    structured texture improves SIDF reconstruction quality.
    """
    _validate_shape(shape)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    rng = np.random.default_rng(seed)
    field = rng.normal(0.0, 1.0, shape).astype(np.float64)
    if radius == 0:
        return _normalize_zero_mean(field, sigma)

    kernel = np.full(2 * radius + 1, 1.0 / float(2 * radius + 1), dtype=np.float64)
    padded_y = np.pad(field, ((radius, radius), (0, 0)), mode="edge")
    smoothed_y = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="valid"), 0, padded_y)
    padded_x = np.pad(smoothed_y, ((0, 0), (radius, radius)), mode="edge")
    smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded_x)
    return _normalize_zero_mean(smoothed, sigma)


def _resize_bilinear(source: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    src_h, src_w = source.shape

    y_positions = np.linspace(0.0, float(src_h - 1), height, dtype=np.float64)
    x_positions = np.linspace(0.0, float(src_w - 1), width, dtype=np.float64)
    output = np.empty((height, width), dtype=np.float64)

    for out_y, y in enumerate(y_positions):
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, src_h - 1)
        wy = y - float(y0)
        for out_x, x in enumerate(x_positions):
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, src_w - 1)
            wx = x - float(x0)
            top = (1.0 - wx) * source[y0, x0] + wx * source[y0, x1]
            bottom = (1.0 - wx) * source[y1, x0] + wx * source[y1, x1]
            output[out_y, out_x] = (1.0 - wy) * top + wy * bottom
    return output


def fractal_value_noise(
    shape: tuple[int, int],
    seed: int,
    octaves: int = 4,
    base_frequency: float = 2.0,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Return deterministic multi-octave value noise for texture experiments.

    The output is normalized for comparison against the white-noise baseline;
    it is still only a candidate prior until measured in an experiment.
    """
    _validate_shape(shape)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if octaves < 1:
        raise ValueError("octaves must be at least one")
    if base_frequency <= 0:
        raise ValueError("base_frequency must be positive")
    if lacunarity <= 0:
        raise ValueError("lacunarity must be positive")
    if gain < 0:
        raise ValueError("gain must be non-negative")

    rng = np.random.default_rng(seed)
    combined = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    frequency = float(base_frequency)

    for _ in range(octaves):
        grid_h = max(2, int(np.ceil(frequency * shape[0] / min(shape))) + 1)
        grid_w = max(2, int(np.ceil(frequency * shape[1] / min(shape))) + 1)
        grid = rng.normal(0.0, 1.0, (grid_h, grid_w)).astype(np.float64)
        combined += amplitude * _resize_bilinear(grid, shape)
        amplitude *= gain
        frequency *= lacunarity

    return _normalize_zero_mean(combined, sigma)

