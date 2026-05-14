"""Energy functions for SIDF reconstruction models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelCParams:
    """Parameters for edge-preserving stochastic relaxation."""

    j_base: float = 2.0
    lambda_data: float = 5.0
    gamma: float = 40.0


def valid_neighbors(i: int, j: int, height: int, width: int) -> list[tuple[int, int]]:
    """Return 4-neighborhood without torus wrapping."""
    neighbors: list[tuple[int, int]] = []
    if i > 0:
        neighbors.append((i - 1, j))
    if i < height - 1:
        neighbors.append((i + 1, j))
    if j > 0:
        neighbors.append((i, j - 1))
    if j < width - 1:
        neighbors.append((i, j + 1))
    return neighbors


def model_c_local_energy(
    value: float,
    state: np.ndarray,
    guide: np.ndarray,
    i: int,
    j: int,
    params: ModelCParams,
) -> float:
    """Compute local Model C energy for one candidate pixel value."""
    height, width = guide.shape
    s_i = float(guide[i, j])
    e_fidelity = params.lambda_data * (value - s_i) ** 2
    e_smooth = 0.0
    for ni, nj in valid_neighbors(i, j, height, width):
        n_val = float(state[ni, nj])
        s_n = float(guide[ni, nj])
        j_ij = params.j_base * math.exp(-params.gamma * (s_i - s_n) ** 2)
        e_smooth += j_ij * (value - n_val) ** 2
    return e_fidelity + e_smooth

