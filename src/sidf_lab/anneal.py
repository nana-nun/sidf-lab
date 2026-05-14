"""Annealing loops for SIDF experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sidf_lab.energy import ModelCParams, model_c_local_energy


@dataclass(frozen=True)
class AnnealConfig:
    """Generic stochastic relaxation settings."""

    decoder_seed: int
    sweeps: int = 40
    temp_start: float = 0.5
    temp_end: float = 0.01
    proposal_sigma: float = 0.15


def model_c_decode(guide: np.ndarray, params: ModelCParams, config: AnnealConfig) -> np.ndarray:
    """Decode a grayscale guide with Model C stochastic relaxation."""
    if config.sweeps <= 0:
        raise ValueError("sweeps must be positive")
    if config.temp_start <= 0 or config.temp_end <= 0:
        raise ValueError("temperatures must be positive")
    if config.proposal_sigma < 0:
        raise ValueError("proposal_sigma must be non-negative")

    guide = np.asarray(guide, dtype=np.float64)
    height, width = guide.shape
    rng = np.random.default_rng(config.decoder_seed)
    state = rng.random((height, width), dtype=np.float64)
    temperatures = np.geomspace(config.temp_start, config.temp_end, config.sweeps)

    for temp in temperatures:
        for idx in rng.permutation(height * width):
            i, j = divmod(int(idx), width)
            old_val = float(state[i, j])
            new_val = float(np.clip(old_val + rng.normal(0.0, config.proposal_sigma), 0.0, 1.0))
            old_energy = model_c_local_energy(old_val, state, guide, i, j, params)
            new_energy = model_c_local_energy(new_val, state, guide, i, j, params)
            delta = new_energy - old_energy
            if delta < 0.0 or rng.random() < math.exp(-delta / float(temp)):
                state[i, j] = new_val

    return state

