"""Annealing loops for SIDF experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sidf_lab.energy import ModelCParams, model_c_local_energy, valid_neighbors


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


def quadratic_objective(
    state: np.ndarray,
    guide: np.ndarray,
    confidence: np.ndarray,
    *,
    j_base: float,
    lambda_data: float,
    gamma: float,
) -> float:
    """Return the once-counted quadratic data and pairwise objective."""
    values = np.asarray(state, dtype=np.float64)
    guide_values = np.asarray(guide, dtype=np.float64)
    confidence_values = np.asarray(confidence, dtype=np.float64)
    if values.shape != guide_values.shape or values.shape != confidence_values.shape:
        raise ValueError("state, guide, and confidence must have the same shape")

    energy = lambda_data * float(
        np.sum(confidence_values * (values - guide_values) ** 2)
    )
    height, width = values.shape
    for i in range(height):
        for j in range(width):
            for ni, nj in valid_neighbors(i, j, height, width):
                if (ni, nj) <= (i, j):
                    continue
                interaction = j_base * math.exp(
                    -gamma
                    * (
                        float(guide_values[i, j])
                        - float(guide_values[ni, nj])
                    )
                    ** 2
                )
                energy += interaction * (
                    float(values[i, j]) - float(values[ni, nj])
                ) ** 2
    return float(energy)


def quadratic_coordinate_descent(
    guide: np.ndarray,
    initial_state: np.ndarray,
    confidence: np.ndarray,
    *,
    j_base: float,
    lambda_data: float,
    gamma: float,
    max_sweeps: int,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Minimize the quadratic objective with fixed row-major coordinate updates."""
    if j_base < 0.0 or lambda_data < 0.0 or gamma < 0.0:
        raise ValueError("objective parameters must be non-negative")
    if max_sweeps <= 0:
        raise ValueError("max_sweeps must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    guide_values = np.asarray(guide, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    confidence_values = np.asarray(confidence, dtype=np.float64)
    if state.shape != guide_values.shape or state.shape != confidence_values.shape:
        raise ValueError("state, guide, and confidence must have the same shape")
    if np.any(confidence_values < 0.0):
        raise ValueError("confidence must be non-negative")

    initial_objective = quadratic_objective(
        state,
        guide_values,
        confidence_values,
        j_base=j_base,
        lambda_data=lambda_data,
        gamma=gamma,
    )
    height, width = state.shape
    updates = 0
    sweeps_completed = 0
    converged = False
    max_change = 0.0

    for sweep in range(max_sweeps):
        max_change = 0.0
        for i in range(height):
            for j in range(width):
                numerator = (
                    lambda_data
                    * float(confidence_values[i, j])
                    * float(guide_values[i, j])
                )
                denominator = lambda_data * float(confidence_values[i, j])
                for ni, nj in valid_neighbors(i, j, height, width):
                    interaction = j_base * math.exp(
                        -gamma
                        * (
                            float(guide_values[i, j])
                            - float(guide_values[ni, nj])
                        )
                        ** 2
                    )
                    numerator += interaction * float(state[ni, nj])
                    denominator += interaction

                if denominator == 0.0:
                    new_value = float(state[i, j])
                else:
                    new_value = float(np.clip(numerator / denominator, 0.0, 1.0))
                change = abs(new_value - float(state[i, j]))
                if change > tolerance:
                    state[i, j] = new_value
                    updates += 1
                    max_change = max(max_change, change)

        sweeps_completed = sweep + 1
        if max_change <= tolerance:
            converged = True
            break

    final_objective = quadratic_objective(
        state,
        guide_values,
        confidence_values,
        j_base=j_base,
        lambda_data=lambda_data,
        gamma=gamma,
    )
    diagnostics: dict[str, float | int | bool] = {
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "objective_decrease": initial_objective - final_objective,
        "updates": updates,
        "sweeps_completed": sweeps_completed,
        "converged": converged,
        "final_max_change": max_change,
    }
    return state, diagnostics

