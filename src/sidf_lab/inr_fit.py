"""Minimal fitting helpers for Model E and classical INR baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sidf_lab.model_e import (
    ModelEParams,
    QuantizationSpec,
    bilinear_resize,
    model_e_features,
    model_e_residual,
)


INRFamily = Literal[
    "fourier",
    "rff",
    "siren",
    "mlp",
    "model_e_single",
    "model_e_coupled",
    "model_e_ladder",
    "model_e_frequency_table",
    "model_e_modulated",
]


@dataclass(frozen=True)
class INRSpec:
    """Small model descriptor for trainable INR residual baselines."""

    family: INRFamily
    order: int = 1
    feature_count: int = 6
    depth: int = 3
    states: int = 1
    residual_limit: float = 0.35

    def __post_init__(self) -> None:
        if self.family not in {
            "fourier",
            "rff",
            "siren",
            "mlp",
            "model_e_single",
            "model_e_coupled",
            "model_e_ladder",
            "model_e_frequency_table",
            "model_e_modulated",
        }:
            raise ValueError(f"unknown INR family: {self.family}")
        if self.order <= 0:
            raise ValueError("order must be positive")
        if self.feature_count <= 0:
            raise ValueError("feature_count must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.states <= 0:
            raise ValueError("states must be positive")
        if not np.isfinite(self.residual_limit) or self.residual_limit < 0.0:
            raise ValueError("residual_limit must be finite and non-negative")
        if self.family == "model_e_single" and self.states != 1:
            raise ValueError("model_e_single must use exactly one state")
        if self.family == "model_e_coupled" and self.states < 2:
            raise ValueError("model_e_coupled must use at least two states")


@dataclass(frozen=True)
class ParameterLayout:
    """Stable vector layout for one INR spec."""

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]

    @property
    def parameter_count(self) -> int:
        return int(sum(np.prod(shape, dtype=np.int64) for shape in self.shapes))


@dataclass(frozen=True)
class FitResult:
    """Result from a deterministic small fitting run."""

    spec: INRSpec
    parameters: np.ndarray
    quantized_parameters: np.ndarray
    restored_parameters: np.ndarray
    float_decoded: np.ndarray
    quantized_decoded: np.ndarray
    fit_seconds: float
    decode_seconds: float
    float_mse: float
    quantized_mse: float
    bits: dict[str, object]


def make_parameter_layout(spec: INRSpec) -> ParameterLayout:
    """Return the trainable parameter layout for an INR spec."""
    if spec.family == "fourier":
        return ParameterLayout(("readout",), ((1 + 6 * spec.order,),))
    if spec.family == "rff":
        return ParameterLayout(
            ("weights", "bias", "readout"),
            ((5, spec.feature_count), (spec.feature_count,), (2 * spec.feature_count,)),
        )
    if spec.family == "siren":
        return ParameterLayout(
            ("weights", "bias", "readout"),
            ((5, spec.feature_count), (spec.feature_count,), (spec.feature_count,)),
        )
    if spec.family == "mlp":
        return ParameterLayout(
            ("weights", "bias", "readout", "output_bias"),
            ((5, spec.feature_count), (spec.feature_count,), (spec.feature_count,), (1,)),
        )
    if spec.family == "model_e_ladder":
        mode_count = _fixed_ladder_mode_count()
        return ParameterLayout(
            ("scale", "phase", "readout", "residual_scale"),
            (
                (spec.depth, spec.states, mode_count),
                (spec.depth, spec.states),
                (spec.states,),
                (1,),
            ),
        )
    if spec.family == "model_e_frequency_table":
        return ParameterLayout(
            ("frequencies", "phase", "mixing", "readout", "residual_scale"),
            (
                (5, spec.feature_count),
                (spec.feature_count,),
                (spec.depth, spec.states, spec.feature_count),
                (spec.states,),
                (1,),
            ),
        )
    if spec.family == "model_e_modulated":
        return ParameterLayout(
            ("coord_frequency", "phase", "gate", "readout", "residual_scale"),
            (
                (spec.depth, spec.states, 4),
                (spec.depth, spec.states),
                (4,),
                (spec.states,),
                (1,),
            ),
        )
    return ParameterLayout(
        ("layers", "readout", "residual_scale"),
        ((spec.depth, spec.states, 5), (spec.states,), (1,)),
    )


def flatten_parameters(parameters: dict[str, np.ndarray], layout: ParameterLayout) -> np.ndarray:
    """Flatten named parameter arrays according to a stable layout."""
    chunks: list[np.ndarray] = []
    for name, shape in zip(layout.names, layout.shapes):
        if name not in parameters:
            raise ValueError(f"missing parameter: {name}")
        value = np.asarray(parameters[name], dtype=np.float64)
        if value.shape != shape:
            raise ValueError(f"parameter {name} has shape {value.shape}, expected {shape}")
        chunks.append(value.reshape(-1))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)


def unflatten_parameters(vector: np.ndarray, layout: ParameterLayout) -> dict[str, np.ndarray]:
    """Restore named parameter arrays from a flat vector."""
    flat = np.asarray(vector, dtype=np.float64).reshape(-1)
    if flat.size != layout.parameter_count:
        raise ValueError(f"vector has {flat.size} values, expected {layout.parameter_count}")
    out: dict[str, np.ndarray] = {}
    start = 0
    for name, shape in zip(layout.names, layout.shapes):
        size = int(np.prod(shape, dtype=np.int64))
        out[name] = flat[start : start + size].reshape(shape).copy()
        start += size
    return out


def initialize_parameters(spec: INRSpec, seed: int = 0) -> np.ndarray:
    """Create deterministic initial parameters for a spec."""
    rng = np.random.default_rng(seed)
    layout = make_parameter_layout(spec)
    if spec.family == "fourier":
        return np.zeros(layout.parameter_count, dtype=np.float64)
    if spec.family == "rff":
        params = {
            "weights": rng.normal(0.0, 2.0, size=(5, spec.feature_count)),
            "bias": rng.uniform(-np.pi, np.pi, size=spec.feature_count),
            "readout": np.zeros(2 * spec.feature_count, dtype=np.float64),
        }
        return flatten_parameters(params, layout)
    if spec.family == "siren":
        params = {
            "weights": rng.normal(0.0, 1.5, size=(5, spec.feature_count)),
            "bias": rng.uniform(-1.0, 1.0, size=spec.feature_count),
            "readout": np.zeros(spec.feature_count, dtype=np.float64),
        }
        return flatten_parameters(params, layout)
    if spec.family == "mlp":
        params = {
            "weights": rng.normal(0.0, 0.75, size=(5, spec.feature_count)),
            "bias": np.zeros(spec.feature_count, dtype=np.float64),
            "readout": np.zeros(spec.feature_count, dtype=np.float64),
            "output_bias": np.zeros(1, dtype=np.float64),
        }
        return flatten_parameters(params, layout)
    if spec.family == "model_e_ladder":
        params = {
            "scale": rng.normal(0.0, 0.18, size=(spec.depth, spec.states, _fixed_ladder_mode_count())),
            "phase": rng.uniform(-0.25, 0.25, size=(spec.depth, spec.states)),
            "readout": np.linspace(0.5, 1.0, spec.states, dtype=np.float64),
            "residual_scale": np.array([min(0.1, spec.residual_limit)], dtype=np.float64),
        }
        return flatten_parameters(params, layout)
    if spec.family == "model_e_frequency_table":
        params = {
            "frequencies": rng.normal(0.0, 1.0, size=(5, spec.feature_count)),
            "phase": rng.uniform(-np.pi, np.pi, size=spec.feature_count),
            "mixing": rng.normal(0.0, 0.35, size=(spec.depth, spec.states, spec.feature_count)),
            "readout": np.linspace(0.5, 1.0, spec.states, dtype=np.float64),
            "residual_scale": np.array([min(0.1, spec.residual_limit)], dtype=np.float64),
        }
        return flatten_parameters(params, layout)
    if spec.family == "model_e_modulated":
        params = {
            "coord_frequency": rng.normal(0.0, 0.8, size=(spec.depth, spec.states, 4)),
            "phase": rng.uniform(-0.25, 0.25, size=(spec.depth, spec.states)),
            "gate": np.array([0.0, 0.5, 0.25, 0.25], dtype=np.float64),
            "readout": np.linspace(0.5, 1.0, spec.states, dtype=np.float64),
            "residual_scale": np.array([min(0.1, spec.residual_limit)], dtype=np.float64),
        }
        return flatten_parameters(params, layout)
    params = {
        "layers": rng.normal(0.0, 0.5, size=(spec.depth, spec.states, 5)),
        "readout": np.linspace(0.5, 1.0, spec.states, dtype=np.float64),
        "residual_scale": np.array([min(0.1, spec.residual_limit)], dtype=np.float64),
    }
    return flatten_parameters(params, layout)


def decode_inr(
    spec: INRSpec,
    low_guide: np.ndarray,
    output_shape: tuple[int, int],
    parameters: np.ndarray,
) -> np.ndarray:
    """Decode an INR residual reconstruction from flat parameters."""
    base = bilinear_resize(low_guide, output_shape)
    features = model_e_features(low_guide, output_shape)
    residual = _residual_from_features(spec, features, parameters)
    return np.clip(base + residual.reshape(output_shape), 0.0, 1.0)


def quantize_vector(vector: np.ndarray, quantization: QuantizationSpec) -> np.ndarray:
    """Uniformly quantize a parameter vector."""
    levels = (1 << quantization.bits_per_value) - 1
    values = np.asarray(vector, dtype=np.float64)
    normalized = (values - quantization.min_value) / (quantization.max_value - quantization.min_value)
    return np.rint(np.clip(normalized, 0.0, 1.0) * levels).astype(np.int64)


def dequantize_vector(vector: np.ndarray, quantization: QuantizationSpec) -> np.ndarray:
    """Restore a uniformly quantized parameter vector."""
    levels = (1 << quantization.bits_per_value) - 1
    normalized = np.asarray(vector, dtype=np.float64) / levels
    return quantization.min_value + normalized * (quantization.max_value - quantization.min_value)


def estimate_inr_bits(
    spec: INRSpec,
    quantization: QuantizationSpec,
    *,
    extra_structure_bits: int = 32,
) -> dict[str, object]:
    """Return a #98-compatible incremental side-bit estimate."""
    layout = make_parameter_layout(spec)
    parameter_count = layout.parameter_count
    parameter_bits = parameter_count * quantization.bits_per_value
    group_bits = {
        name: int(np.prod(shape, dtype=np.int64) * quantization.bits_per_value)
        for name, shape in zip(layout.names, layout.shapes)
    }
    return {
        "model_header_bits": int(quantization.header_bits),
        "structure_bits": int(extra_structure_bits),
        "quantized_parameter_bits": int(parameter_bits),
        "parameter_count": int(parameter_count),
        "parameter_group_bits": group_bits,
        "incremental_side_bits": int(
            quantization.header_bits + extra_structure_bits + parameter_bits
        ),
    }


def fit_inr(
    spec: INRSpec,
    low_guide: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int = 0,
    steps: int = 64,
    initial_step_scale: float = 0.05,
    quantization: QuantizationSpec | None = None,
) -> FitResult:
    """Fit one INR spec with a deterministic dependency-free random search."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if initial_step_scale < 0.0:
        raise ValueError("initial_step_scale must be non-negative")
    target = np.asarray(reference, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("reference must be a 2D grayscale image")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    params = initialize_parameters(spec, seed)
    params = _fit_linear_readout_if_available(spec, low_guide, target, params)
    best_mse = _mse(decode_inr(spec, low_guide, target.shape, params), target)

    scale = float(initial_step_scale)
    for index in range(steps):
        decay = 1.0 - (index / max(steps, 1))
        proposal = params + rng.normal(0.0, scale * max(decay, 0.05), size=params.shape)
        decoded = decode_inr(spec, low_guide, target.shape, proposal)
        proposal_mse = _mse(decoded, target)
        if proposal_mse <= best_mse:
            params = proposal
            best_mse = proposal_mse

    fit_seconds = float(time.perf_counter() - start)
    quant = quantization or QuantizationSpec(bits_per_value=12, min_value=-1.0, max_value=1.0)
    quantized = quantize_vector(params, quant)
    restored = dequantize_vector(quantized, quant)

    decode_start = time.perf_counter()
    float_decoded = decode_inr(spec, low_guide, target.shape, params)
    quantized_decoded = decode_inr(spec, low_guide, target.shape, restored)
    decode_seconds = float(time.perf_counter() - decode_start)
    return FitResult(
        spec=spec,
        parameters=params.copy(),
        quantized_parameters=quantized.copy(),
        restored_parameters=restored.copy(),
        float_decoded=float_decoded,
        quantized_decoded=quantized_decoded,
        fit_seconds=fit_seconds,
        decode_seconds=decode_seconds,
        float_mse=_mse(float_decoded, target),
        quantized_mse=_mse(quantized_decoded, target),
        bits=estimate_inr_bits(spec, quant),
    )


def _residual_from_features(spec: INRSpec, features: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    params = unflatten_parameters(parameters, make_parameter_layout(spec))
    if spec.family == "fourier":
        return _fourier_matrix(features, spec.order) @ params["readout"]
    if spec.family == "rff":
        matrix = _rff_matrix(features, params["weights"], params["bias"])
        return spec.residual_limit * np.tanh(matrix @ params["readout"])
    if spec.family == "siren":
        matrix = _siren_matrix(features, params["weights"], params["bias"])
        return spec.residual_limit * np.tanh(matrix @ params["readout"])
    if spec.family == "mlp":
        hidden = np.tanh(features.reshape(-1, features.shape[-1]) @ params["weights"] + params["bias"])
        return spec.residual_limit * np.tanh(hidden @ params["readout"] + params["output_bias"][0])
    if spec.family == "model_e_ladder":
        return _ladder_model_e_residual(features, params, spec.residual_limit)
    if spec.family == "model_e_frequency_table":
        return _frequency_table_model_e_residual(features, params, spec.residual_limit)
    if spec.family == "model_e_modulated":
        return _modulated_model_e_residual(features, params, spec.residual_limit)

    residual_scale = float(np.clip(params["residual_scale"][0], 0.0, spec.residual_limit))
    kind = "single_state" if spec.family == "model_e_single" else "coupled_state"
    model_params = ModelEParams(
        kind=kind,
        layers=params["layers"],
        readout=params["readout"],
        residual_scale=residual_scale,
    )
    return model_e_residual(features, model_params).reshape(-1)


def _fit_linear_readout_if_available(
    spec: INRSpec,
    low_guide: np.ndarray,
    reference: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    if spec.family not in {"fourier", "rff", "siren", "mlp"}:
        return parameters
    layout = make_parameter_layout(spec)
    params = unflatten_parameters(parameters, layout)
    features = model_e_features(low_guide, reference.shape)
    base = features[..., 2]
    target = np.clip((reference - base).reshape(-1), -spec.residual_limit, spec.residual_limit)
    if spec.family == "fourier":
        matrix = _fourier_matrix(features, spec.order)
    elif spec.family == "rff":
        matrix = _rff_matrix(features, params["weights"], params["bias"])
    elif spec.family == "siren":
        matrix = _siren_matrix(features, params["weights"], params["bias"])
    else:
        matrix = np.tanh(features.reshape(-1, features.shape[-1]) @ params["weights"] + params["bias"])
    solution = np.linalg.lstsq(matrix, target, rcond=None)[0]
    params["readout"] = solution
    return flatten_parameters(params, layout)


def _fourier_matrix(features: np.ndarray, order: int) -> np.ndarray:
    x = features[..., 0]
    y = features[..., 1]
    columns = [np.ones_like(x)]
    for freq in range(1, order + 1):
        columns.extend(
            [
                np.sin(np.pi * freq * x),
                np.cos(np.pi * freq * x),
                np.sin(np.pi * freq * y),
                np.cos(np.pi * freq * y),
                np.sin(np.pi * freq * (x + y)),
                np.cos(np.pi * freq * (x - y)),
            ]
        )
    return np.stack(columns, axis=-1).reshape(-1, len(columns))


def _rff_matrix(features: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    inputs = features.reshape(-1, features.shape[-1])
    projected = inputs @ weights + bias
    return np.concatenate([np.sin(projected), np.cos(projected)], axis=-1)


def _siren_matrix(features: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    inputs = features.reshape(-1, features.shape[-1])
    return np.sin(6.0 * (inputs @ weights + bias))


def _fixed_ladder_modes(features: np.ndarray) -> np.ndarray:
    x = features[..., 0]
    y = features[..., 1]
    base_modes = [x, y, x + y, x - y]
    columns = []
    for freq in (1.0, 2.0, 4.0, 8.0):
        for mode in base_modes:
            columns.append(np.sin(np.pi * freq * mode))
    return np.stack(columns, axis=-1)


def _fixed_ladder_mode_count() -> int:
    return 16


def _initial_states(shape: tuple[int, int], state_count: int) -> np.ndarray:
    states = np.zeros((*shape, state_count), dtype=np.float64)
    states[..., 0] = 1.0
    if state_count > 1:
        states[..., 1:] = -1.0 / state_count
    return states


def _readout_residual(states: np.ndarray, readout: np.ndarray, residual_scale: float) -> np.ndarray:
    readout_norm = max(float(np.linalg.norm(readout)), 1.0)
    expectation = np.tensordot(states, readout / readout_norm, axes=([-1], [0]))
    return residual_scale * np.tanh(expectation)


def _ladder_model_e_residual(
    features: np.ndarray,
    params: dict[str, np.ndarray],
    residual_limit: float,
) -> np.ndarray:
    modes = _fixed_ladder_modes(features)
    states = _initial_states(features.shape[:2], params["readout"].size)
    for scale, phase in zip(params["scale"], params["phase"]):
        angles = np.tensordot(modes, scale, axes=([-1], [-1])) + phase
        rotated = np.sin(angles + states)
        if states.shape[-1] > 1:
            rotated = _couple_candidate_states(rotated)
        states = np.tanh(rotated)
    residual_scale = float(np.clip(params["residual_scale"][0], 0.0, residual_limit))
    return _readout_residual(states, params["readout"], residual_scale).reshape(-1)


def _frequency_table_model_e_residual(
    features: np.ndarray,
    params: dict[str, np.ndarray],
    residual_limit: float,
) -> np.ndarray:
    inputs = features.reshape(-1, features.shape[-1])
    table_angles = inputs @ params["frequencies"] + params["phase"]
    table_angles = table_angles.reshape(*features.shape[:2], -1)
    states = _initial_states(features.shape[:2], params["readout"].size)
    for mixing in params["mixing"]:
        angles = np.tensordot(table_angles, mixing, axes=([-1], [-1]))
        rotated = np.sin(angles + states)
        if states.shape[-1] > 1:
            rotated = _couple_candidate_states(rotated)
        states = np.tanh(rotated)
    residual_scale = float(np.clip(params["residual_scale"][0], 0.0, residual_limit))
    return _readout_residual(states, params["readout"], residual_scale).reshape(-1)


def _modulated_model_e_residual(
    features: np.ndarray,
    params: dict[str, np.ndarray],
    residual_limit: float,
) -> np.ndarray:
    x = features[..., 0]
    y = features[..., 1]
    coord_modes = np.stack([x, y, x + y, x - y], axis=-1)
    gate_inputs = np.stack(
        [
            np.ones_like(x),
            features[..., 2],
            features[..., 3],
            features[..., 4],
        ],
        axis=-1,
    )
    gate = _sigmoid(np.tensordot(gate_inputs, params["gate"], axes=([-1], [0])))
    states = _initial_states(features.shape[:2], params["readout"].size)
    for coord_frequency, phase in zip(params["coord_frequency"], params["phase"]):
        angles = np.tensordot(coord_modes, coord_frequency, axes=([-1], [-1])) + phase
        rotated = gate[..., None] * np.sin(angles + states)
        if states.shape[-1] > 1:
            rotated = _couple_candidate_states(rotated)
        states = np.tanh(rotated)
    residual_scale = float(np.clip(params["residual_scale"][0], 0.0, residual_limit))
    return (gate * _readout_residual(states, params["readout"], residual_scale)).reshape(-1)


def _couple_candidate_states(states: np.ndarray) -> np.ndarray:
    left = np.roll(states, 1, axis=-1)
    right = np.roll(states, -1, axis=-1)
    return states + 0.25 * left * right


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _mse(candidate: np.ndarray, reference: np.ndarray) -> float:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return float(np.mean(diff * diff))
