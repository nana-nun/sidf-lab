"""Minimal Model E coordinate functions for quantum-inspired residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ModelEKind = Literal["single_state", "coupled_state"]


@dataclass(frozen=True)
class QuantizationSpec:
    """Uniform signed quantization metadata for Model E parameters."""

    bits_per_value: int = 16
    min_value: float = -np.pi
    max_value: float = np.pi
    header_bits: int = 128

    def __post_init__(self) -> None:
        if self.bits_per_value <= 0:
            raise ValueError("bits_per_value must be positive")
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        if self.header_bits < 0:
            raise ValueError("header_bits must be non-negative")


@dataclass(frozen=True)
class ModelEParams:
    """Stored parameters for the minimal Model E decoder."""

    kind: ModelEKind
    layers: np.ndarray
    readout: np.ndarray
    residual_scale: float = 0.1

    def __post_init__(self) -> None:
        layers = np.asarray(self.layers, dtype=np.float64)
        readout = np.asarray(self.readout, dtype=np.float64)
        if self.kind not in {"single_state", "coupled_state"}:
            raise ValueError("kind must be 'single_state' or 'coupled_state'")
        if layers.ndim != 3:
            raise ValueError("layers must have shape (depth, states, features)")
        if layers.shape[0] == 0 or layers.shape[1] == 0 or layers.shape[2] == 0:
            raise ValueError("layers dimensions must be non-empty")
        if readout.shape != (layers.shape[1],):
            raise ValueError("readout length must match the state count")
        if not np.isfinite(layers).all() or not np.isfinite(readout).all():
            raise ValueError("parameters must be finite")
        if not np.isfinite(self.residual_scale) or self.residual_scale < 0.0:
            raise ValueError("residual_scale must be a finite non-negative value")
        if self.kind == "single_state" and layers.shape[1] != 1:
            raise ValueError("single_state parameters must have one state")
        if self.kind == "coupled_state" and layers.shape[1] < 2:
            raise ValueError("coupled_state parameters must have at least two states")

        object.__setattr__(self, "layers", layers.copy())
        object.__setattr__(self, "readout", readout.copy())

    @property
    def feature_count(self) -> int:
        return int(self.layers.shape[2])

    @property
    def state_count(self) -> int:
        return int(self.layers.shape[1])

    @property
    def parameter_count(self) -> int:
        return int(self.layers.size + self.readout.size + 1)


def model_e_features(low_guide: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Return normalized coordinate and guide-derived features.

    Feature order is ``x, y, bilinear guide, gradient_x, gradient_y``. This
    matches the draft Model E design while keeping the initial implementation
    independent of any quantum SDK.
    """
    base = bilinear_resize(low_guide, output_shape)
    grad_y, grad_x = np.gradient(base)
    height, width = output_shape
    xs = _normalized_axis(width)
    ys = _normalized_axis(height)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x, grid_y, base, grad_x, grad_y], axis=-1)


def bilinear_resize(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D grayscale image with deterministic bilinear interpolation."""
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("image must be a 2D array")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("image dimensions must be non-empty")
    height, width = output_shape
    if height <= 0 or width <= 0:
        raise ValueError("output_shape dimensions must be positive")

    src_h, src_w = values.shape
    ys = np.linspace(0.0, src_h - 1.0, height)
    xs = np.linspace(0.0, src_w - 1.0, width)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = ys - y0
    wx = xs - x0

    top = (1.0 - wx)[None, :] * values[y0[:, None], x0[None, :]]
    top += wx[None, :] * values[y0[:, None], x1[None, :]]
    bottom = (1.0 - wx)[None, :] * values[y1[:, None], x0[None, :]]
    bottom += wx[None, :] * values[y1[:, None], x1[None, :]]
    return (1.0 - wy)[:, None] * top + wy[:, None] * bottom


def model_e_residual(features: np.ndarray, params: ModelEParams) -> np.ndarray:
    """Evaluate the bounded residual field for a Model E parameter set."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("features must have shape (height, width, features)")
    if values.shape[-1] != params.feature_count:
        raise ValueError("feature count does not match parameter shape")

    states = np.zeros((*values.shape[:2], params.state_count), dtype=np.float64)
    states[..., 0] = 1.0
    if params.state_count > 1:
        states[..., 1:] = -1.0 / params.state_count

    for layer in params.layers:
        angles = np.tensordot(values, layer, axes=([-1], [-1]))
        rotated = np.sin(angles + states)
        if params.kind == "coupled_state":
            rotated = _couple_states(rotated)
        states = np.tanh(rotated)

    readout_norm = max(float(np.linalg.norm(params.readout)), 1.0)
    expectation = np.tensordot(states, params.readout / readout_norm, axes=([-1], [0]))
    return params.residual_scale * np.tanh(expectation)


def decode_model_e(low_guide: np.ndarray, output_shape: tuple[int, int], params: ModelEParams) -> np.ndarray:
    """Decode a grayscale reconstruction as bilinear guide plus bounded residual."""
    base = bilinear_resize(low_guide, output_shape)
    residual = model_e_residual(model_e_features(low_guide, output_shape), params)
    return np.clip(base + residual, 0.0, 1.0)


def serialize_model_e_params(
    params: ModelEParams,
    quantization: QuantizationSpec | None = None,
) -> dict[str, object]:
    """Serialize Model E parameters with uniform integer quantization."""
    spec = quantization or QuantizationSpec()
    layers_q = _quantize(params.layers, spec)
    readout_q = _quantize(params.readout, spec)
    residual_scale_q = _quantize(np.array([params.residual_scale]), spec)
    return {
        "kind": params.kind,
        "shape": list(params.layers.shape),
        "layers": layers_q.astype(int).tolist(),
        "readout": readout_q.astype(int).tolist(),
        "residual_scale": int(residual_scale_q[0]),
        "quantization": {
            "bits_per_value": spec.bits_per_value,
            "min_value": spec.min_value,
            "max_value": spec.max_value,
            "header_bits": spec.header_bits,
        },
    }


def deserialize_model_e_params(serialized: dict[str, object]) -> ModelEParams:
    """Restore Model E parameters from ``serialize_model_e_params`` output."""
    quant = serialized["quantization"]
    if not isinstance(quant, dict):
        raise ValueError("quantization must be a mapping")
    spec = QuantizationSpec(
        bits_per_value=int(quant["bits_per_value"]),
        min_value=float(quant["min_value"]),
        max_value=float(quant["max_value"]),
        header_bits=int(quant["header_bits"]),
    )
    shape = tuple(int(v) for v in serialized["shape"])  # type: ignore[arg-type]
    layers = _dequantize(np.asarray(serialized["layers"], dtype=np.int64), spec).reshape(shape)
    readout = _dequantize(np.asarray(serialized["readout"], dtype=np.int64), spec)
    residual_scale = float(
        _dequantize(np.asarray([serialized["residual_scale"]], dtype=np.int64), spec)[0]
    )
    return ModelEParams(
        kind=serialized["kind"],  # type: ignore[arg-type]
        layers=layers,
        readout=readout,
        residual_scale=residual_scale,
    )


def estimate_model_e_bits(params: ModelEParams, quantization: QuantizationSpec | None = None) -> dict[str, int]:
    """Return a simple serialized side-information bit estimate."""
    spec = quantization or QuantizationSpec()
    parameter_bits = params.parameter_count * spec.bits_per_value
    structure_bits = 32
    return {
        "model_header_bits": spec.header_bits,
        "structure_bits": structure_bits,
        "quantized_parameter_bits": parameter_bits,
        "incremental_side_bits": spec.header_bits + structure_bits + parameter_bits,
    }


def example_model_e_params(
    kind: ModelEKind = "single_state",
    *,
    depth: int = 3,
    feature_count: int = 5,
) -> ModelEParams:
    """Return deterministic non-fitted parameters for smoke tests and examples."""
    state_count = 1 if kind == "single_state" else 3
    index = np.arange(depth * state_count * feature_count, dtype=np.float64)
    layers = 0.35 * np.sin(index.reshape(depth, state_count, feature_count) + 1.0)
    readout = np.linspace(0.5, 1.0, state_count, dtype=np.float64)
    return ModelEParams(kind=kind, layers=layers, readout=readout, residual_scale=0.08)


def _couple_states(states: np.ndarray) -> np.ndarray:
    left = np.roll(states, 1, axis=-1)
    right = np.roll(states, -1, axis=-1)
    return states + 0.25 * left * right


def _normalized_axis(size: int) -> np.ndarray:
    if size == 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(-1.0, 1.0, size, dtype=np.float64)


def _quantize(values: np.ndarray, spec: QuantizationSpec) -> np.ndarray:
    levels = (1 << spec.bits_per_value) - 1
    normalized = (np.asarray(values, dtype=np.float64) - spec.min_value) / (
        spec.max_value - spec.min_value
    )
    return np.rint(np.clip(normalized, 0.0, 1.0) * levels).astype(np.int64)


def _dequantize(values: np.ndarray, spec: QuantizationSpec) -> np.ndarray:
    levels = (1 << spec.bits_per_value) - 1
    normalized = np.asarray(values, dtype=np.float64) / levels
    return spec.min_value + normalized * (spec.max_value - spec.min_value)
