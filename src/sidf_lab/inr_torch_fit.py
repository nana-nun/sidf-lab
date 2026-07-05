"""Optional PyTorch optimizer adapter for small INR fitting diagnostics."""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sidf_lab.inr_fit import (
    FitResult,
    INRSpec,
    decode_inr,
    estimate_inr_bits,
    fit_inr,
    initialize_parameters,
    make_parameter_layout,
    quantize_vector,
    dequantize_vector,
)
from sidf_lab.model_e import QuantizationSpec, model_e_features


OptimizerBackend = Literal["numpy_random_search", "torch"]
OptimizerMethod = Literal["adam", "lbfgs"]


class TorchBackendUnavailable(RuntimeError):
    """Raised when the optional PyTorch backend is requested but unavailable."""


@dataclass(frozen=True)
class OptimizerSpec:
    """Small optimizer descriptor shared by NumPy and optional torch paths."""

    backend: OptimizerBackend = "numpy_random_search"
    method: OptimizerMethod = "adam"
    steps: int = 32
    learning_rate: float = 0.03
    seed: int = 0

    def __post_init__(self) -> None:
        if self.backend not in {"numpy_random_search", "torch"}:
            raise ValueError("backend must be 'numpy_random_search' or 'torch'")
        if self.method not in {"adam", "lbfgs"}:
            raise ValueError("method must be 'adam' or 'lbfgs'")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if not np.isfinite(self.learning_rate) or self.learning_rate < 0.0:
            raise ValueError("learning_rate must be finite and non-negative")


def torch_available() -> bool:
    """Return whether PyTorch can be imported without importing it eagerly."""

    return importlib.util.find_spec("torch") is not None


def fit_inr_with_optimizer(
    spec: INRSpec,
    low_guide: np.ndarray,
    reference: np.ndarray,
    *,
    optimizer: OptimizerSpec,
    quantization: QuantizationSpec | None = None,
) -> FitResult:
    """Fit an INR with either the existing NumPy path or optional torch autograd."""

    quant = quantization or QuantizationSpec(bits_per_value=12, min_value=-1.0, max_value=1.0)
    if optimizer.backend == "numpy_random_search":
        return fit_inr(
            spec,
            low_guide,
            reference,
            seed=optimizer.seed,
            steps=optimizer.steps,
            initial_step_scale=optimizer.learning_rate,
            quantization=quant,
        )
    return _fit_inr_with_torch(spec, low_guide, reference, optimizer=optimizer, quantization=quant)


def _fit_inr_with_torch(
    spec: INRSpec,
    low_guide: np.ndarray,
    reference: np.ndarray,
    *,
    optimizer: OptimizerSpec,
    quantization: QuantizationSpec,
) -> FitResult:
    torch = _import_torch()
    target = np.asarray(reference, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("reference must be a 2D grayscale image")

    start = time.perf_counter()
    initial = initialize_parameters(spec, optimizer.seed)
    params = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    features = torch.tensor(model_e_features(low_guide, target.shape), dtype=torch.float64)
    target_tensor = torch.tensor(target, dtype=torch.float64)

    trace: list[dict[str, float | int]] = []

    def closure() -> object:
        if params.grad is not None:
            params.grad.zero_()
        decoded = _torch_decode_inr(spec, features, params)
        loss = torch.mean((decoded - target_tensor) ** 2)
        loss.backward()
        return loss

    if optimizer.method == "adam":
        opt = torch.optim.Adam([params], lr=optimizer.learning_rate)
        for step in range(optimizer.steps):
            opt.zero_grad()
            decoded = _torch_decode_inr(spec, features, params)
            loss = torch.mean((decoded - target_tensor) ** 2)
            loss.backward()
            grad_norm = _grad_norm(torch, params)
            opt.step()
            trace.append({"step": step + 1, "loss": float(loss.detach()), "gradient_norm": grad_norm})
    else:
        opt = torch.optim.LBFGS([params], lr=optimizer.learning_rate, max_iter=1, line_search_fn="strong_wolfe")
        for step in range(optimizer.steps):
            loss = opt.step(closure)
            grad_norm = _grad_norm(torch, params)
            trace.append({"step": step + 1, "loss": float(loss.detach()), "gradient_norm": grad_norm})

    fit_seconds = float(time.perf_counter() - start)
    params_np = params.detach().cpu().numpy().astype(np.float64, copy=True)
    quantized = quantize_vector(params_np, quantization)
    restored = dequantize_vector(quantized, quantization)

    decode_start = time.perf_counter()
    float_decoded = decode_inr(spec, low_guide, target.shape, params_np)
    quantized_decoded = decode_inr(spec, low_guide, target.shape, restored)
    decode_seconds = float(time.perf_counter() - decode_start)

    bits = estimate_inr_bits(spec, quantization)
    bits["optimizer"] = {
        "backend": optimizer.backend,
        "method": optimizer.method,
        "steps": optimizer.steps,
        "learning_rate": optimizer.learning_rate,
        "trace": trace,
    }
    return FitResult(
        spec=spec,
        parameters=params_np,
        quantized_parameters=quantized.copy(),
        restored_parameters=restored.copy(),
        float_decoded=float_decoded,
        quantized_decoded=quantized_decoded,
        fit_seconds=fit_seconds,
        decode_seconds=decode_seconds,
        float_mse=_mse(float_decoded, target),
        quantized_mse=_mse(quantized_decoded, target),
        bits=bits,
    )


def _import_torch() -> object:
    if not torch_available():
        raise TorchBackendUnavailable(
            "PyTorch is not installed. Install it only for optional optimizer spike runs; "
            "it is intentionally not a default sidf-lab dependency."
        )
    import torch

    torch.set_num_threads(1)
    return torch


def _torch_decode_inr(spec: INRSpec, features: object, parameters: object) -> object:
    torch = _import_torch()
    residual = _torch_residual_from_features(spec, features, parameters)
    base = features[..., 2]
    return torch.clamp(base + residual.reshape(base.shape), 0.0, 1.0)


def _torch_residual_from_features(spec: INRSpec, features: object, parameters: object) -> object:
    torch = _import_torch()
    params = _torch_unflatten(parameters, spec)
    if spec.family == "fourier":
        return _torch_fourier_matrix(features, spec.order) @ params["readout"]
    if spec.family == "model_e_single" or spec.family == "model_e_coupled":
        residual_scale = torch.clamp(params["residual_scale"][0], 0.0, spec.residual_limit)
        states = torch.zeros((*features.shape[:2], spec.states), dtype=torch.float64)
        states[..., 0] = 1.0
        if spec.states > 1:
            states[..., 1:] = -1.0 / spec.states
        for layer in params["layers"]:
            angles = torch.tensordot(features, layer, dims=([-1], [-1]))
            rotated = torch.sin(angles + states)
            if spec.family == "model_e_coupled":
                rotated = _torch_couple_states(rotated)
            states = torch.tanh(rotated)
        readout_norm = torch.maximum(torch.linalg.vector_norm(params["readout"]), torch.tensor(1.0, dtype=torch.float64))
        expectation = torch.tensordot(states, params["readout"] / readout_norm, dims=([-1], [0]))
        return residual_scale * torch.tanh(expectation).reshape(-1)
    raise ValueError(f"torch optimizer spike supports fourier and current Model E specs, got {spec.family}")


def _torch_unflatten(parameters: object, spec: INRSpec) -> dict[str, object]:
    torch = _import_torch()
    layout = make_parameter_layout(spec)
    if parameters.numel() != layout.parameter_count:
        raise ValueError(f"vector has {parameters.numel()} values, expected {layout.parameter_count}")
    out: dict[str, object] = {}
    start = 0
    for name, shape in zip(layout.names, layout.shapes):
        size = int(np.prod(shape, dtype=np.int64))
        out[name] = parameters[start : start + size].reshape(shape)
        start += size
    return out


def _torch_fourier_matrix(features: object, order: int) -> object:
    torch = _import_torch()
    x = features[..., 0]
    y = features[..., 1]
    columns = [torch.ones_like(x)]
    for freq in range(1, order + 1):
        columns.extend(
            [
                torch.sin(torch.pi * freq * x),
                torch.cos(torch.pi * freq * x),
                torch.sin(torch.pi * freq * y),
                torch.cos(torch.pi * freq * y),
                torch.sin(torch.pi * freq * (x + y)),
                torch.cos(torch.pi * freq * (x - y)),
            ]
        )
    return torch.stack(columns, dim=-1).reshape(-1, len(columns))


def _torch_couple_states(states: object) -> object:
    left = states.roll(1, dims=-1)
    right = states.roll(-1, dims=-1)
    return states + 0.25 * left * right


def _grad_norm(torch: object, params: object) -> float:
    if params.grad is None:
        return float("nan")
    return float(torch.linalg.vector_norm(params.grad.detach()).cpu())


def _mse(candidate: np.ndarray, reference: np.ndarray) -> float:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return float(np.mean(diff * diff))
