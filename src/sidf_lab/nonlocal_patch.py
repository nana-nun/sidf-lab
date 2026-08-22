"""Low-guide-only non-local self-similarity helpers for Model D candidates.

This module implements deterministic, dependency-free (NumPy only) helpers for
two low-guide-only candidates studied in Issue #130:

- ``self_guided_non_local_means``: a self-guided Non-local Means smoother that
  computes patch similarity from the bilinear-upscaled low guide only.
- ``nonlocal_patch_graph_decode``: a decoder that builds a non-local patch
  graph from low-guide-derived patch descriptors and solves a quadratic
  objective (data fidelity + local edge-aware smoothing + non-local
  patch-similarity smoothing) with a deterministic Jacobi solver.

Both helpers only ever use the low guide (or its deterministic upscale) to
compute patch similarity. Ground Truth or independent high-resolution guidance
must not be passed as the similarity source; keeping that separation is the
point of the low-guide-only condition in Issue #130.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def box_mean(image: np.ndarray, radius: int) -> np.ndarray:
    """Return a reflect-padded square-window mean using an integral image."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    values = np.asarray(image, dtype=np.float64)
    if radius == 0:
        return values.copy()
    size = 2 * radius + 1
    padded = np.pad(values, radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    window_sum = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return window_sum / float(size * size)


def _shifted(padded: np.ndarray, pad: int, dy: int, dx: int, height: int, width: int) -> np.ndarray:
    """Return the (height, width) window of ``padded`` shifted by (dy, dx)."""
    return padded[pad + dy : pad + dy + height, pad + dx : pad + dx + width]


def _patch_distance_map(
    guide: np.ndarray,
    padded_guide: np.ndarray,
    pad: int,
    dy: int,
    dx: int,
    patch_radius: int,
) -> np.ndarray:
    """Mean squared patch difference between the guide and an offset copy."""
    height, width = guide.shape
    shifted = _shifted(padded_guide, pad, dy, dx, height, width)
    return box_mean((guide - shifted) ** 2, patch_radius)


def self_guided_non_local_means(
    guidance: np.ndarray,
    source: np.ndarray | None = None,
    *,
    patch_radius: int = 1,
    search_radius: int = 5,
    h: float = 0.08,
) -> np.ndarray:
    """Apply deterministic self-guided Non-local Means.

    ``guidance`` is the low-guide-derived image used both to compute patch
    similarity and, by default, as the values being averaged (self-guided). An
    explicit ``source`` may be supplied when the averaged values differ from the
    similarity source, but it must still be derived from the low guide only.
    """
    if patch_radius < 0:
        raise ValueError("patch_radius must be non-negative")
    if search_radius < 1:
        raise ValueError("search_radius must be positive")
    if h <= 0.0:
        raise ValueError("h must be positive")

    guide = np.asarray(guidance, dtype=np.float64)
    if guide.ndim != 2:
        raise ValueError("guidance must be a 2D array")
    values = guide if source is None else np.asarray(source, dtype=np.float64)
    if values.shape != guide.shape:
        raise ValueError("source and guidance must have the same shape")

    height, width = guide.shape
    padded_guide = np.pad(guide, search_radius, mode="reflect")
    padded_values = np.pad(values, search_radius, mode="reflect")
    accum = np.zeros_like(guide)
    weight_sum = np.zeros_like(guide)
    denominator = h * h

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            patch_distance = _patch_distance_map(
                guide, padded_guide, search_radius, dy, dx, patch_radius
            )
            weight = np.exp(-patch_distance / denominator)
            shifted_values = _shifted(padded_values, search_radius, dy, dx, height, width)
            accum += weight * shifted_values
            weight_sum += weight
    return np.clip(accum / weight_sum, 0.0, 1.0)


@dataclass(frozen=True)
class PatchGraph:
    """A symmetric weighted edge list over flattened pixel indices."""

    rows: np.ndarray
    cols: np.ndarray
    weights: np.ndarray
    num_nodes: int
    nonlocal_degree: np.ndarray
    selected_distances: np.ndarray

    def statistics(self) -> dict[str, float]:
        """Return summary statistics for the non-local edges."""
        distances = np.asarray(self.selected_distances, dtype=np.float64)
        degree = np.asarray(self.nonlocal_degree, dtype=np.float64)
        if distances.size == 0:
            return {
                "mean_nonlocal_degree": 0.0,
                "max_nonlocal_degree": 0.0,
                "mean_patch_distance": 0.0,
                "min_patch_distance": 0.0,
                "max_patch_distance": 0.0,
                "median_patch_distance": 0.0,
                "nonlocal_edge_count": 0.0,
            }
        return {
            "mean_nonlocal_degree": float(degree.mean()),
            "max_nonlocal_degree": float(degree.max()),
            "mean_patch_distance": float(distances.mean()),
            "min_patch_distance": float(distances.min()),
            "max_patch_distance": float(distances.max()),
            "median_patch_distance": float(np.median(distances)),
            "nonlocal_edge_count": float(distances.size),
        }


def build_nonlocal_patch_graph(
    guide: np.ndarray,
    *,
    patch_radius: int = 1,
    search_radius: int = 7,
    num_neighbors: int = 5,
    local_exclude_radius: int = 1,
    h: float = 0.08,
    j_nonlocal: float = 1.0,
) -> PatchGraph:
    """Build a symmetric non-local patch graph from a low-guide-derived image.

    For each pixel, the ``num_neighbors`` most similar patches inside the search
    window (excluding the local Chebyshev neighborhood) are connected with a
    weight ``j_nonlocal * exp(-patch_distance / h**2)``. Edges are symmetrized so
    the resulting quadratic objective is well defined.
    """
    if num_neighbors < 1:
        raise ValueError("num_neighbors must be positive")
    if j_nonlocal <= 0.0:
        raise ValueError("j_nonlocal must be positive")

    guide = np.asarray(guide, dtype=np.float64)
    if guide.ndim != 2:
        raise ValueError("guide must be a 2D array")
    height, width = guide.shape
    num_nodes = height * width
    padded_guide = np.pad(guide, search_radius, mode="reflect")

    offsets: list[tuple[int, int]] = []
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if max(abs(dy), abs(dx)) <= local_exclude_radius:
                continue
            offsets.append((dy, dx))
    if not offsets:
        raise ValueError("search_radius must exceed local_exclude_radius")

    distance_stack = np.empty((len(offsets), height, width), dtype=np.float64)
    for index, (dy, dx) in enumerate(offsets):
        distance_stack[index] = _patch_distance_map(
            guide, padded_guide, search_radius, dy, dx, patch_radius
        )

    k = min(num_neighbors, len(offsets))
    selection = np.argpartition(distance_stack, kth=k - 1, axis=0)[:k]  # (k, H, W)

    row_indices = np.arange(height)[:, None]
    col_indices = np.arange(width)[None, :]
    base_index = (row_indices * width + col_indices).astype(np.int64)

    offsets_y = np.array([dy for dy, _ in offsets], dtype=np.int64)
    offsets_x = np.array([dx for _, dx in offsets], dtype=np.int64)

    edge_rows: list[np.ndarray] = []
    edge_cols: list[np.ndarray] = []
    edge_weights: list[np.ndarray] = []
    selected_distances: list[np.ndarray] = []

    for slot in range(k):
        offset_index = selection[slot]  # (H, W)
        dy = offsets_y[offset_index]
        dx = offsets_x[offset_index]
        neighbor_y = np.clip(row_indices + dy, 0, height - 1)
        neighbor_x = np.clip(col_indices + dx, 0, width - 1)
        neighbor_index = neighbor_y * width + neighbor_x
        distance = np.take_along_axis(distance_stack, offset_index[None], axis=0)[0]

        source_flat = base_index.reshape(-1)
        neighbor_flat = neighbor_index.reshape(-1)
        distance_flat = distance.reshape(-1)
        # Keep only edges whose clipped neighbor is genuinely non-local. Border
        # clipping can otherwise map a non-local offset onto a nearby pixel.
        chebyshev = np.maximum(
            np.abs(neighbor_y - row_indices), np.abs(neighbor_x - col_indices)
        ).reshape(-1)
        valid = (source_flat != neighbor_flat) & (chebyshev > local_exclude_radius)
        source_flat = source_flat[valid]
        neighbor_flat = neighbor_flat[valid]
        distance_flat = distance_flat[valid]
        weight_flat = j_nonlocal * np.exp(-distance_flat / (h * h))

        # Symmetrize: add both directions so accumulation stays symmetric.
        edge_rows.append(source_flat)
        edge_cols.append(neighbor_flat)
        edge_weights.append(weight_flat)
        edge_rows.append(neighbor_flat)
        edge_cols.append(source_flat)
        edge_weights.append(weight_flat)
        selected_distances.append(distance_flat)

    rows = np.concatenate(edge_rows)
    cols = np.concatenate(edge_cols)
    weights = np.concatenate(edge_weights)
    distances = np.concatenate(selected_distances)

    nonlocal_degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(nonlocal_degree, rows, 1.0)

    return PatchGraph(
        rows=rows,
        cols=cols,
        weights=weights,
        num_nodes=num_nodes,
        nonlocal_degree=nonlocal_degree,
        selected_distances=distances,
    )


def local_edge_list(
    guide: np.ndarray,
    *,
    j_base: float = 1.8,
    gamma: float = 35.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return symmetric edge-aware 4-neighbor edges over a guide image.

    Weights follow the Model C/D form ``j_base * exp(-gamma * (g_i - g_j)**2)``,
    weakening the coupling across large guide differences.
    """
    guide = np.asarray(guide, dtype=np.float64)
    if guide.ndim != 2:
        raise ValueError("guide must be a 2D array")
    height, width = guide.shape
    row_indices = np.arange(height)[:, None]
    col_indices = np.arange(width)[None, :]
    base_index = (row_indices * width + col_indices).astype(np.int64)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        y0 = max(0, -dy)
        y1 = min(height, height - dy)
        x0 = max(0, -dx)
        x1 = min(width, width - dx)
        source = base_index[y0:y1, x0:x1]
        neighbor = base_index[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        guide_source = guide[y0:y1, x0:x1]
        guide_neighbor = guide[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        weight = j_base * np.exp(-gamma * (guide_source - guide_neighbor) ** 2)
        rows.append(source.reshape(-1))
        cols.append(neighbor.reshape(-1))
        weights.append(weight.reshape(-1))
    return (
        np.concatenate(rows),
        np.concatenate(cols),
        np.concatenate(weights),
    )


def quadratic_objective(
    values: np.ndarray,
    guide_flat: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    lambda_data: float,
) -> float:
    """Evaluate the quadratic graph objective for flattened values.

    The symmetric edge list double-counts each undirected edge, so the pairwise
    term is halved to report the undirected objective value.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    data_term = lambda_data * float(np.sum((values - guide_flat) ** 2))
    pairwise = float(np.sum(weights * (values[rows] - values[cols]) ** 2))
    return data_term + 0.5 * pairwise


def solve_quadratic_graph(
    guide: np.ndarray,
    initial: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    *,
    lambda_data: float,
    max_sweeps: int = 80,
    tol: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float]]:
    """Minimize a quadratic graph objective with deterministic Jacobi sweeps.

    Objective:
        sum_i lambda_data * (v_i - guide_i)**2
        + sum_edges weight_ij * (v_i - v_j)**2

    The edge list is assumed symmetric. Values are clamped to [0, 1] each sweep.
    """
    if lambda_data <= 0.0:
        raise ValueError("lambda_data must be positive")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be positive")

    guide = np.asarray(guide, dtype=np.float64)
    num_nodes = guide.size
    guide_flat = guide.reshape(-1)
    values = np.asarray(initial, dtype=np.float64).reshape(-1).copy()

    degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(degree, rows, weights)
    denominator = lambda_data + degree

    sweeps_run = 0
    final_delta = 0.0
    for _ in range(max_sweeps):
        contribution = np.zeros(num_nodes, dtype=np.float64)
        np.add.at(contribution, rows, weights * values[cols])
        updated = (lambda_data * guide_flat + contribution) / denominator
        updated = np.clip(updated, 0.0, 1.0)
        final_delta = float(np.max(np.abs(updated - values)))
        values = updated
        sweeps_run += 1
        if final_delta < tol:
            break

    objective = quadratic_objective(values, guide_flat, rows, cols, weights, lambda_data)
    diagnostics = {
        "sweeps_run": float(sweeps_run),
        "final_max_delta": final_delta,
        "final_objective": objective,
    }
    return values.reshape(guide.shape), diagnostics


def nonlocal_patch_graph_decode(
    guide: np.ndarray,
    *,
    lambda_data: float = 6.0,
    j_base: float = 1.8,
    gamma: float = 35.0,
    j_nonlocal: float = 1.0,
    patch_radius: int = 1,
    search_radius: int = 7,
    num_neighbors: int = 5,
    local_exclude_radius: int = 1,
    h: float = 0.08,
    max_sweeps: int = 80,
    tol: float = 1e-6,
) -> tuple[np.ndarray, dict[str, object]]:
    """Decode a high-resolution image from a low-guide-derived guide image.

    ``guide`` must be the bilinear-upscaled low guide (or another deterministic
    low-guide upscale). Patch similarity and all graph structure derive only
    from this low-guide-derived image, never from Ground Truth or independent
    high-resolution guidance.
    """
    guide = np.asarray(guide, dtype=np.float64)
    if guide.ndim != 2:
        raise ValueError("guide must be a 2D array")

    graph = build_nonlocal_patch_graph(
        guide,
        patch_radius=patch_radius,
        search_radius=search_radius,
        num_neighbors=num_neighbors,
        local_exclude_radius=local_exclude_radius,
        h=h,
        j_nonlocal=j_nonlocal,
    )
    local_rows, local_cols, local_weights = local_edge_list(guide, j_base=j_base, gamma=gamma)

    rows = np.concatenate([local_rows, graph.rows])
    cols = np.concatenate([local_cols, graph.cols])
    weights = np.concatenate([local_weights, graph.weights])

    rendered, diagnostics = solve_quadratic_graph(
        guide,
        guide,
        rows,
        cols,
        weights,
        lambda_data=lambda_data,
        max_sweeps=max_sweeps,
        tol=tol,
    )
    info: dict[str, object] = {
        "graph_statistics": graph.statistics(),
        "solver": diagnostics,
        "local_edge_count": float(local_weights.size),
    }
    return rendered, info
