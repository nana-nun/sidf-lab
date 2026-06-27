"""Helpers for small grayscale patch fixtures used by experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MANIFEST = Path("experiments/assets/source_split_grayscale/manifest.json")


def load_patch_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and validate a grayscale patch fixture manifest."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_patch_manifest(manifest)
    return manifest


def validate_patch_manifest(manifest: dict[str, Any]) -> None:
    """Validate the manifest structure needed by source-split experiments."""
    sources = manifest.get("sources")
    patches = manifest.get("patches")
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest must contain a non-empty sources list")
    if not isinstance(patches, list) or not patches:
        raise ValueError("manifest must contain a non-empty patches list")

    source_splits: dict[str, str] = {}
    for source in sources:
        source_id = _required_str(source, "source_id")
        split = _required_str(source, "split")
        if split not in {"development", "evaluation"}:
            raise ValueError(f"unsupported split for source {source_id}: {split}")
        if source_id in source_splits:
            raise ValueError(f"duplicate source_id: {source_id}")
        source_splits[source_id] = split
        _required_str(source, "license")
        _required_str(source, "source_page")

    seen_patch_names: set[str] = set()
    for patch in patches:
        name = _required_str(patch, "name")
        source_id = _required_str(patch, "source_id")
        split = _required_str(patch, "split")
        if name in seen_patch_names:
            raise ValueError(f"duplicate patch name: {name}")
        seen_patch_names.add(name)
        if source_id not in source_splits:
            raise ValueError(f"patch {name} references unknown source_id: {source_id}")
        if split != source_splits[source_id]:
            raise ValueError(f"patch {name} split does not match source {source_id}")
        if split not in {"development", "evaluation"}:
            raise ValueError(f"unsupported split for patch {name}: {split}")
        _required_str(patch, "npy_path")
        _required_str(patch, "png_path")
        shape = patch.get("shape")
        if shape != [128, 128]:
            raise ValueError(f"patch {name} must be a 128x128 grayscale fixture")


def list_patch_records(split: str | None = None, path: str | Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    """Return patch records, optionally filtered by development/evaluation split."""
    manifest = load_patch_manifest(path)
    patches = list(manifest["patches"])
    if split is None:
        return patches
    if split not in {"development", "evaluation"}:
        raise ValueError(f"unsupported split: {split}")
    return [patch for patch in patches if patch["split"] == split]


def load_grayscale_patch(name: str, path: str | Path = DEFAULT_MANIFEST) -> np.ndarray:
    """Load a named grayscale patch as float64 values in [0, 1]."""
    manifest_path = Path(path)
    records = list_patch_records(path=manifest_path)
    matches = [record for record in records if record["name"] == name]
    if not matches:
        raise KeyError(f"unknown patch fixture: {name}")
    patch_path = _resolve_manifest_path(manifest_path, matches[0]["npy_path"])
    image = np.load(patch_path).astype(np.float64)
    if image.shape != (128, 128):
        raise ValueError(f"patch {name} has shape {image.shape}, expected (128, 128)")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError(f"patch {name} contains values outside [0, 1]")
    return image


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing string field: {key}")
    return value
