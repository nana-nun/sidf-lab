"""I/O helpers for experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    """Save JSON with stable formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_grayscale_png(path: str | Path, image: np.ndarray) -> None:
    """Save a [0, 1] grayscale image as PNG using matplotlib."""
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(target, np.clip(image, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)

