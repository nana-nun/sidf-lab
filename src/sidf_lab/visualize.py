"""Visualization helpers for experiment comparison figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_image_grid(path: str | Path, images: list[tuple[str, np.ndarray]]) -> None:
    """Save a one-row comparison grid."""
    import matplotlib.pyplot as plt

    if not images:
        raise ValueError("images must not be empty")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]
    for ax, (title, image) in zip(axes, images):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)

