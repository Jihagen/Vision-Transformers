"""
PCA 2D scatter visualization of layer activations.

Migrated from metrics/metrics.py (PCA section).
Used in analytics to inspect dominant axes and low-dimensional geometry —
not as proof of structure, but as a human-readable complement to GDV.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt


def run_pca(
    X: np.ndarray,        # (N, D)
    n_components: int = 2,
) -> np.ndarray:
    """Fit PCA and return projected coordinates, shape (N, n_components)."""
    raise NotImplementedError


def plot_pca_scatter(
    coords: np.ndarray,   # (N, 2)
    labels: list[str],
    title: str = "",
    save_path: str | Path | None = None,
):
    """Scatter plot of 2D PCA coordinates coloured by label."""
    raise NotImplementedError


def run_pca_pipeline(
    results: list[dict],
    layer_keys: list[str] | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Run PCA for each layer and save scatter plots.

    Returns:
        {layer_key: coords_array}
    """
    raise NotImplementedError
