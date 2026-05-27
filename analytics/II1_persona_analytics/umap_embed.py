"""
UMAP embedding + trustworthiness for layer activations.

Migrated from metrics/umap.py.

Multiple hyperparameter setups are swept to avoid cherry-picking:
  n_neighbors ∈ {15, 30}, min_dist ∈ {0.0, 0.1}, metric ∈ {'euclidean', 'cosine'}

Trustworthiness (sklearn) measures how well the local neighbourhood structure
is preserved in the 2D embedding.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# import umap
# from sklearn.manifold import trustworthiness
# import matplotlib.pyplot as plt


@dataclass
class UMAPSetup:
    n_neighbors: int
    min_dist: float
    metric: str


DEFAULT_SETUPS = [
    UMAPSetup(15, 0.0, "euclidean"),
    UMAPSetup(15, 0.0, "cosine"),
    UMAPSetup(15, 0.1, "euclidean"),
    UMAPSetup(15, 0.1, "cosine"),
    UMAPSetup(30, 0.0, "euclidean"),
    UMAPSetup(30, 0.0, "cosine"),
    UMAPSetup(30, 0.1, "euclidean"),
    UMAPSetup(30, 0.1, "cosine"),
]


def run_umap_for_layer(
    X: np.ndarray,            # (N, D)
    labels: list[str],
    layer_key: str,
    setups: list[UMAPSetup] = DEFAULT_SETUPS,
    save_dir: str | Path | None = None,
) -> list[dict]:
    """
    Run UMAP with each setup, compute trustworthiness, save scatter plots.

    Returns:
        list of dicts: {setup, coords, trustworthiness_score}
    """
    raise NotImplementedError


def run_umap_pipeline(
    results: list[dict],
    layer_keys: list[str] | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, list[dict]]:
    """
    Run UMAP pipeline for all layers.

    Returns:
        {layer_key: [umap_result per setup]}
    """
    raise NotImplementedError
