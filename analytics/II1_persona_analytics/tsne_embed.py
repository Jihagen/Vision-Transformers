"""
t-SNE embedding + trustworthiness for layer activations.

Migrated from metrics/tsne.py.

Multiple hyperparameter setups are swept:
  perplexity ∈ {10, 30, 50}, learning_rate ∈ {100, 200, 'auto'},
  metric ∈ {'euclidean', 'cosine'}

Trustworthiness (sklearn) is the primary quantitative output alongside GDV.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# from sklearn.manifold import TSNE, trustworthiness
# import matplotlib.pyplot as plt


@dataclass
class TSNESetup:
    perplexity: int
    learning_rate: int | str
    metric: str


DEFAULT_SETUPS = [
    TSNESetup(10, 100, "euclidean"),
    TSNESetup(10, 100, "cosine"),
    TSNESetup(10, 200, "euclidean"),
    TSNESetup(10, "auto", "euclidean"),
    TSNESetup(30, 100, "euclidean"),
    TSNESetup(30, 100, "cosine"),
    TSNESetup(30, 200, "euclidean"),
    TSNESetup(30, "auto", "euclidean"),
    TSNESetup(50, 100, "euclidean"),
    TSNESetup(50, 100, "cosine"),
    TSNESetup(50, 200, "euclidean"),
    TSNESetup(50, "auto", "euclidean"),
]


def run_tsne_for_layer(
    X: np.ndarray,
    labels: list[str],
    layer_key: str,
    setups: list[TSNESetup] = DEFAULT_SETUPS,
    save_dir: str | Path | None = None,
) -> list[dict]:
    """
    Run t-SNE with each setup, compute trustworthiness, save scatter plots.

    Returns:
        list of dicts: {setup, coords, trustworthiness_score}
    """
    raise NotImplementedError


def run_tsne_pipeline(
    results: list[dict],
    layer_keys: list[str] | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, list[dict]]:
    """
    Run t-SNE pipeline for all layers.

    Returns:
        {layer_key: [tsne_result per setup]}
    """
    raise NotImplementedError
