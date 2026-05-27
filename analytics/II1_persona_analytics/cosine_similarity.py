"""
Pairwise cosine similarity between persona-feature vectors.

Answers: do countries / jobs / emotions form meaningful distance structures?
Is gender represented as a binary contrast or a gradient?

Input: a set of concept direction vectors (from I.2 or I.3) for multiple conditions.
Output: similarity matrix + visualization.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# import matplotlib.pyplot as plt
# import seaborn as sns


def cosine_similarity_matrix(
    vectors: dict[str, np.ndarray],   # {condition_name: unit_vector}
) -> tuple[np.ndarray, list[str]]:
    """
    Compute pairwise cosine similarities between all vectors.

    Returns:
        (sim_matrix, ordered_labels)   sim_matrix shape (K, K)
    """
    raise NotImplementedError


def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    labels: list[str],
    title: str = "",
    save_path: str | Path | None = None,
):
    """Plot annotated heatmap of the cosine similarity matrix."""
    raise NotImplementedError


def compare_persona_vectors(
    md_vectors: dict[str, dict[str, np.ndarray]],   # {contrast: {layer: vector}}
    target_layer: str,
    save_dir: str | Path | None = None,
) -> np.ndarray:
    """
    Compare mean-difference vectors for all contrasts at a given layer.

    Returns cosine similarity matrix over contrasts.
    """
    raise NotImplementedError
