"""
Distance matrix analysis for persona geometry.

Inspects how conditions (personas, emotions, countries, jobs) are arranged
in activation space — analogous to Representational Similarity Analysis (RSA).

Answers: do semantically related personas cluster together in activation space?

Ref: RSA (Frontiers in Systems Neuroscience 2008), CKA (arxiv 1905.00414)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.spatial.distance import cdist


def mean_activation(
    results: list[dict],
    layer_key: str,
    label_field: str = "interestingness",
) -> dict[str, np.ndarray]:
    """
    Compute the mean activation vector per unique label for a given layer.

    Returns:
        {label: mean_vector}
    """
    raise NotImplementedError


def compute_distance_matrix(
    condition_vectors: dict[str, np.ndarray],   # {condition_name: vector}
    metric: str = "cosine",
) -> tuple[np.ndarray, list[str]]:
    """
    Pairwise distance matrix between condition mean vectors.

    Returns:
        (dist_matrix, ordered_labels)
    """
    raise NotImplementedError


def plot_distance_heatmap(
    dist_matrix: np.ndarray,
    labels: list[str],
    title: str = "",
    save_path: str | Path | None = None,
):
    raise NotImplementedError


def rsa_correlation(
    model_rdm: np.ndarray,    # representational dissimilarity matrix from activations
    target_rdm: np.ndarray,   # theoretical / behavioural RDM
) -> float:
    """
    Spearman correlation between model RDM and a target RDM (RSA score).

    Higher = model geometry matches the theoretical structure.
    """
    raise NotImplementedError
