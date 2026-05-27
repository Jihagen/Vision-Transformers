"""
Persona-specificity of the interestingness direction.

Answers: do different personas have different interest directions?

Compare per-persona interest vectors via cosine similarity.  High pairwise
similarity → shared global direction.  Low or structured similarity (e.g.
emotion clusters) → persona-specific or feature-mediated interest direction.

Also tests whether the interestingness direction rotates with emotional valence,
cognitive load, or demographic features.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# from analytics.II2_interestingness_analytics.global_direction import split_by_interestingness
# from analytics.II1_persona_analytics.cosine_similarity import cosine_similarity_matrix


def compute_per_persona_interest_vectors(
    all_results: dict[str, list[dict]],   # {persona_key: results_list}
    layer_key: str,
) -> dict[str, np.ndarray]:
    """
    For each persona, compute the mean-difference interestingness vector at layer_key.

    Returns:
        {persona_key: direction_vector}
    """
    raise NotImplementedError


def compare_persona_interest_vectors(
    per_persona_vectors: dict[str, np.ndarray],   # {persona_key: direction_vector}
    save_dir: str | Path | None = None,
) -> np.ndarray:
    """
    Pairwise cosine similarity between per-persona interest directions.

    Returns the similarity matrix (K × K) and saves a heatmap if save_dir given.
    """
    raise NotImplementedError


def correlate_with_persona_features(
    per_persona_vectors: dict[str, np.ndarray],
    feature_table: object,    # DataFrame: persona_key × feature values
) -> dict[str, float]:
    """
    Correlate interest direction similarity with persona feature distances
    (e.g. valence rating, cognitive load level, gender).

    Returns:
        {feature_name: spearman_r}  — RSA-style correlation
    """
    raise NotImplementedError
