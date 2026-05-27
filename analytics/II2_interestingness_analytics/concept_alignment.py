"""
Interestingness–concept alignment.

Does the interestingness direction align with other known concept directions?
Candidates: emotion (valence), novelty, aesthetics, topic/style.

Method:
  - Extract concept vectors for emotion contrasts (from I.2 persona vectors).
  - Compute cosine similarity between interest direction and each concept direction.
  - High alignment with a concept → that concept partially explains interestingness.
  - Near-zero alignment with all concepts → interestingness is an independent axis.

If interestingness looks like a compound space (from I.5), inspect how each
subspace basis vector aligns with the named concept directions.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# from analytics.II1_persona_analytics.cosine_similarity import cosine_similarity_matrix


# Named concept categories to probe against the interest direction
CONCEPT_GROUPS = {
    "valence": ["female_excitement_vs_female_anger", "female_amusement_vs_female_fear"],
    "arousal": ["female_excitement_vs_female_contentment", "female_anger_vs_female_sadness"],
    "cognitive_load": ["high_load_vs_low_load"],
}


def align_with_concepts(
    interest_direction: np.ndarray,                # (D,)
    concept_vectors: dict[str, np.ndarray],        # {contrast_name: vector}
    concept_groups: dict[str, list[str]] = CONCEPT_GROUPS,
) -> dict[str, float]:
    """
    Compute cosine similarity between the interest direction and each concept vector.

    Returns:
        {contrast_name: cosine_similarity}
    """
    raise NotImplementedError


def plot_concept_alignment(
    alignment_scores: dict[str, float],
    concept_groups: dict[str, list[str]] = CONCEPT_GROUPS,
    title: str = "",
    save_path: str | Path | None = None,
):
    """Bar chart of alignment scores grouped by concept category."""
    raise NotImplementedError


def run_concept_alignment_pipeline(
    interest_vectors_by_layer: dict[str, np.ndarray],
    md_vectors: dict[str, dict[str, np.ndarray]],   # {contrast: {layer: vector}}
    target_layer: str,
    save_dir: str | Path | None = None,
) -> dict[str, float]:
    """
    Full alignment pipeline for a single target layer.

    Returns alignment_scores dict.
    """
    raise NotImplementedError
