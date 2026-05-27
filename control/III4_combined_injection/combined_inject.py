"""
Combined feature-vector injection: test compositional control.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# from control.III1_vector_addition.add_vector import inject_vector


def combine_vectors(
    vectors: dict[str, np.ndarray],    # {concept_name: direction_vector}
    weights: dict[str, float] | None = None,   # optional per-concept alpha
    normalize: bool = False,
) -> np.ndarray:
    """
    Sum multiple concept vectors (optionally weighted) into a single injection vector.

    v_combined = sum(w_i * v_i)

    Args:
        normalize: if True, L2-normalize the combined vector before returning.
    """
    raise NotImplementedError


def inject_combined_vectors(
    model,
    processor,
    image,
    prompt: str,
    vectors: dict[str, np.ndarray],    # {concept_name: direction_vector} for one layer
    layer_key: str,
    weights: dict[str, float] | None = None,
    state=None,
) -> dict:
    """
    Inject the sum of multiple concept vectors at one layer and return model output.
    """
    raise NotImplementedError


def compare_with_compound_persona(
    combined_results: list[dict],      # outputs from inject_combined_vectors
    compound_results: list[dict],      # outputs collected from compound-persona condition (I.1)
    metric: str = "label_agreement",   # "label_agreement" | "ordinal_distance"
) -> dict:
    """
    Quantify how well the combined intervention reproduces the compound-persona output.

    Metrics:
      label_agreement   — fraction of images where combined matches compound label
      ordinal_distance  — mean absolute ordinal distance between label sets

    Returns summary dict with the chosen metric and supporting statistics.
    """
    raise NotImplementedError


def run_combined_injection_experiment(
    model,
    processor,
    images: list[str | Path],
    prompt: str,
    feature_vectors: dict[str, np.ndarray],   # {concept: vector} at target layer
    layer_key: str,
    compound_results: list[dict],
    combination_sets: list[list[str]] | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, dict]:
    """
    Try different combinations of feature vectors and compare to compound persona.

    Args:
        combination_sets: list of subsets of feature_vectors to try
                          (None = try all features combined)

    Returns:
        {combination_label: comparison_summary_dict}
    """
    raise NotImplementedError
