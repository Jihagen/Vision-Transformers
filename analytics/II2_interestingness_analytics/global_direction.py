"""
Global interestingness direction.

Is there a single direction v_interest such that projecting activations
from any persona onto v_interest separates high-interest from low-interest images?

Approach:
  1. Pool activations across all persona conditions (or use baseline condition).
  2. Compute mean-difference vector: mean(H_high) − mean(H_low) per layer.
  3. Evaluate whether this direction generalises across personas via cross-persona
     projection AUROC (train on one persona, test on another).

If AUROC generalises well → global direction exists.
If AUROC drops to chance on held-out personas → direction is persona-specific.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# from representation.I2_mean_difference.mean_difference import compute_contrast_vectors
# from representation.I4_vector_evaluation.evaluate import projection_separation


# Interestingness label groupings used for high/low contrast
HIGH_INTEREST_LABELS = {"Very Interesting", "Extremely Interesting"}
LOW_INTEREST_LABELS  = {"Not Interesting", "Slightly Interesting"}


def split_by_interestingness(
    results: list[dict],
    high_labels: set[str] = HIGH_INTEREST_LABELS,
    low_labels: set[str] = LOW_INTEREST_LABELS,
    layer_key: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split activation matrix into high-interest and low-interest subsets.

    Returns:
        (H_high, H_low) — activation arrays for one layer
    """
    raise NotImplementedError


def find_global_interest_direction(
    all_results: dict[str, list[dict]],   # {persona_key: results_list}
    layer_keys: list[str] | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute a pooled interestingness direction across all persona conditions.

    Returns:
        {layer_key: direction_vector} — the global interest direction per layer
    """
    raise NotImplementedError


def cross_persona_generalisation(
    interest_directions: dict[str, dict[str, np.ndarray]],
    # {persona_key: {layer_key: direction_vector}}
    all_results: dict[str, list[dict]],
    target_layer: str,
) -> dict[str, dict[str, float]]:
    """
    Evaluate how well each persona's interest direction generalises to other personas.

    For each (source_persona, target_persona) pair, project target activations
    onto source direction and compute AUROC.

    Returns:
        {source_persona: {target_persona: auroc}}
    """
    raise NotImplementedError
