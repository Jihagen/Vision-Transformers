"""
Vector evaluation: decide which candidate directions are worth keeping.

Each evaluation function is stateless — pass in the vectors and activations,
get back a score or boolean decision.  The summary function collects all checks
and returns a single VectorReport per (contrast, layer).
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

# from representation.I2_mean_difference.mean_difference import load_vectors
# from representation.I3_linear_probe.probe import ProbeResult


@dataclass
class VectorReport:
    contrast: str
    layer_key: str
    projection_auc: float           # AUROC when separating pos/neg via scalar projection
    md_cav_cosine: float            # cosine similarity between mean-diff and CAV vectors
    probe_accuracy_cv: float        # from I.3
    probe_above_chance: bool        # probe_accuracy_cv > chance_threshold
    separation_significant: bool    # projection_auc > auc_threshold
    keep: bool                      # overall recommendation
    notes: str = ""
    extra: dict = field(default_factory=dict)


def projection_separation(
    H_pos: np.ndarray,    # (N_pos, D)
    H_neg: np.ndarray,    # (N_neg, D)
    vector: np.ndarray,   # (D,) candidate direction
) -> float:
    """
    Project all activations onto vector and measure how well the scalar
    projection separates positive from negative examples.

    Returns:
        AUROC score (0.5 = chance, 1.0 = perfect separation)
    """
    raise NotImplementedError


def alignment_score(
    mean_diff_vector: np.ndarray,    # (D,) from I.2
    cav_vector: np.ndarray,          # (D,) from I.3
) -> float:
    """
    Cosine similarity between the mean-difference and CAV vectors.

    High alignment (> ~0.7) indicates a clean, approximately linear concept.
    Low alignment suggests the concept may be compound or the probe is unstable.
    """
    raise NotImplementedError


def layer_profile(
    vectors_by_layer: dict[str, np.ndarray],  # {layer_key: vector}
    H_pos_by_layer: dict[str, np.ndarray],
    H_neg_by_layer: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Compute projection_separation for each layer.

    Returns:
        {layer_key: auroc} — useful for locating where signal is strongest.
    """
    raise NotImplementedError


def evaluate_vector(
    contrast: str,
    layer_key: str,
    H_pos: np.ndarray,
    H_neg: np.ndarray,
    mean_diff_vector: np.ndarray,
    cav_vector: np.ndarray,
    probe_accuracy_cv: float,
    chance_threshold: float = 0.60,
    auc_threshold: float = 0.65,
) -> VectorReport:
    """
    Run all checks and return a VectorReport with a keep/drop recommendation.
    """
    raise NotImplementedError


def evaluate_all_vectors(
    loaded_data: dict[str, dict],
    md_vectors: dict[str, dict[str, np.ndarray]],    # {contrast: {layer: vector}}
    cav_vectors: dict[str, dict[str, np.ndarray]],   # {contrast: {layer: vector}}
    probe_results: dict[str, list],
    save_dir=None,
) -> dict[str, list[VectorReport]]:
    """
    Evaluate all contrasts × layers and return {contrast: [VectorReport]}.

    Optionally saves a summary CSV to save_dir.
    """
    raise NotImplementedError
