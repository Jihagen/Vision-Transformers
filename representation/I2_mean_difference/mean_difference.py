"""
Mean-difference vector computation.

For a given contrast (positive condition vs negative condition) and a given layer,
compute v = mean(H_pos) - mean(H_neg) and optionally L2-normalize.

Vectors are computed per layer and saved as a dict so downstream steps can
select the most informative layer(s) after I.4 evaluation.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# from representation.I1_contrast_design.load import stack_activations


ContrastDef = tuple[str, str]  # (positive_persona_key, negative_persona_key)


def compute_mean_difference_vector(
    H_pos: np.ndarray,   # shape (N_pos, D)
    H_neg: np.ndarray,   # shape (N_neg, D)
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute v = mean(H_pos) − mean(H_neg).

    Args:
        normalize: if True, return unit-norm vector (recommended for injection)

    Returns:
        v: np.ndarray of shape (D,)
    """
    raise NotImplementedError


def compute_contrast_vectors(
    data_pos: dict,    # loaded activation result dict for positive condition
    data_neg: dict,    # loaded activation result dict for negative condition
    layer_keys: list[str] | None = None,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute mean-difference vectors for all (or specified) layers.

    Returns:
        {layer_key: direction_vector}
    """
    raise NotImplementedError


def compute_all_contrasts(
    loaded_data: dict[str, dict],   # {persona_key: data_dict}
    contrasts: list[ContrastDef],
    normalize: bool = True,
    save_dir: str | Path | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute mean-difference vectors for a list of (positive, negative) contrasts.

    Args:
        loaded_data: mapping from persona_key to loaded result dict
        contrasts: list of (pos_key, neg_key) pairs defining each contrast
        save_dir: if given, save vectors as {contrast_name}.npy in this dir

    Returns:
        {contrast_name: {layer_key: direction_vector}}
        contrast_name = f"{pos_key}_vs_{neg_key}"
    """
    raise NotImplementedError


def load_vectors(path: str | Path) -> dict[str, np.ndarray]:
    """Load a saved {layer_key: vector} dict from a .npy file."""
    return np.load(path, allow_pickle=True).item()
