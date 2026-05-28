"""
Mean-difference vector computation.

For a given contrast (positive condition vs negative condition) and a given layer,
compute v = mean(H_pos) - mean(H_neg) and optionally L2-normalize.

Vectors are computed per layer and saved as a dict so downstream steps can
select the most informative layer(s) after I.4 evaluation.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

from representation.I1_contrast_design.load import (
    build_layer_matrix,
    get_results_list,
    sort_layer_keys,
)

logger = logging.getLogger(__name__)

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
    v = H_pos.mean(axis=0) - H_neg.mean(axis=0)
    if normalize:
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v = v / norm
    return v.astype(np.float32)


def compute_contrast_vectors(
    data_pos: dict,    # loaded activation result dict for positive condition
    data_neg: dict,    # loaded activation result dict for negative condition
    layer_keys: list[str] | None = None,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute mean-difference vectors for all (or specified) layers.

    Returns:
        {comp_key: direction_vector}
    """
    results_pos = get_results_list(data_pos)
    results_neg = get_results_list(data_neg)

    matrix_pos = build_layer_matrix(results_pos)
    matrix_neg = build_layer_matrix(results_neg)

    common_keys = set(matrix_pos.keys()) & set(matrix_neg.keys())
    if layer_keys is not None:
        common_keys = common_keys & set(layer_keys)

    vectors: dict[str, np.ndarray] = {}
    for key in sort_layer_keys(list(common_keys)):
        H_pos = matrix_pos[key]["X"]
        H_neg = matrix_neg[key]["X"]
        vectors[key] = compute_mean_difference_vector(H_pos, H_neg, normalize=normalize)
        logger.debug(
            f"[MD] {key}: pos_mean={H_pos.mean():.4f} neg_mean={H_neg.mean():.4f} "
            f"norm={np.linalg.norm(vectors[key]):.4f}"
        )

    return vectors


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
        {contrast_name: {comp_key: direction_vector}}
        contrast_name = f"{pos_key}_vs_{neg_key}"
    """
    output: dict[str, dict[str, np.ndarray]] = {}
    for pos_key, neg_key in contrasts:
        name = f"{pos_key}_vs_{neg_key}"
        if pos_key not in loaded_data:
            logger.warning(f"[MD] Missing positive key '{pos_key}' — skipping {name}")
            continue
        if neg_key not in loaded_data:
            logger.warning(f"[MD] Missing negative key '{neg_key}' — skipping {name}")
            continue

        vectors = compute_contrast_vectors(
            loaded_data[pos_key],
            loaded_data[neg_key],
            normalize=normalize,
        )
        output[name] = vectors
        logger.info(f"[MD] {name}: computed vectors for {len(vectors)} layers")

        if save_dir is not None:
            out_path = Path(save_dir) / f"{name}.npy"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, vectors)
            logger.info(f"[MD] Saved → {out_path}")

    return output


def load_vectors(path: str | Path) -> dict[str, np.ndarray]:
    """Load a saved {comp_key: vector} dict from a .npy file."""
    return np.load(path, allow_pickle=True).item()
