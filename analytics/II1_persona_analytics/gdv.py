"""
General Discrimination Value (GDV) — cluster separation metric.

Migrated from metrics/gdv.py.  Corrected formula matches Hagen et al.:
  GDV = (intra_mean − inter_mean) / √D

Negative GDV → well-separated clusters (good).
Positive GDV → within-class > between-class (anti-clustering).
Near-zero   → no discriminative structure.

Computed for both Euclidean and cosine distance.
Macro-weighted: each class / class-pair counts equally regardless of support.
"""

from __future__ import annotations

import numpy as np

# NOTE: Implementation body should be lifted from metrics/gdv.py after migration.


def _mean_intra_inter(
    X: np.ndarray,         # (N, D)
    labels: list[str],
    metric: str = "euclidean",  # "euclidean" | "cosine"
) -> tuple[float, float]:
    """
    Compute macro-averaged mean intra-class and inter-class pairwise distances.

    Returns:
        (intra_mean, inter_mean)
    """
    raise NotImplementedError


def compute_gdv(
    X: np.ndarray,
    labels: list[str],
    metric: str = "euclidean",
) -> float:
    """
    Compute GDV = (intra_mean − inter_mean) / √D for one layer.

    Handles z-score normalization (scale by ½) before distance computation,
    matching the original Hagen et al. implementation.
    """
    raise NotImplementedError


def compute_gdv_both(
    X: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    """Return {'euclidean': gdv, 'cosine': gdv} for convenience."""
    return {
        "euclidean": compute_gdv(X, labels, metric="euclidean"),
        "cosine": compute_gdv(X, labels, metric="cosine"),
    }


def run_gdv_pipeline(
    results: list[dict],
    layer_keys: list[str] | None = None,
    save_dir=None,
) -> dict[str, dict]:
    """
    Compute GDV for all layers in a result set.

    Args:
        results: list of per-image result dicts (from I1_contrast_design/load.py)
        layer_keys: subset of layers to evaluate (None = all)
        save_dir: if given, write gdv.pkl and gdv_values.csv here

    Returns:
        {layer_key: {'euclidean': float, 'cosine': float, 'n_samples': int, 'D': int}}
    """
    raise NotImplementedError
