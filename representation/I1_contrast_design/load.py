"""
Loading and preprocessing of stored activation result files.

Provides a clean interface so downstream modules (I.2–I.5, analytics, control)
never need to know the raw .npy dict schema.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np


def load_activation_results(path: str | Path) -> dict:
    """
    Load a results_<persona_key>.npy file.

    Returns the top-level dict with keys:
        version, persona_key, selected_images, results
    """
    data = np.load(path, allow_pickle=True).item()
    return data


def get_results_list(data: dict) -> list[dict]:
    """Extract the list of per-image result dicts from a loaded data dict."""
    return data["results"]


def get_layer_keys(results: list[dict]) -> list[str]:
    """Return sorted list of embedding layer keys from the first valid result."""
    for r in results:
        if r.get("embeddings"):
            return sorted(r["embeddings"].keys())
    return []


def filter_by_label_support(
    results: list[dict],
    min_fraction: float = 0.10,
    label_field: str = "interestingness",
) -> list[dict]:
    """
    Drop results from labels that have fewer than min_fraction of total samples.

    Mirrors the filtering in the original metrics.py to maintain consistency.
    """
    raise NotImplementedError


def stack_activations(
    results: list[dict],
    layer_key: str,
    label_field: str = "interestingness",
) -> tuple:
    """
    Stack per-image activation vectors for a given layer into arrays.

    Returns:
        X: np.ndarray of shape (N, D)
        y: list of str labels, length N
    """
    raise NotImplementedError


def load_multiple(paths: list[str | Path]) -> dict[str, dict]:
    """
    Load several result files and return {persona_key: data_dict}.

    persona_key is read from the stored dict, not the filename.
    """
    return {
        load_activation_results(p)["persona_key"]: load_activation_results(p)
        for p in paths
    }
