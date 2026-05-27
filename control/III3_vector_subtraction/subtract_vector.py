"""
Vector subtraction and projection-removal ablation experiments.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# from utils.hooks import HookState, register_inject_hooks, register_projection_removal_hooks
# from representation.I1_contrast_design.collect import model_response


def subtract_vector(
    model,
    processor,
    image,
    prompt: str,
    vector: np.ndarray,
    layer_key: str,
    alpha: float = 1.0,
    state=None,
) -> dict:
    """
    Run inference with alpha * vector subtracted from the specified layer.

    Equivalent to inject_vector(..., alpha=-alpha).  Kept as a separate entry
    point for clarity in experiment scripts.
    """
    raise NotImplementedError


def remove_projection(
    model,
    processor,
    image,
    prompt: str,
    vector: np.ndarray,   # will be unit-normalised internally
    layer_key: str,
    state=None,
) -> dict:
    """
    Run inference with the full projection of h onto vector removed.

    h' = h - (h · v̂) * v̂

    Stronger ablation than scalar subtraction; removes all variance along v.
    """
    raise NotImplementedError


def run_subtraction_experiment(
    model,
    processor,
    images: list[str | Path],
    persona_dict: dict | None,
    vectors: dict[str, np.ndarray],
    contrast: str,
    alpha: float = 1.0,
    mode: str = "subtract",   # "subtract" | "project_remove"
    save_dir: str | Path | None = None,
) -> list[dict]:
    """
    Run ablation for each image, collecting baseline vs ablated outputs.

    Args:
        mode: "subtract" uses soft scalar suppression;
              "project_remove" uses full projection removal.
    """
    raise NotImplementedError
