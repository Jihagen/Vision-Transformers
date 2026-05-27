"""
Vector swap: replace one persona direction with another during inference.

h' = h - alpha * v_A + alpha * v_B

where v_A is the direction associated with the input persona and v_B is the
target persona direction.  The prompt still encodes persona A; only the
activations are steered toward B.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# from control.III1_vector_addition.add_vector import inject_vector


def swap_concept_vector(
    model,
    processor,
    image,
    prompt: str,
    v_source: np.ndarray,    # concept direction of the prompt's persona (to remove)
    v_target: np.ndarray,    # concept direction to inject
    layer_key: str,
    alpha: float = 1.0,
    state=None,
) -> dict:
    """
    Subtract source direction and add target direction at the specified layer.

    h' = h - alpha * v_source + alpha * v_target
    """
    raise NotImplementedError


def run_vector_swap_experiment(
    model,
    processor,
    images: list[str | Path],
    persona_dict_source: dict | None,
    v_source: np.ndarray,
    v_target: np.ndarray,
    contrast_source: str,
    contrast_target: str,
    layer_key: str,
    alpha: float = 1.0,
    save_dir: str | Path | None = None,
) -> list[dict]:
    """
    Run swap experiment for each image and collect source vs swapped outputs.

    For each image, compare:
      - Baseline (no injection, prompt = persona A)
      - Swapped (h' = h - alpha*v_A + alpha*v_B, prompt = persona A)
      - Reference (no injection, prompt = persona B — collected in I.1)

    Returns list of result dicts with all three conditions.
    """
    raise NotImplementedError
