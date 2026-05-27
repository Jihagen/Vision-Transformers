"""
Vector addition experiment: inject alpha * v into model activations and measure output shift.

The injection hook is registered via utils/hooks.py before inference.
After inference, the hook is removed so the model returns to normal operation.

Output shift is measured as:
  - For interestingness: change in modal label or mean ordinal rating score.
  - For persona features: change in explanation content (keyword overlap, sentiment).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# from utils.hooks import HookState, register_inject_hooks
# from representation.I1_contrast_design.collect import model_response
# from utils.prompt_builder import build_persona_prompt


@dataclass
class InjectionResult:
    image_path: str
    persona_key: str
    layer_key: str
    alpha: float
    contrast: str                    # which concept vector was injected
    baseline_label: str              # rating without injection
    injected_label: str              # rating with injection
    baseline_explanation: str
    injected_explanation: str
    label_shifted: bool
    extra: dict = field(default_factory=dict)


def inject_vector(
    model,
    processor,
    image,
    prompt: str,
    vector: np.ndarray,      # (D,) concept direction
    layer_key: str,          # which layer to inject into
    alpha: float = 1.0,
    state=None,              # HookState — created fresh if None
) -> dict:
    """
    Run inference with alpha * vector added to the specified layer's hidden state.

    Returns the raw model output dict (interestingness, explanation).
    """
    raise NotImplementedError


def run_addition_experiment(
    model,
    processor,
    images: list[str | Path],
    persona_dict: dict | None,
    vectors: dict[str, np.ndarray],   # {layer_key: direction_vector}
    contrast: str,
    alpha: float = 1.0,
    save_dir: str | Path | None = None,
) -> list[InjectionResult]:
    """
    Run vector addition for each image and collect baseline vs injected outputs.

    Args:
        images: list of image paths
        persona_dict: persona for the prompt (None = blank)
        vectors: per-layer concept directions (from I.2 or I.3)
        contrast: name of the concept being injected (for logging)
        alpha: injection strength

    Returns:
        list of InjectionResult, one per image
    """
    raise NotImplementedError
