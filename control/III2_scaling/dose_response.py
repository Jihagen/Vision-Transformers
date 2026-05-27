"""
Dose-response experiment: sweep injection strength alpha and measure output shift.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# from control.III1_vector_addition.add_vector import inject_vector

DEFAULT_ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

ORDINAL_LABEL_MAP = {
    "Not Interesting": 0,
    "Slightly Interesting": 1,
    "Moderately Interesting": 2,
    "Very Interesting": 3,
    "Extremely Interesting": 4,
}


def run_dose_response(
    model,
    processor,
    images: list[str | Path],
    persona_dict: dict | None,
    vector: np.ndarray,       # (D,) concept direction for one layer
    layer_key: str,
    contrast: str,
    alphas: list[float] = DEFAULT_ALPHAS,
    save_dir: str | Path | None = None,
) -> dict[float, list[dict]]:
    """
    For each alpha in alphas, collect injection results for all images.

    Returns:
        {alpha: [result_dict per image]}
    """
    raise NotImplementedError


def compute_mean_ordinal_shift(
    dose_response_results: dict[float, list[dict]],
) -> dict[float, float]:
    """
    Convert labels to ordinal scores and return mean score per alpha.

    Used to plot the dose-response curve: alpha → mean_ordinal_score.
    """
    raise NotImplementedError


def plot_scaling_curve(
    mean_scores_by_alpha: dict[float, float],
    contrast: str,
    layer_key: str,
    save_path: str | Path | None = None,
):
    """
    Line plot of mean ordinal interestingness score vs injection alpha.

    Highlight alpha=0 as the no-injection baseline.
    Flag non-monotonic regions if present.
    """
    raise NotImplementedError
