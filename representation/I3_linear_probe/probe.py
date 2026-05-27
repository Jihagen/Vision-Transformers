"""
Linear probe / CAV-style concept vector extraction.

For each layer and each contrast pair, train a logistic regression on
(H_pos, H_neg) and extract the decision boundary normal as the concept direction.

Cross-validation accuracy is reported as the decodability score.
The weight vector (after L2 normalization) is the CAV direction.

When the concept is binary (e.g. female vs male), a single linear probe suffices.
For multi-class concepts (e.g. 5-way interestingness labels), use one-vs-rest
or pairwise probes and report per-class accuracy.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    layer_key: str
    contrast: str                       # e.g. "female_anger_vs_male_anger"
    accuracy_cv: float                  # mean cross-val accuracy
    accuracy_std: float                 # std cross-val accuracy
    cav_vector: np.ndarray              # unit-norm weight vector, shape (D,)
    n_pos: int
    n_neg: int
    extra: dict = field(default_factory=dict)


def train_linear_probe(
    H_pos: np.ndarray,    # shape (N_pos, D)
    H_neg: np.ndarray,    # shape (N_neg, D)
    cv_folds: int = 5,
    max_iter: int = 1000,
    normalize_vector: bool = True,
) -> ProbeResult:
    """
    Train a logistic regression and return accuracy + CAV direction.

    The CAV direction is the L2-normalized coefficient vector of the fitted
    LogisticRegression, equivalent to the TCAV concept activation vector.

    Args:
        H_pos: activations for positive class
        H_neg: activations for negative class
        cv_folds: number of cross-validation folds for accuracy estimate
        normalize_vector: if True, L2-normalize the weight vector

    Returns:
        ProbeResult with cav_vector and cross-val accuracy
    """
    raise NotImplementedError


def train_all_probes(
    loaded_data: dict[str, dict],     # {persona_key: data_dict}
    contrasts: list[tuple[str, str]], # list of (pos_key, neg_key)
    layer_keys: list[str] | None = None,
    cv_folds: int = 5,
    save_dir: str | Path | None = None,
) -> dict[str, list[ProbeResult]]:
    """
    Train probes for all contrasts × all layers.

    Returns:
        {contrast_name: [ProbeResult per layer]}
    """
    raise NotImplementedError


def get_cav_vectors(probe_results: list[ProbeResult]) -> dict[str, np.ndarray]:
    """Extract {layer_key: cav_vector} from a list of ProbeResults for one contrast."""
    return {r.layer_key: r.cav_vector for r in probe_results}


def summarize_probe_accuracy(
    all_results: dict[str, list[ProbeResult]],
) -> object:
    """
    Return a DataFrame summarising cross-val accuracy per contrast × layer.

    Useful for identifying which layers have strong decodability.
    """
    raise NotImplementedError
