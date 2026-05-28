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
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from representation.I1_contrast_design.load import (
    build_layer_matrix,
    get_results_list,
    sort_layer_keys,
)

logger = logging.getLogger(__name__)


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
    layer_key: str = "",
    contrast: str = "",
) -> ProbeResult:
    """
    Train a logistic regression and return accuracy + CAV direction.

    The CAV direction is the L2-normalized coefficient vector of the fitted
    LogisticRegression, equivalent to the TCAV concept activation vector.

    Args:
        H_pos: activations for positive class, shape (N_pos, D)
        H_neg: activations for negative class, shape (N_neg, D)
        cv_folds: number of stratified cross-validation folds for accuracy estimate
        normalize_vector: if True, L2-normalize the weight vector
        layer_key: stored in ProbeResult for reference
        contrast: stored in ProbeResult for reference

    Returns:
        ProbeResult with cav_vector and cross-val accuracy
    """
    X = np.vstack([H_pos, H_neg]).astype(np.float64)
    y = np.concatenate([np.ones(len(H_pos)), np.zeros(len(H_neg))])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_splits = min(cv_folds, min(len(H_pos), len(H_neg)))
    if n_splits < 2:
        logger.warning(
            f"[Probe] {contrast} {layer_key}: too few samples for CV "
            f"(pos={len(H_pos)}, neg={len(H_neg)}) — fitting without CV"
        )
        cv_scores = np.array([float("nan")])
    else:
        clf_cv = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf_cv, X_scaled, y, cv=cv, scoring="accuracy")

    # Fit on full data to get stable weight vector
    clf = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
    clf.fit(X_scaled, y)
    w = clf.coef_[0].astype(np.float32)

    if normalize_vector:
        norm = np.linalg.norm(w)
        if norm > 1e-12:
            w = w / norm

    return ProbeResult(
        layer_key=layer_key,
        contrast=contrast,
        accuracy_cv=float(np.nanmean(cv_scores)),
        accuracy_std=float(np.nanstd(cv_scores)),
        cav_vector=w,
        n_pos=len(H_pos),
        n_neg=len(H_neg),
    )


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
    output: dict[str, list[ProbeResult]] = {}

    for pos_key, neg_key in contrasts:
        name = f"{pos_key}_vs_{neg_key}"
        if pos_key not in loaded_data or neg_key not in loaded_data:
            logger.warning(f"[Probe] Missing keys for contrast {name} — skipping")
            continue

        matrix_pos = build_layer_matrix(get_results_list(loaded_data[pos_key]))
        matrix_neg = build_layer_matrix(get_results_list(loaded_data[neg_key]))

        common = set(matrix_pos.keys()) & set(matrix_neg.keys())
        if layer_keys is not None:
            common = common & set(layer_keys)

        probe_list: list[ProbeResult] = []
        for lk in sort_layer_keys(list(common)):
            H_pos = matrix_pos[lk]["X"]
            H_neg = matrix_neg[lk]["X"]
            try:
                result = train_linear_probe(
                    H_pos, H_neg,
                    cv_folds=cv_folds,
                    layer_key=lk,
                    contrast=name,
                )
                probe_list.append(result)
                logger.debug(
                    f"[Probe] {name} | {lk}: "
                    f"acc={result.accuracy_cv:.3f}±{result.accuracy_std:.3f}"
                )
            except Exception as e:
                logger.warning(f"[Probe] {name} | {lk} failed: {e}")

        output[name] = probe_list
        logger.info(
            f"[Probe] {name}: trained probes for {len(probe_list)} layers"
        )

    if save_dir is not None:
        _save_probe_results(output, Path(save_dir))

    return output


def _save_probe_results(
    all_results: dict[str, list[ProbeResult]],
    save_dir: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "probe_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "layer_key", "accuracy_cv", "accuracy_std",
                    "n_pos", "n_neg"])
        for contrast_name, results in all_results.items():
            for r in results:
                w.writerow([
                    contrast_name, r.layer_key,
                    r.accuracy_cv, r.accuracy_std,
                    r.n_pos, r.n_neg,
                ])
    logger.info(f"[Probe] Saved accuracy CSV → {csv_path}")

    # Save CAV vectors per contrast
    for contrast_name, results in all_results.items():
        cavs = {r.layer_key: r.cav_vector for r in results}
        np.save(save_dir / f"cav_{contrast_name}.npy", cavs)


def get_cav_vectors(probe_results: list[ProbeResult]) -> dict[str, np.ndarray]:
    """Extract {layer_key: cav_vector} from a list of ProbeResults for one contrast."""
    return {r.layer_key: r.cav_vector for r in probe_results}


def summarize_probe_accuracy(
    all_results: dict[str, list[ProbeResult]],
) -> object:
    """
    Return a DataFrame summarising cross-val accuracy per contrast × layer.

    Falls back to a plain list of dicts if pandas is not available.
    """
    rows = [
        {
            "contrast":     contrast,
            "layer_key":    r.layer_key,
            "accuracy_cv":  r.accuracy_cv,
            "accuracy_std": r.accuracy_std,
            "n_pos":        r.n_pos,
            "n_neg":        r.n_neg,
        }
        for contrast, results in all_results.items()
        for r in results
    ]
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows
