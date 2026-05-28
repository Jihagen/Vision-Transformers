"""
General Discrimination Value (GDV) — cluster separation metric.

Migrated from metrics/gdv.py.  Corrected formula matches Hagen et al.:
  GDV = (intra_mean − inter_mean) / √D

Negative GDV → well-separated clusters (good).
Positive GDV → within-class > between-class distances (anti-clustering).
Near-zero   → no discriminative structure.

Computed for both Euclidean and cosine distance (scipy 'cosine' = 1 − cos_sim).
Macro-weighted: each class / class-pair counts equally regardless of support.
"""

from __future__ import annotations
import csv
import logging
import os
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist

from representation.I1_contrast_design.load import build_layer_matrix, sort_layer_keys

logger = logging.getLogger(__name__)


# ── Core distance + GDV functions (ported from metrics/gdv.py) ────────────────

def _mean_intra_inter(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    weighting: str = "macro",
) -> tuple[float, float]:
    """
    Compute macro-averaged mean intra-class and inter-class pairwise distances.

    Returns:
        (intra_mean, inter_mean)
    """
    labels = np.array([str(l).strip().lower() for l in labels])
    classes, _ = np.unique(labels, return_counts=True)

    # Intra-class
    intra_vals, intra_weights = [], []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        d = pdist(X[idx], metric=metric)
        if len(d) == 0:
            continue
        intra_vals.append(np.mean(d))
        intra_weights.append(len(d) if weighting == "micro" else 1.0)
    intra_mean = float(np.average(intra_vals, weights=intra_weights)) if intra_vals else 0.0

    # Inter-class
    inter_vals, inter_weights = [], []
    if len(classes) >= 2:
        for i, j in combinations(range(len(classes)), 2):
            idx_i = np.where(labels == classes[i])[0]
            idx_j = np.where(labels == classes[j])[0]
            if len(idx_i) == 0 or len(idx_j) == 0:
                continue
            Dij = cdist(X[idx_i], X[idx_j], metric=metric)
            inter_vals.append(float(np.mean(Dij)))
            inter_weights.append(Dij.size if weighting == "micro" else 1.0)
    inter_mean = float(np.average(inter_vals, weights=inter_weights)) if inter_vals else 0.0

    return intra_mean, inter_mean


def compute_gdv(
    X: np.ndarray,
    labels,
    metric: str = "euclidean",
    weighting: str = "macro",
    zscore: bool = True,
) -> dict:
    """
    Compute GDV for a given metric.

    Returns dict with keys: metric, intra, inter, gdv.
    """
    X_ = X
    if zscore:
        mu    = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-12
        X_    = (X - mu) / sigma * 0.5      # scale per Hagen et al.

    D = X_.shape[1]
    K = len(np.unique([str(l).strip().lower() for l in labels]))
    if K < 2:
        return {"metric": metric, "intra": 0.0, "inter": 0.0, "gdv": 0.0}

    intra, inter = _mean_intra_inter(X_, labels, metric=metric, weighting=weighting)
    gdv = (intra - inter) / np.sqrt(D)      # Hagen et al. formula (K-weighted variant removed)
    return {"metric": metric, "intra": intra, "inter": inter, "gdv": float(gdv)}


def compute_gdv_both(X: np.ndarray, labels, weighting: str = "macro") -> dict:
    """
    Return {'euclidean': {...}, 'cosine': {...}} GDV dicts for one layer.
    """
    return {
        "euclidean": compute_gdv(X, labels, metric="euclidean", weighting=weighting),
        "cosine":    compute_gdv(X, labels, metric="cosine",    weighting=weighting),
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_gdv_pipeline(
    results: list[dict],
    layer_keys: list[str] | None = None,
    min_fraction: float = 0.0,
    save_dir: str | Path | None = None,
) -> dict[str, dict]:
    """
    Compute GDV for all layers in a result set.

    Args:
        results:      list of per-image result dicts (from load.py)
        layer_keys:   subset of comp_keys to evaluate (None = all)
        min_fraction: minimum label fraction for inclusion (0.0 = all classes kept)
        save_dir:     if given, write gdv.pkl and gdv_values.csv here

    Returns:
        {comp_key: {'euclidean': {...}, 'cosine': {...}, 'pooling': str,
                    'modality': str, 'layer_num': int, 'D': int, 'n_samples': int}}
    """
    layer_data = build_layer_matrix(results, min_fraction=min_fraction)

    if layer_keys is not None:
        layer_data = {k: v for k, v in layer_data.items() if k in layer_keys}

    output: dict[str, dict] = {}
    for comp_key, info in layer_data.items():
        X, y = info["X"], info["y"]
        gdv_both = compute_gdv_both(X, y)
        output[comp_key] = {
            **gdv_both,
            "hook_selection": info["hook_selection"],
            "modality":       info["modality"],
            "layer_num":      info["layer_num"],
            "D":              info["D"],
            "n_samples":      info["n_samples"],
        }
        logger.debug(
            f"{comp_key}: GDV_euc={gdv_both['euclidean']['gdv']:.4f}  "
            f"GDV_cos={gdv_both['cosine']['gdv']:.4f}"
        )

    if save_dir is not None:
        _save_gdv_results(output, Path(save_dir))

    return output


def _save_gdv_results(output: dict[str, dict], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    sorted_keys = sort_layer_keys(list(output.keys()))

    # CSV
    csv_path = save_dir / "gdv_values.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Modality", "Layer_Num", "Width_D", "LayerKey",
            "GDV_Euclidean", "Intra_Euclidean", "Inter_Euclidean",
            "GDV_Cosine",    "Intra_Cosine",    "Inter_Cosine",
            "Pooling", "Samples_Used",
        ])
        for k in sorted_keys:
            info = output[k]
            euc  = info["euclidean"]
            cos  = info["cosine"]
            w.writerow([
                info["modality"], info["layer_num"], info["D"], k,
                euc["gdv"], euc["intra"], euc["inter"],
                cos["gdv"], cos["intra"], cos["inter"],
                info["pooling"], info["n_samples"],
            ])
    logger.info(f"Saved GDV CSV → {csv_path}")

    # PKL
    pkl_path = save_dir / "gdv.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "sorted_layers":       sorted_keys,
            "layer_data":          output,
            "gdv_per_layer":       {k: output[k]["euclidean"]["gdv"] for k in sorted_keys},
            "gdv_per_layer_cosine":{k: output[k]["cosine"]["gdv"]    for k in sorted_keys},
        }, f)
    logger.info(f"Saved GDV PKL  → {pkl_path}")
