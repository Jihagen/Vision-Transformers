import numpy as np
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from itertools import combinations
import csv
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import pickle
from collections import defaultdict
from typing import Optional

# ───────────────────────────────────────────────────────────────────────────────
# GDV FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────────────
# DISTANCES & GDV (supports Euclidean and Cosine, macro/micro averaging)
# ───────────────────────────────────────────────────────────────────────────────

def _mean_intra_inter(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    weighting: str = "macro",
):
    """
    Returns:
      intra_mean, inter_mean
    - metric: 'euclidean' or 'cosine' (scipy 'cosine' = cosine *distance* = 1 - cosine_similarity)
    - weighting:
        'macro' → each class (intra) / class-pair (inter) contributes equally
        'micro' → weighted by number of pairs within class / between class-pair
    """
    labels = np.array([str(l).strip().lower() for l in labels])
    classes, counts = np.unique(labels, return_counts=True)

    # ---- Intra-class ----
    intra_vals, intra_weights = [], []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        Xi = X[idx]
        # pdist for 'euclidean' and 'cosine' 
        d = pdist(Xi, metric=metric)
        if len(d) == 0:
            continue
        intra_vals.append(np.mean(d))
        # weight = number of pairs if micro; 1 if macro 
        w = len(d) if weighting == "micro" else 1.0
        intra_weights.append(w)
    intra_mean = float(np.average(intra_vals, weights=intra_weights)) if intra_vals else 0.0

    # ---- Inter-class ----
    inter_vals, inter_weights = [], []
    if len(classes) >= 2:
        for i, j in combinations(range(len(classes)), 2):
            idx_i = np.where(labels == classes[i])[0]
            idx_j = np.where(labels == classes[j])[0]
            if len(idx_i) == 0 or len(idx_j) == 0:
                continue
            # cdist supports both metrics; returns (len_i, len_j) matrix
            Dij = cdist(X[idx_i], X[idx_j], metric=metric)
            m = float(np.mean(Dij))
            inter_vals.append(m)
            # weight by number of cross-pairs if micro; else 1
            w = (Dij.size if weighting == "micro" else 1.0)
            inter_weights.append(w)
    inter_mean = float(np.average(inter_vals, weights=inter_weights)) if inter_vals else 0.0

    return intra_mean, inter_mean


def compute_gdv_metric(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    weighting: str = "macro",
    zscore: bool = True,
) -> dict:
    """
    Compute GDV for a given metric, returning a dict with intra/inter & gdv.
    """
    X_ = X
    if zscore:
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-12
        X_ = (X - mu) / sigma
        X_ *= 0.5  # scale per paper

    D = X_.shape[1]
    K = len(np.unique([str(l).strip().lower() for l in labels]))
    if K < 2:
        return {"metric": metric, "intra": 0.0, "inter": 0.0, "gdv": 0.0}

    intra, inter = _mean_intra_inter(X_, labels, metric=metric, weighting=weighting)
    if K == 2:
        # Centered 2-class formula: no-separation case (intra ~= inter) -> GDV ~= 0
        gdv = (intra - inter) / np.sqrt(D)
    else:
        gdv = (1 / np.sqrt(D)) * ((1 / K) * intra - (2 / (K * (K - 1))) * inter)
    return {"metric": metric, "intra": intra, "inter": inter, "gdv": float(gdv)}


def compute_gdv_both(X: np.ndarray, labels: np.ndarray, weighting: str = "macro") -> dict:
    """
    Returns:
      {
        'euclidean': {'intra':..., 'inter':..., 'gdv':...},
        'cosine':    {'intra':..., 'inter':..., 'gdv':...}
      }
    """
    return {
        "euclidean": compute_gdv_metric(X, labels, metric="euclidean", weighting=weighting),
        "cosine":    compute_gdv_metric(X, labels, metric="cosine",    weighting=weighting),
    }



