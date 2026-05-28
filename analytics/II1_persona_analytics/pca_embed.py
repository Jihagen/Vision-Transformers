"""
PCA 2D scatter visualization of layer activations.

Migrated from metrics/metrics.py (PCA section of run_metrics).
"""

from __future__ import annotations
import logging
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from representation.I1_contrast_design.load import build_layer_matrix, sort_layer_keys

logger = logging.getLogger(__name__)


def _safe_title(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-+=.]", "_", str(s))


def _save_scatter(X2: np.ndarray, labels, title: str, out_png: str) -> None:
    plt.figure(figsize=(6, 6))
    for g in np.unique(labels):
        m = np.array(labels) == g
        plt.scatter(X2[m, 0], X2[m, 1], label=str(g))
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(title)
    plt.legend(); plt.grid(True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def run_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Fit PCA and return projected coordinates, shape (N, n_components)."""
    pcs = min(n_components, X.shape[0], X.shape[1])
    return PCA(n_components=pcs).fit_transform(X)


def plot_pca_scatter(
    coords: np.ndarray,
    labels,
    title: str = "",
    save_path: str | Path | None = None,
) -> None:
    """Scatter plot of 2D PCA coordinates coloured by label."""
    plt.figure(figsize=(6, 6))
    for g in np.unique(labels):
        m = np.array(labels) == g
        plt.scatter(coords[m, 0], coords[m, 1], label=str(g))
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(title)
    plt.legend(); plt.grid(True)
    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def run_pca_for_layer(
    X: np.ndarray,
    labels,
    layer_id: str,
    gdv_euc: float = float("nan"),
    gdv_cos: float = float("nan"),
    hook_selection: str = "",
    out_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Run PCA with n_components ∈ {2, 4} and save scatter plots (PC1v2, PC2v3, PC3v4).

    Returns:
        {"PC12": coords_Nx2, "PC23": coords_Nx2, "PC34": coords_Nx2}
    """
    max_pcs  = 4
    n_pcs    = int(min(max_pcs, X.shape[0], X.shape[1]))
    Xp       = PCA(n_components=max(2, n_pcs)).fit_transform(X)
    projections: dict[str, np.ndarray] = {}

    pairs = [(0, 1, "PC12"), (1, 2, "PC23"), (2, 3, "PC34")]
    for a, b, tag in pairs:
        if b >= n_pcs:
            break
        coords = Xp[:, [a, b]].astype(np.float32)
        projections[tag] = coords

        if out_dir is not None:
            safe   = _safe_title(layer_id)
            out_png = os.path.join(
                str(out_dir), safe, "GDV", f"{tag}.png"
            )
            title = (
                f"{layer_id} [{hook_selection}]\n"
                f"GDV(Euc)={gdv_euc:.4f} | GDV(Cos)={gdv_cos:.4f}\n"
                f"PC{a+1} vs PC{b+1}"
            )
            _save_scatter(coords, labels, title, out_png)

    return projections


def run_pca_pipeline(
    results: list[dict],
    layer_keys: list[str] | None = None,
    min_fraction: float = 0.0,
    save_dir: str | Path | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Run PCA for each layer and optionally save scatter plots.

    Returns:
        {comp_key: {"PC12": coords, "PC23": coords, ...}}
    """
    layer_data = build_layer_matrix(results, min_fraction=min_fraction)

    if layer_keys is not None:
        layer_data = {k: v for k, v in layer_data.items() if k in layer_keys}

    output: dict[str, dict[str, np.ndarray]] = {}
    for comp_key in sort_layer_keys(list(layer_data.keys())):
        info = layer_data[comp_key]
        out_dir = (
            os.path.join(str(save_dir), info["modality"]) if save_dir else None
        )
        projections = run_pca_for_layer(
            X=info["X"],
            labels=info["y"],
            layer_id=f"{info['modality']} {info['layer_num']} (D={info['D']})",
            hook_selection=info["hook_selection"],
            out_dir=out_dir,
        )
        output[comp_key] = projections

    return output
