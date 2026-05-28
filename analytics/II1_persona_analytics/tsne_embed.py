"""
t-SNE embedding + trustworthiness for layer activations.

Migrated from metrics/tsne.py.  Multiple hyperparameter setups are swept so no
single configuration is cherry-picked; trustworthiness (sklearn) is the primary
quantitative output alongside GDV.
"""

from __future__ import annotations
import csv
import inspect
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score

from representation.I1_contrast_design.load import build_layer_matrix, sort_layer_keys

logger = logging.getLogger(__name__)


@dataclass
class TSNESetup:
    perplexity: int
    learning_rate: int | str
    metric: str
    init: str = "pca"


DEFAULT_SETUPS: list[TSNESetup] = [
    TSNESetup(30, "auto", "euclidean"),
    TSNESetup(50, 200,    "euclidean"),
    TSNESetup(30, "auto", "cosine"),
    TSNESetup(10, 100,    "cosine"),
]


def _safe_title(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-+=.]", "_", str(s))


def _save_scatter(X2: np.ndarray, labels, title: str, out_png: str) -> None:
    plt.figure(figsize=(6, 6))
    for g in np.unique(labels):
        m = np.array(labels) == g
        plt.scatter(X2[m, 0], X2[m, 1], label=str(g))
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.title(title)
    plt.legend(); plt.grid(True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def _make_tsne(**kwargs) -> TSNE:
    """Build TSNE instance across sklearn versions (n_iter → max_iter in 1.5+)."""
    sig = inspect.signature(TSNE.__init__)
    if "max_iter" in sig.parameters:
        kwargs["max_iter"] = kwargs.pop("n_iter", 1000)
    else:
        kwargs.setdefault("n_iter", 1000)
    return TSNE(**kwargs)


def run_tsne_for_layer(
    X: np.ndarray,
    labels,
    layer_id: str,
    out_dir: str | Path | None = None,
    hook_selection: str = "",
    setups: list[TSNESetup] | None = None,
) -> list[dict]:
    """
    Run t-SNE with each setup, compute trustworthiness, optionally save scatter plots.

    Returns:
        list of dicts with keys: tag, setup, trustworthiness, silhouette_on_2D,
                                  coords (Nx2 np.float32), plot, coords_csv
    """
    if setups is None:
        setups = DEFAULT_SETUPS

    N = len(X)
    if N < 3:
        logger.info(f"[t-SNE] Not enough samples (N={N}) — skipping {layer_id}")
        return []

    safe_layer = _safe_title(layer_id)
    base_dir   = os.path.join(str(out_dir), safe_layer, "TSNE") if out_dir else None
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    results = []
    for s in setups:
        perpl = min(s.perplexity, max(5, N - 1))
        tag   = f"TSNE_p{perpl}__{s.metric}__lr{s.learning_rate}__init{s.init}"

        tsne = _make_tsne(
            n_components=2,
            perplexity=perpl,
            learning_rate=s.learning_rate,
            metric=s.metric,
            init=s.init,
            random_state=42,
            n_iter=1000,
            n_iter_without_progress=300,
        )
        X2 = tsne.fit_transform(X)

        tw = trustworthiness(X, X2, n_neighbors=min(10, max(2, N // 10)))
        try:
            sil = silhouette_score(X2, labels) if (len(np.unique(labels)) > 1 and N >= 10) else np.nan
        except Exception:
            sil = np.nan

        plot_path, csv_path = None, None
        if base_dir:
            title = (
                f"{layer_id} [{hook_selection}]\n"
                f"{tag} | trust={tw:.3f} | "
                f"sil={'nan' if np.isnan(sil) else round(sil, 3)}"
            )
            plot_path = os.path.join(base_dir, f"{tag}.png")
            _save_scatter(X2, labels, title, plot_path)

            csv_path = os.path.join(base_dir, f"{tag}__coords.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x", "y", "label"])
                for (x1, x2), lab in zip(X2, labels):
                    w.writerow([float(x1), float(x2), str(lab)])

        results.append({
            "tag":             tag,
            "setup":           {"algo": "TSNE", "perplexity": perpl,
                                "learning_rate": s.learning_rate,
                                "metric": s.metric, "init": s.init},
            "trustworthiness": float(tw),
            "silhouette_on_2D": None if np.isnan(sil) else float(sil),
            "coords":          X2.astype(np.float32),
            "plot":            plot_path,
            "coords_csv":      csv_path,
        })

    if base_dir and results:
        idx_csv = os.path.join(base_dir, "_summary.csv")
        with open(idx_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "trustworthiness", "silhouette", "coords_csv", "plot"])
            for r in results:
                w.writerow([r["tag"], r["trustworthiness"], r["silhouette_on_2D"],
                            r["coords_csv"], r["plot"]])

    return results


def run_tsne_pipeline(
    results: list[dict],
    layer_keys: list[str] | None = None,
    setups: list[TSNESetup] | None = None,
    min_fraction: float = 0.0,
    save_dir: str | Path | None = None,
) -> dict[str, list[dict]]:
    """
    Run t-SNE pipeline for all (or specified) layers.

    Returns:
        {comp_key: [tsne_result per setup]}
    """
    layer_data = build_layer_matrix(results, min_fraction=min_fraction)

    if layer_keys is not None:
        layer_data = {k: v for k, v in layer_data.items() if k in layer_keys}

    output: dict[str, list[dict]] = {}
    for comp_key in sort_layer_keys(list(layer_data.keys())):
        info    = layer_data[comp_key]
        out_dir = (
            os.path.join(str(save_dir), info["modality"]) if save_dir else None
        )
        layer_id = f"{info['modality']} {info['layer_num']} (D={info['D']})"
        try:
            output[comp_key] = run_tsne_for_layer(
                X=info["X"],
                labels=info["y"],
                layer_id=layer_id,
                out_dir=out_dir,
                hook_selection=info["hook_selection"],
                setups=setups,
            )
        except Exception as e:
            logger.warning(f"[t-SNE] {comp_key} failed: {e}")
            output[comp_key] = []

    return output
