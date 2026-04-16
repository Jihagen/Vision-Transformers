# metrics/umap_embed.py
import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

try:
    import umap
except Exception:
    umap = None

from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness


def _safe_title(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-+=.]', '_', str(s))


def _save_scatter(X2, labels, title, out_png):
    plt.figure(figsize=(6, 6))
    for g in np.unique(labels):
        m = labels == g
        plt.scatter(X2[m, 0], X2[m, 1], label=str(g))
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.title(title)
    plt.legend(); plt.grid(True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()


def _coords_csv_path(base_dir, tag):
    return os.path.join(base_dir, f"{tag}__coords.csv")


def run_umap_for_layer(
    X: np.ndarray,
    y: np.ndarray,
    layer_id: str,
    out_dir: str,
    pooling_method: str,
    setups=None,
):
    """
    X: (N, D) pooled activations for one layer
    y: (N,) labels aligned to X
    layer_id: e.g., "vision 12 (D=1408)"
    out_dir: e.g., results/_gdv/plots/vision
    setups: list of dicts with UMAP params (n_neighbors, min_dist, metric, [n_components=2])
    Returns: list[dict] with keys:
      - setup, trustworthiness, silhouette_on_2D, plot, coords_csv, tag, coords (Nx2)
    """
    if umap is None:
        print("[UMAP] 'umap-learn' not installed. Skipping.")
        return []

    # Default setups (always 2D for interactivity)
    if setups is None:
        setups = [
            {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "n_components": 2},
            {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine",    "n_components": 2},
            {"n_neighbors": 5,  "min_dist": 0.0, "metric": "euclidean", "n_components": 2},
            {"n_neighbors": 50, "min_dist": 0.5, "metric": "cosine",    "n_components": 2},
        ]
    else:
        # Force 2D, silently, for all provided setups
        setups = [dict(s, n_components=2) for s in setups]

    N = len(X)
    if N < 3:
        print(f"[UMAP] Not enough samples (N={N}). Skipping.")
        return []

    safe_layer = _safe_title(layer_id)
    base_dir = os.path.join(out_dir, safe_layer, "UMAP")
    os.makedirs(base_dir, exist_ok=True)

    results = []
    for params in setups:
        tag = f"UMAP_n{params['n_neighbors']}__md{params['min_dist']}__{params['metric']}__nc{params['n_components']}"
        reducer = umap.UMAP(
            n_neighbors=min(params["n_neighbors"], max(2, N-1)),
            min_dist=float(params["min_dist"]),
            metric=params["metric"],
            n_components=2,
            random_state=42,
        )
        X2 = reducer.fit_transform(X)

        # Metrics
        tw = trustworthiness(X, X2, n_neighbors=min(10, max(2, N // 10)))
        try:
            sil = silhouette_score(X2, y) if (len(np.unique(y)) > 1 and N >= 10) else np.nan
        except Exception:
            sil = np.nan

        title = (f"{layer_id} ({pooling_method})\n"
                 f"{tag} | trust={tw:.3f} | silhouette={np.nan if np.isnan(sil) else round(sil,3)}")
        _save_scatter(X2, y, title, os.path.join(base_dir, f"{tag}.png"))

        # Save coords
        csv_path = _coords_csv_path(base_dir, tag)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "label"])
            for (x1, x2), lab in zip(X2, y):
                w.writerow([float(x1), float(x2), str(lab)])

        res = {
            "tag": tag,
            "setup": {"algo": "UMAP", **params},
            "trustworthiness": float(tw),
            "silhouette_on_2D": (None if np.isnan(sil) else float(sil)),
            "plot": os.path.join(base_dir, f"{tag}.png"),
            "coords_csv": csv_path,
            "coords": X2.astype(np.float32),  # <-- unified schema: Nx2
        }
        results.append(res)

    # Optional index file
    idx_csv = os.path.join(base_dir, "_summary.csv")
    with open(idx_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["tag","trustworthiness","silhouette","coords_csv","plot"])
        for r in results:
            w.writerow([r["tag"], r["trustworthiness"], r["silhouette_on_2D"], r["coords_csv"], r["plot"]])

    return results
