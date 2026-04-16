# metrics/tsne_embed.py
import os
import csv
import re
import inspect
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score


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


def _make_tsne(**kwargs):
    """Build a TSNE instance across sklearn versions.

    sklearn 1.5+ renamed `n_iter` to `max_iter`. Older versions still expect
    `n_iter`, so we choose the right keyword at runtime from the constructor
    signature.
    """
    sig = inspect.signature(TSNE.__init__)
    if "max_iter" in sig.parameters:
        kwargs["max_iter"] = kwargs.pop("n_iter")
    return TSNE(**kwargs)


def run_tsne_for_layer(
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
    setups: list of dicts: perplexity, learning_rate, metric, init
    Returns: list[dict] with keys:
      - setup, trustworthiness, silhouette_on_2D, plot, coords_csv, tag, coords (Nx2)
    """
    # Defaults (always 2D)
    if setups is None:
        setups = [
            {"perplexity": 30, "learning_rate": "auto", "metric": "euclidean", "init": "pca"},
            {"perplexity": 50, "learning_rate": 200,     "metric": "euclidean", "init": "pca"},
            {"perplexity": 30, "learning_rate": "auto",  "metric": "cosine",    "init": "pca"},
            {"perplexity": 10, "learning_rate": 100,     "metric": "cosine",    "init": "pca"},
        ]

    N = len(X)
    if N < 3:
        print(f"[TSNE] Not enough samples (N={N}). Skipping.")
        return []

    safe_layer = _safe_title(layer_id)
    base_dir = os.path.join(out_dir, safe_layer, "TSNE")
    os.makedirs(base_dir, exist_ok=True)

    results = []
    for params in setups:
        # t-SNE requires perplexity < N
        perpl = min(params["perplexity"], max(5, N - 1))
        tag = f"TSNE_p{perpl}__{params['metric']}__lr{params['learning_rate']}__init{params['init']}"

        tsne = _make_tsne(
            n_components=2,
            perplexity=perpl,
            learning_rate=params["learning_rate"],
            metric=params["metric"],
            init=params["init"],
            random_state=42,
            n_iter=1000,
            n_iter_without_progress=300,
        )
        X2 = tsne.fit_transform(X)

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
            "setup": {"algo": "TSNE", **params, "perplexity": perpl},
            "trustworthiness": float(tw),
            "silhouette_on_2D": (None if np.isnan(sil) else float(sil)),
            "plot": os.path.join(base_dir, f"{tag}.png"),
            "coords_csv": csv_path,
            "coords": X2.astype(np.float32),  # <-- unified schema: Nx2
        }
        results.append(res)

    # Optional index
    idx_csv = os.path.join(base_dir, "_summary.csv")
    with open(idx_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["tag","trustworthiness","silhouette","coords_csv","plot"])
        for r in results:
            w.writerow([r["tag"], r["trustworthiness"], r["silhouette_on_2D"], r["coords_csv"], r["plot"]])

    return results
