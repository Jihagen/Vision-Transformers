"""
II.1 Condition geometry analytics.

Extracts one mean activation vector per condition at a target layer, then:
  - Computes pairwise cosine similarity matrix across all conditions
  - Produces UMAP / PCA projection of the condition mean vectors
  - Answers: does the model separate conditions by gender, by emotion, or both?

This is the primary II.1 analytics step — it operates on the raw collected
activations (not on contrast/difference vectors).
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

EMOTION_COLORS = {
    "anger":       "#e74c3c",
    "amusement":   "#f39c12",
    "awe":         "#9b59b6",
    "contentment": "#27ae60",
    "disgust":     "#795548",
    "excitement":  "#e91e63",
    "fear":        "#1a237e",
    "sad":         "#2196f3",
}
GENDER_MARKERS = {"female": "o", "male": "^"}


def get_condition_mean_vectors(
    loaded_data: dict[str, dict],
    layer_key: str,
) -> dict[str, np.ndarray]:
    """
    Extract the mean activation vector per condition at the given layer.

    Args:
        loaded_data: {persona_key: data_dict} as returned by load_multiple
        layer_key:   comp_key like "language_29_D5120"

    Returns:
        {persona_key: mean_vector}  — only conditions where layer_key exists
    """
    from representation.I1_contrast_design.load import build_layer_matrix, get_results_list

    means: dict[str, np.ndarray] = {}
    for pk, data in loaded_data.items():
        results = get_results_list(data)
        if not results:
            logger.warning(f"{pk}: empty results — skipping")
            continue
        mat = build_layer_matrix(results)
        if layer_key not in mat:
            logger.warning(f"{pk}: layer {layer_key!r} not found — skipping")
            continue
        means[pk] = mat[layer_key]["X"].mean(axis=0).astype(np.float32)
    return means


def run_geometry_analysis(
    loaded_data: dict[str, dict],
    layer_key: str,
    out_dir: Path,
    variant_label: str = "base",
) -> dict:
    """
    Full geometry analysis for a set of conditions:
      1. Extract mean vectors
      2. Cosine similarity matrix + heatmap
      3. UMAP + PCA scatter plots

    Returns dict with mean_vectors, sim_matrix, labels.
    """
    from analytics.II1_persona_analytics.cosine_similarity import (
        cosine_similarity_matrix, plot_similarity_heatmap, sort_labels_by_gender_emotion,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    layer_slug = layer_key.replace("_D5120", "").replace("_D1408", "")

    # 1. Mean vectors
    means = get_condition_mean_vectors(loaded_data, layer_key)
    if len(means) < 2:
        logger.warning("Need at least 2 conditions for geometry analysis.")
        return {}
    logger.info(f"Geometry: {len(means)} conditions at {layer_key}")

    # 2. Cosine similarity
    sim, labels = cosine_similarity_matrix(means)
    fig = plot_similarity_heatmap(
        sim, labels,
        title=f"Condition cosine similarity @ {layer_slug} ({variant_label})",
        save_path=out_dir / f"cosine_sim_{layer_slug}.png",
    )
    plt.close(fig)
    np.save(out_dir / f"mean_vectors_{layer_slug}.npy", means)
    logger.info(f"Saved cosine similarity heatmap → {out_dir}")

    # 3. UMAP (if available) then PCA fallback
    sorted_labels = sort_labels_by_gender_emotion(labels)
    vecs  = np.stack([means[l] for l in sorted_labels]).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / np.where(norms < 1e-12, 1.0, norms)

    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=min(5, len(vecs_n)-1),
                                 random_state=42, metric="cosine")
        emb = reducer.fit_transform(vecs_n)
        method = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=42).fit_transform(vecs_n)
        method = "PCA"

    fig = _scatter_conditions(emb, sorted_labels,
                               title=f"{method} of condition mean vectors @ {layer_slug}\n({variant_label})",
                               method=method)
    fig.savefig(out_dir / f"scatter_{method.lower()}_{layer_slug}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {method} scatter → {out_dir}")

    # 4. Summary statistics
    sorted_idx = [labels.index(l) for l in sorted_labels]
    mat = sim[np.ix_(sorted_idx, sorted_idx)]
    n_female = sum(1 for l in sorted_labels if "female" in l)
    n_male   = len(sorted_labels) - n_female
    if n_female > 0 and n_male > 0:
        within_female = mat[:n_female, :n_female]
        within_male   = mat[n_female:, n_female:]
        cross          = mat[:n_female, n_female:]
        logger.info(
            f"Cosine stats — within-female: {within_female[~np.eye(n_female,dtype=bool)].mean():.3f}  "
            f"within-male: {within_male[~np.eye(n_male,dtype=bool)].mean():.3f}  "
            f"cross-gender: {cross.mean():.3f}"
        )

    return {
        "mean_vectors": means,
        "sim_matrix":   sim,
        "labels":       labels,
        "embedding":    emb,
        "embed_labels": sorted_labels,
        "embed_method": method,
    }


def _scatter_conditions(
    emb: np.ndarray,
    labels: list[str],
    title: str,
    method: str,
) -> plt.Figure:
    """Scatter plot where color = emotion, marker = gender."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, label in enumerate(labels):
        gender  = "female" if "female" in label else "male"
        emotion = next((e for e in EMOTION_COLORS if e in label), "unknown")
        color   = EMOTION_COLORS.get(emotion, "#888888")
        marker  = GENDER_MARKERS[gender]
        ax.scatter(emb[i, 0], emb[i, 1], c=color, marker=marker, s=120,
                   edgecolors="white", linewidths=0.8, zorder=3)
        short = label.replace("female_", "F_").replace("male_", "M_")
        ax.annotate(short, (emb[i, 0], emb[i, 1]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7.5)

    # Legend: emotion colors
    emo_patches = [mpatches.Patch(color=c, label=e.capitalize())
                   for e, c in EMOTION_COLORS.items()]
    gender_lines = [
        plt.Line2D([0], [0], marker="o", color="gray", ls="", ms=8, label="Female"),
        plt.Line2D([0], [0], marker="^", color="gray", ls="", ms=8, label="Male"),
    ]
    leg1 = ax.legend(handles=emo_patches, loc="upper left",  fontsize=8, title="Emotion")
    ax.add_artist(leg1)
    ax.legend(handles=gender_lines, loc="lower left", fontsize=8, title="Gender")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"{method} 1"); ax.set_ylabel(f"{method} 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig
