"""
Pairwise cosine similarity between persona-feature vectors.

Answers: do conditions form meaningful distance structures?
Is gender represented as a binary contrast or a gradient?

Input: concept direction vectors (from I.2 or I.3) OR mean activation vectors
       per condition extracted via condition_geometry.py.
Output: similarity matrix + annotated heatmap.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def cosine_similarity_matrix(
    vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """
    Compute pairwise cosine similarities between all vectors.

    Returns:
        (sim_matrix, ordered_labels)  — sim_matrix shape (K, K)
    """
    labels = list(vectors.keys())
    vecs   = np.stack([vectors[l] for l in labels]).astype(np.float32)
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms  = np.where(norms < 1e-12, 1.0, norms)
    vecs   = vecs / norms
    sim    = vecs @ vecs.T
    return sim, labels


def sort_labels_by_gender_emotion(labels: list[str]) -> list[str]:
    """
    Sort condition labels so all female conditions come first, then male,
    and within each gender emotions follow EMOTIONS order.
    """
    from runners.experiment_definitions import EMOTIONS
    order = {e: i for i, e in enumerate(EMOTIONS)}

    def key(l: str):
        gender = 0 if "female" in l else 1
        for e in EMOTIONS:
            if e in l:
                return (gender, order[e])
        return (gender, 99)

    return sorted(labels, key=key)


def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    labels: list[str],
    title: str = "",
    save_path: str | Path | None = None,
    annotate: bool = True,
) -> plt.Figure:
    """Annotated heatmap of cosine similarity, sorted by gender then emotion."""
    sorted_labels = sort_labels_by_gender_emotion(labels)
    idx = [labels.index(l) for l in sorted_labels]
    mat = sim_matrix[np.ix_(idx, idx)]

    short = [l.replace("female_", "F_").replace("male_", "M_")
               .replace("_extended", "").split("_D")[0] for l in sorted_labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.65),
                                    max(7, len(labels) * 0.55)))
    sns.heatmap(
        mat, ax=ax,
        xticklabels=short, yticklabels=short,
        cmap="RdYlGn", vmin=-1, vmax=1,
        annot=annotate and len(labels) <= 24,
        fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Cosine similarity"},
    )
    # Draw gender separator line
    n_female = sum(1 for l in sorted_labels if "female" in l)
    if 0 < n_female < len(sorted_labels):
        ax.axhline(n_female, color="navy", lw=2)
        ax.axvline(n_female, color="navy", lw=2)

    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
