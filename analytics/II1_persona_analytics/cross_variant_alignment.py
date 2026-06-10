"""
II.1 Cross-variant vector alignment.

Answers the questions:
  - Is the gender concept vector the same across base / Germany / Nigeria?
  - Does adding country context shift or rotate the gender direction?
  - How do country vectors (Germany, Nigeria) relate to each other and to gender?
  - Are all feature vectors approximately orthogonal? (supports additive hypothesis)

Input: md_vectors directories from multiple representation discovery runs.
Output: pairwise cosine similarity matrix + visualisations.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)


def load_averaged_vector(
    md_dir: Path,
    layer_key: str,
    pattern: str = "*.npy",
) -> np.ndarray | None:
    """
    Load all .npy files matching pattern in md_dir, extract the vector at
    layer_key from each, average, and unit-normalise.

    Works for gender vectors (female_*_vs_male_*.npy) or country vectors
    (female_*_extended_*_vs_female_*.npy) — just pass the right pattern.
    """
    vecs = []
    for f in sorted(md_dir.glob(pattern)):
        try:
            d = np.load(f, allow_pickle=True).item()
            if layer_key in d:
                vecs.append(d[layer_key].astype(np.float32))
        except Exception as e:
            logger.warning(f"Could not load {f.name}: {e}")

    if not vecs:
        return None

    mean = np.mean(np.stack(vecs), axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm) if norm > 1e-12 else mean


def load_all_feature_vectors(
    discovery_dirs: dict[str, Path],
    layer_key: str,
) -> dict[str, np.ndarray]:
    """
    Load one averaged vector per named discovery run.

    Args:
        discovery_dirs: {label: path_to_md_vectors_dir}
            e.g. {
              "gender_base":    Path("results/representation_discovery/base/md_vectors"),
              "gender_germany": Path("results/representation_discovery/extended_germany_country/md_vectors"),
              "country_germany":Path("results/representation_discovery/extended_germany_country/md_vectors"),
            }
        layer_key: e.g. "language_29_D5120"

    Returns:
        {label: unit_vector}  — only labels where a vector was found
    """
    out: dict[str, np.ndarray] = {}
    for label, md_dir in discovery_dirs.items():
        if not md_dir.exists():
            logger.warning(f"[{label}] md_vectors dir not found: {md_dir}")
            continue
        v = load_averaged_vector(md_dir, layer_key)
        if v is not None:
            out[label] = v
            logger.info(f"  Loaded '{label}': {len(list(md_dir.glob('*.npy')))} vectors averaged")
        else:
            logger.warning(f"  [{label}] No vector found for layer {layer_key}")
    return out


def alignment_matrix(
    vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Pairwise cosine similarity matrix between all vectors."""
    labels = list(vectors.keys())
    mat = np.stack([vectors[l] for l in labels])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms < 1e-12, 1.0, norms)
    sim = mat @ mat.T
    return sim, labels


def plot_alignment_heatmap(
    sim: np.ndarray,
    labels: list[str],
    title: str = "Feature vector alignment",
    save_path: Path | None = None,
) -> plt.Figure:
    """Annotated heatmap of pairwise cosine similarity between feature vectors."""
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), max(5, len(labels) * 0.8)))
    sns.heatmap(
        sim, ax=ax,
        xticklabels=labels, yticklabels=labels,
        cmap="RdYlGn", vmin=-1, vmax=1,
        annot=True, fmt=".3f", annot_kws={"size": 9},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Cosine similarity"},
    )
    ax.set_title(title, fontsize=11)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def run_cross_variant_alignment(
    layer_key: str,
    results_root: Path,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Full cross-variant alignment analysis.

    Loads gender vectors from base/Germany/Nigeria discovery runs and country
    vectors from country-mode runs, computes pairwise cosine similarity, and
    saves the heatmap and a summary CSV.

    Returns a DataFrame with all pairwise similarities.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover which results exist
    discovery_dirs: dict[str, Path] = {}

    base_md = results_root / "base" / "md_vectors"
    if base_md.exists():
        discovery_dirs["gender_base"] = base_md

    for country in ["germany", "nigeria"]:
        # Gender vectors for country variant (female vs male, same country)
        gender_dir = results_root / f"extended_{country}_gender" / "md_vectors"
        if gender_dir.exists():
            discovery_dirs[f"gender_{country}"] = gender_dir

        # Country vectors (country variant vs base)
        country_dir = results_root / f"extended_{country}_country" / "md_vectors"
        if country_dir.exists():
            discovery_dirs[f"country_{country}"] = country_dir

    # Emotion vectors — anchor-free one-vs-rest contrasts: avg_{emotion} vs
    # avg_rest_{emotion} (pooled mean of the other seven emotions). Replaces an
    # earlier anchor-based design ("X vs contentment") that privileged one
    # emotion as a "neutral" reference — see get_emotion_vs_rest_contrasts().
    emotion_dir = results_root / "base_emotion" / "md_vectors"

    def _emotion_vector_files():
        return sorted(emotion_dir.glob("avg_*_vs_avg_rest_*.npy"))

    def _emotion_name_from_stem(stem: str) -> str:
        # stem == f"avg_{emotion}_vs_avg_rest_{emotion}"
        return stem[len("avg_"):].split("_vs_avg_rest_")[0]

    if emotion_dir.exists():
        # Load each emotion separately
        for f in _emotion_vector_files():
            emotion = _emotion_name_from_stem(f.stem)
            d = np.load(f, allow_pickle=True).item()
            if layer_key in d:
                discovery_dirs[f"emotion_{emotion}"] = emotion_dir

    if len(discovery_dirs) < 2:
        logger.warning("Need at least 2 feature vectors for alignment analysis.")
        return pd.DataFrame()

    logger.info(f"Loading {len(discovery_dirs)} feature vectors at {layer_key}")
    vectors = load_all_feature_vectors(discovery_dirs, layer_key)

    # For emotion vectors, load individually (not averaged across all emotions)
    emotion_vecs: dict[str, np.ndarray] = {}
    if emotion_dir.exists():
        for f in _emotion_vector_files():
            emotion = _emotion_name_from_stem(f.stem)
            d = np.load(f, allow_pickle=True).item()
            if layer_key in d:
                v = d[layer_key].astype(np.float32)
                norm = np.linalg.norm(v)
                emotion_vecs[f"emotion_{emotion}"] = v / norm if norm > 1e-12 else v
        # Remove the emotion_dir placeholder entries
        vectors = {k: v for k, v in vectors.items() if not k.startswith("emotion_")}
        vectors.update(emotion_vecs)

    if len(vectors) < 2:
        logger.warning("Not enough vectors loaded.")
        return pd.DataFrame()

    sim, labels = alignment_matrix(vectors)

    # Save heatmap
    fig = plot_alignment_heatmap(
        sim, labels,
        title=f"Feature vector alignment @ {layer_key.replace('_D5120','')}\n"
              f"(cosine similarity — 1.0 = same direction, 0 = orthogonal, -1 = opposite)",
        save_path=out_dir / f"feature_alignment_{layer_key.replace('_D5120','')}.png",
    )
    plt.close(fig)

    # Summary DataFrame
    rows = []
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i < j:
                rows.append({"vec_a": l1, "vec_b": l2, "cosine": round(float(sim[i, j]), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "feature_alignment.csv", index=False)

    # Print summary
    gender_vecs = {k: v for k, v in vectors.items() if k.startswith("gender")}
    if len(gender_vecs) > 1:
        g_labels = list(gender_vecs.keys())
        g_mat, g_labels = alignment_matrix(gender_vecs)
        print(f"\nGender vector alignment across variants:")
        for i, l1 in enumerate(g_labels):
            for j, l2 in enumerate(g_labels):
                if i < j:
                    print(f"  {l1} ↔ {l2}: {g_mat[i,j]:.4f}")

    logger.info(f"Alignment analysis saved → {out_dir}")
    return df
