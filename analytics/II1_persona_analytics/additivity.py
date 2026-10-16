"""
II.1 Additivity / compositionality check for persona feature vectors.

Question: is a compound persona (e.g. female + angry + German) approximately
equal to the sum of its individual feature directions?

For each compound condition C = (gender g, emotion e, country c):
  predicted_shift = v_gender[g] + v_emotion[e] + v_country[c]
  actual_shift    = h(C) - h(base_avg)

  additivity_score = cosine(actual_shift, predicted_shift)

A high score (> 0.8) supports the linear representation hypothesis for personas.
A low score means the features interact non-linearly or occupy a subspace.

Usage
-----
This module is called by runners/run_analytics.py --additivity.
Requires:
  - mean vectors for all conditions in the base variant
  - mean vectors for all conditions in country variant(s)
  - gender vectors (from I.2 output, averaged across emotions)
  - country vectors (from I.2 output on country-vs-base contrasts)
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def compute_averaged_vector(
    vectors_dict: dict[str, np.ndarray],
    keys: list[str],
) -> np.ndarray:
    """Average and unit-normalise a set of vectors."""
    vecs = np.stack([vectors_dict[k] for k in keys if k in vectors_dict])
    mean = vecs.mean(axis=0)
    return _unit(mean)


def derive_feature_vectors(
    base_means: dict[str, np.ndarray],      # {persona_key: mean_vec} base variant
    country_means: dict[str, np.ndarray],   # {persona_key: mean_vec} country variant
    gender_md_vectors: dict[str, np.ndarray],  # {contrast_name: vec} from I.2
    country_md_vectors: dict[str, np.ndarray], # {contrast_name: vec} from I.2
    layer_key: str,
) -> dict[str, np.ndarray]:
    """
    Derive clean single-feature direction vectors by averaging over nuisance variables.

    Returns:
        {
          "gender":  unit vector pointing female-ward (averaged over emotions)
          "emotion_{e}": unit vector for each emotion (averaged over genders)
          "country_{c}": unit vector for country (averaged over all conditions)
        }
    """
    from runners.experiment_definitions import EMOTIONS, GENDERS

    feature_vecs: dict[str, np.ndarray] = {}

    # Gender direction: average gender contrast vectors across all emotions
    gender_keys = [k for k in gender_md_vectors if "female" in k and "vs_male" in k]
    if gender_keys:
        feature_vecs["gender"] = compute_averaged_vector(gender_md_vectors, gender_keys)
        logger.info(f"Gender vector: averaged over {len(gender_keys)} emotion contrasts")

    # Emotion directions: for each emotion, average h(female_e + male_e) - h(grand_mean)
    grand_mean = np.mean(np.stack(list(base_means.values())), axis=0)
    for e in EMOTIONS:
        emo_keys = [k for k in base_means if e in k]
        if emo_keys:
            emo_mean = np.mean(np.stack([base_means[k] for k in emo_keys]), axis=0)
            feature_vecs[f"emotion_{e}"] = _unit(emo_mean - grand_mean)

    # Country direction: mean(country_conditions) - mean(base_conditions) at layer
    if country_means:
        country_shift = np.mean(np.stack(list(country_means.values())), axis=0) - grand_mean
        country_name  = next(
            (k.split("_extended_")[1] if "_extended_" in k else "country"
             for k in country_means), "country"
        )
        feature_vecs[f"country_{country_name}"] = _unit(country_shift)
        logger.info(f"Country vector '{country_name}' derived from {len(country_means)} conditions")

    return feature_vecs


def check_additivity(
    base_means: dict[str, np.ndarray],
    country_means: dict[str, np.ndarray],
    feature_vecs: dict[str, np.ndarray],
    out_dir: Path,
) -> pd.DataFrame:
    """
    For each compound condition, compare actual shift from base_avg to the
    predicted shift from summing individual feature vectors.

    Returns a DataFrame with one row per condition, with additivity_score column.
    """
    from runners.experiment_definitions import EMOTIONS, GENDERS

    grand_mean = np.mean(np.stack(list(base_means.values())), axis=0)
    rows = []

    for pk, h_compound in country_means.items():
        # Identify features
        gender  = "female" if "female" in pk else "male"
        emotion = next((e for e in EMOTIONS if e in pk), None)
        # Determine country from key
        country = next(
            (c.lower() for c in ["germany", "nigeria"] if c.lower() in pk.lower()), None
        )

        actual_shift = h_compound - grand_mean

        # Build predicted shift
        parts: list[np.ndarray] = []
        if "gender" in feature_vecs and gender == "female":
            parts.append(feature_vecs["gender"])
        elif "gender" in feature_vecs and gender == "male":
            parts.append(-feature_vecs["gender"])
        if emotion and f"emotion_{emotion}" in feature_vecs:
            parts.append(feature_vecs[f"emotion_{emotion}"])
        country_key = f"country_{country}" if country else None
        if country_key and country_key in feature_vecs:
            parts.append(feature_vecs[country_key])

        if not parts:
            continue

        predicted_shift = np.sum(np.stack(parts), axis=0)
        score = cosine(actual_shift, predicted_shift)
        residual_cos = cosine(
            actual_shift - predicted_shift,
            base_means.get(pk.replace(f"_{country}", "").replace("extended_", ""), grand_mean) - grand_mean
        ) if parts else None

        rows.append({
            "condition":         pk,
            "gender":            gender,
            "emotion":           emotion,
            "country":           country,
            "additivity_score":  round(score, 4),
            "n_features_summed": len(parts),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No additivity rows computed — check that country_means has data.")
        return df

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "additivity_scores.csv", index=False)

    logger.info(
        f"Additivity scores: mean={df.additivity_score.mean():.3f}  "
        f"min={df.additivity_score.min():.3f}  max={df.additivity_score.max():.3f}"
    )

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(df.additivity_score, bins=20, color="steelblue", edgecolor="white")
    axes[0].axvline(0.8, color="red", ls="--", lw=1.5, label="0.80 threshold")
    axes[0].set_title("Additivity score distribution"); axes[0].set_xlabel("Cosine(actual, predicted)")
    axes[0].legend()

    pivot = df.pivot_table(index="emotion", columns="gender", values="additivity_score")
    import seaborn as sns
    sns.heatmap(pivot, ax=axes[1], cmap="RdYlGn", vmin=0, vmax=1,
                annot=True, fmt=".3f", cbar_kws={"label": "Additivity score"})
    axes[1].set_title("Additivity by emotion × gender")
    plt.tight_layout()
    fig.savefig(out_dir / "additivity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df


# ── Raw-scale staged composition (results/EX1_T2_additivity.ipynb) ────────────
#
# The functions above sum unit-normalised vectors. The staged re-derivation in
# the additivity notebook replaced that with naturally scaled mean differences
# and a halved gender step; the functions below are that convention as plain
# numpy, so that the notebook and scripts/export_web_data.py share one definition.

COMPOSITION_FEATURES = ("gender", "emotion", "country")
COMPOSITION_COEFFICIENTS = {"gender": 0.5, "emotion": 1.0, "country": 1.0}


def derive_raw_feature_vectors(
    base_means: dict[str, np.ndarray],                 # {"female_anger": vec, ...}
    country_means: dict[str, dict[str, np.ndarray]],   # {"Germany": {"female_anger_extended_germany": vec}}
    emotions: list[str],
    genders: tuple[str, str] = ("female", "male"),
) -> dict:
    """
    Feature vectors at their natural (not unit-normalised) scale.

    Returns:
        {
          "grand_mean": mean of the base condition means,
          "gender":     mean(female base conditions) - mean(male base conditions),
          "emotion":    {e: mean(base conditions of emotion e) - grand_mean},
          "country":    {c: mean over gender x emotion of (h_country - h_matching_base)},
        }
    """
    f64 = lambda v: np.asarray(v, dtype=np.float64)
    grand_mean = np.mean(np.stack([f64(v) for v in base_means.values()]), axis=0)
    by_gender = {g: np.mean(np.stack([f64(v) for k, v in base_means.items()
                                      if k.startswith(f"{g}_")]), axis=0) for g in genders}
    v_emotion = {e: np.mean(np.stack([f64(base_means[f"{g}_{e}"]) for g in genders]), axis=0)
                    - grand_mean for e in emotions}
    v_country = {}
    for country, means in country_means.items():
        suffix = f"_extended_{country.lower()}"
        v_country[country] = np.mean(np.stack(
            [f64(means[f"{g}_{e}{suffix}"]) - f64(base_means[f"{g}_{e}"])
             for g in genders for e in emotions]), axis=0)
    return {"grand_mean": grand_mean, "gender": by_gender[genders[0]] - by_gender[genders[1]],
            "emotion": v_emotion, "country": v_country}


def composition_components(
    feature_vecs: dict, gender: str, emotion: str, country: str,
) -> dict[str, np.ndarray]:
    """The three steps of one compound persona: +-1/2 v_gender, v_emotion, v_country."""
    sign = 1.0 if gender == "female" else -1.0
    return {
        "gender": sign * COMPOSITION_COEFFICIENTS["gender"] * feature_vecs["gender"],
        "emotion": feature_vecs["emotion"][emotion],
        "country": feature_vecs["country"][country],
    }


def shift_cosine(predicted: np.ndarray, observed: np.ndarray, origin: np.ndarray) -> float:
    """cos(predicted - origin, observed - origin); 0.0 if either shift is zero."""
    a, b = predicted - origin, observed - origin
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0
