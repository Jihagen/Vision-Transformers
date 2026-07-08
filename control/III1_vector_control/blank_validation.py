"""
III. Blank-prompt gender-vector validation (primary, confound-free).

Runs the blank (no-persona) prompt under 3 GPU conditions at a target layer
(default language_29_D5120):

  inject_pos2  — h' = h + 2*v  ("toward female")
  inject_neg2  — h' = h - 2*v  ("toward male")
  ablate       — h' = h - (h . v_hat) v_hat  (project out the gender direction)

A 4th condition, no_inject (no intervention), is NOT generated here — it is
read from the existing 500-sample blank collection
(data/results_blank_activations.npy), restricted to the same images, via
load_blank_baseline().

Primary measure: pronoun counts (count_pronouns). This operationalises an a
priori prediction from logit_lens.py: the gender vector's +v direction
projects onto "her/she/herself" (+ "husband"/"boyfriend"), -v onto
"his/he/himself" (+ "wife"/"wives"). The blank prompt has no "Gender: X"
field to echo, so any shift in pronoun usage between conditions must
originate from the injected/ablated activation itself.

Secondary measures: rating distribution (causal potency) and, if a gender
probe is supplied, its TF-IDF score (exploratory).
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_FEMALE_PRONOUNS = re.compile(r"\b(she|her|hers|herself)\b", re.I)
_MALE_PRONOUNS   = re.compile(r"\b(he|him|his|himself)\b", re.I)


def count_pronouns(text: str) -> tuple[int, int]:
    """Return (n_female_pronouns, n_male_pronouns) in text (word-boundary, case-insensitive)."""
    return len(_FEMALE_PRONOUNS.findall(text)), len(_MALE_PRONOUNS.findall(text))


@dataclass
class BlankResult:
    filename:    str
    condition:   str
    rating:      str
    score:       int
    explanation: str


def load_blank_baseline(
    images: list[dict],
    blank_path: str | Path = "data/results_blank_activations.npy",
) -> list[BlankResult]:
    """
    Load the no_inject baseline condition from the existing 500-sample blank
    collection, restricted to the given images (matched by filename).
    """
    from control.III1_vector_control.control import LABEL_SCORE

    obj = np.load(blank_path, allow_pickle=True).item()
    by_filename = {r["filename"]: r for r in obj["results"]}

    results = []
    missing = []
    for img in images:
        fn = img["filename"]
        if fn not in by_filename:
            missing.append(fn)
            continue
        r = by_filename[fn]
        rating = r.get("interestingness", r.get("interestingness_label", "?"))
        results.append(BlankResult(
            filename=fn,
            condition="no_inject",
            rating=rating,
            score=LABEL_SCORE.get(rating, 0),
            explanation=r.get("explanation", ""),
        ))
    if missing:
        logger.warning(f"no_inject baseline: {len(missing)} images missing from {blank_path}")
    logger.info(f"no_inject baseline: {len(results)} results loaded from existing blank collection")
    return results


def run_blank_conditions(
    model,
    processor,
    images: list[dict],            # [{"filename": str, "img_path": str}]
    vector: np.ndarray,             # unit-normalised direction, shape (D,)
    layer_key: str,
    inject_alphas: dict[str, float] = {"inject_pos2": 2.0, "inject_neg2": -2.0},
    checkpoint_every: int = 10,
    out_dir: str | Path | None = None,
) -> list[BlankResult]:
    """
    Run the blank prompt under each injection condition + an ablation
    condition, recording rating + explanation per image.

    Returns a flat list of BlankResult (does NOT include no_inject — see
    load_blank_baseline()).
    """
    from utils.hooks import HookState, register_inject_hooks, register_projection_removal_hooks
    from representation.I1_contrast_design.collect import model_response
    from control.III1_vector_control.control import LABEL_SCORE, _comp_key_to_hook_name
    from utils.prompt_builder import build_blank_prompt
    from utils.hpc import emergency_cleanup

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_blank_prompt()
    hook_name = _comp_key_to_hook_name(layer_key)
    logger.info(f"Blank-prompt target: {layer_key} -> hook '{hook_name}'")

    conditions: list[tuple[str, str, float | None]] = [
        (name, "inject", alpha) for name, alpha in inject_alphas.items()
    ]
    conditions.append(("ablate", "ablate", None))

    results: list[BlankResult] = []

    for cond_name, kind, alpha in conditions:
        logger.info(f"Condition '{cond_name}' ({kind}, alpha={alpha}) — {len(images)} images")
        state = HookState()
        if kind == "inject":
            register_inject_hooks(model, state, {hook_name: (vector, float(alpha))})
        else:
            register_projection_removal_hooks(model, state, {hook_name: vector})

        for idx, img in enumerate(images, 1):
            img_path = img["img_path"]
            filename = img["filename"]
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                state.reset_embeddings()
                _, data = model_response(
                    prompt, img_path, model, processor, state,
                    image_max_size=800, max_new_tokens=64,
                )
                rating = data.get("interestingness", "?")
                results.append(BlankResult(
                    filename=filename,
                    condition=cond_name,
                    rating=rating,
                    score=LABEL_SCORE.get(rating, 0),
                    explanation=data.get("explanation", ""),
                ))
            except Exception as e:
                logger.warning(f"  [{cond_name}] {filename}: {e} — skipping")
                emergency_cleanup(embeddings_dict=state.embeddings)

            if out_dir and idx % checkpoint_every == 0:
                _save_blank_results(results, out_dir / "blank_results_partial.csv")

        state.remove_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if out_dir:
        _save_blank_results(results, out_dir / "blank_results.csv")
        logger.info(f"Blank condition results saved -> {out_dir / 'blank_results.csv'}")

    return results


def _save_blank_results(results: list[BlankResult], path: Path) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "condition", "rating", "score", "explanation"])
        for r in results:
            w.writerow([r.filename, r.condition, r.rating, r.score, r.explanation])


def summarise_blank_results(
    results: list[BlankResult],
    gender_probe=None,
    out_dir: str | Path | None = None,
) -> "pd.DataFrame":
    """
    Per-condition summary: n, mean rating score, mean female/male pronoun
    counts and their difference (primary measure), and (if gender_probe is
    given) mean TF-IDF gender-probe score (exploratory).

    Plots: rating distribution per condition, pronoun-count difference per
    condition.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from control.III1_vector_control.control import LABEL_SCORE

    rows = []
    for r in results:
        n_f, n_m = count_pronouns(r.explanation)
        row = {
            "filename": r.filename, "condition": r.condition,
            "rating": r.rating, "score": r.score,
            "n_female_pronouns": n_f, "n_male_pronouns": n_m,
            "pronoun_diff": n_f - n_m,
        }
        rows.append(row)
    df = pd.DataFrame(rows)

    if gender_probe is not None:
        from control.III1_vector_control.gender_probe import score_explanations
        explanations = [r.explanation for r in results]
        df["tfidf_gender_score"] = score_explanations(gender_probe, explanations)

    agg = {"score": ["mean", "std", "count"],
           "n_female_pronouns": "mean", "n_male_pronouns": "mean",
           "pronoun_diff": "mean"}
    if "tfidf_gender_score" in df.columns:
        agg["tfidf_gender_score"] = "mean"

    summary = df.groupby("condition").agg(agg).round(4)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()

    print("\nBlank-prompt condition summary:")
    print(summary.to_string(index=False))

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    label_order = list(LABEL_SCORE.keys())
    colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]
    dist = df.groupby(["condition", "rating"]).size().unstack(fill_value=0)
    dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100
    dist_pct.reindex(columns=[c for c in label_order if c in dist_pct.columns]) \
            .plot(kind="bar", stacked=True, ax=ax, color=colors[:len(dist_pct.columns)],
                  width=0.7, legend=True)
    ax.set_xlabel("Condition"); ax.set_ylabel("% images")
    ax.set_title("Rating distribution per condition")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    ax = axes[1]
    summary.plot(x="condition", y="pronoun_diff_mean", kind="bar", ax=ax,
                  color="steelblue", legend=False)
    ax.axhline(0, color="gray", lw=1)
    ax.set_ylabel("mean(n_female_pronouns - n_male_pronouns)")
    ax.set_title("Pronoun-count difference per condition\n(predicted: inject_pos2>0, inject_neg2<0)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "blank_validation_per_image.csv", index=False)
        summary.to_csv(out_dir / "blank_validation_summary.csv", index=False)
        fig.savefig(out_dir / "blank_validation.png", dpi=150, bbox_inches="tight")
    plt.show()

    return summary
