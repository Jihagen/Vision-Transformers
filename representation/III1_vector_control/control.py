"""
III.1 Vector addition control experiment.

For a given concept vector v and target layer, inject h' = h + alpha * v
during inference and measure how the interestingness rating changes.

This tests causal sufficiency: does the found representation vector
actually drive model output, or is it merely correlated?

Procedure
---------
1. Load a set of test images and a fixed persona prompt.
2. For each alpha in a dose-response range (e.g. -2, -1, -0.5, 0, 0.5, 1, 2):
   a. Register an inject hook at the target layer.
   b. Run model_response on each image.
   c. Record the interestingness rating.
3. Compare rating distributions across alpha values.

Expected result for a valid gender concept vector:
  - alpha=0:        baseline ratings under the given persona
  - alpha > 0:      ratings shift toward female-persona ratings
  - alpha < 0:      ratings shift toward male-persona ratings
  - Monotonic shift: stronger alpha → larger shift (dose-response)

Output
------
Per-image, per-alpha rating table saved as control_results.csv.
Summary statistics + dose-response plot saved to the output directory.
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

LABEL_SCORE = {
    "Not Interesting":       1,
    "Slightly Interesting":  2,
    "Moderately Interesting":3,
    "Very Interesting":      4,
    "Extremely Interesting": 5,
}


@dataclass
class ControlResult:
    filename:      str
    alpha:         float
    layer_key:     str
    rating:        str
    score:         int
    explanation:   str = ""


def load_averaged_gender_vector(
    md_vectors_dir: str | Path,
    layer_key: str = "language_29_D5120",
) -> np.ndarray:
    """
    Load and average the per-emotion gender direction vectors at layer_key.

    The gender direction is defined as female - male; the averaged vector
    factors out emotion-specific residuals.

    Args:
        md_vectors_dir: directory containing female_*_vs_male_*.npy files
        layer_key:      target layer comp_key

    Returns:
        Unit-normalised averaged direction vector, shape (D,)
    """
    md_dir = Path(md_vectors_dir)
    vecs = []
    for f in sorted(md_dir.glob("female_*_vs_male_*.npy")):
        d = np.load(f, allow_pickle=True).item()
        if layer_key in d:
            vecs.append(d[layer_key])
        else:
            logger.warning(f"{f.name}: layer {layer_key!r} not found — skipping")

    if not vecs:
        raise FileNotFoundError(
            f"No gender vectors found for layer '{layer_key}' in {md_vectors_dir}"
        )
    mean_v = np.mean(np.stack(vecs), axis=0).astype(np.float32)
    norm   = np.linalg.norm(mean_v)
    if norm > 1e-12:
        mean_v = mean_v / norm
    logger.info(f"Averaged gender vector over {len(vecs)} emotions @ {layer_key} (norm=1)")
    return mean_v


def load_averaged_country_vector(
    md_vectors_dir: str | Path,
    layer_key: str,
) -> np.ndarray:
    """
    Average all per-condition '<cond>_extended_<country>_vs_<cond>.npy' vectors
    at layer_key (16 gender x emotion conditions), factoring out
    persona-specific residuals to isolate the country direction.

    Returns:
        Unit-normalised averaged direction vector, shape (D,)
    """
    md_dir = Path(md_vectors_dir)
    vecs = []
    for f in sorted(md_dir.glob("*.npy")):
        d = np.load(f, allow_pickle=True).item()
        if layer_key in d:
            vecs.append(d[layer_key])
        else:
            logger.warning(f"{f.name}: layer {layer_key!r} not found — skipping")

    if not vecs:
        raise FileNotFoundError(
            f"No country vectors found for layer '{layer_key}' in {md_vectors_dir}"
        )
    mean_v = np.mean(np.stack(vecs), axis=0).astype(np.float32)
    norm   = np.linalg.norm(mean_v)
    if norm > 1e-12:
        mean_v = mean_v / norm
    logger.info(f"Averaged country vector over {len(vecs)} conditions @ {layer_key} (norm=1)")
    return mean_v


def run_control_experiment(
    model,
    processor,
    images: list[dict],        # list of {"filename": str, "img_path": str}
    prompt: str,
    vector: np.ndarray,
    layer_key: str,
    alphas: list[float],
    checkpoint_every: int = 10,
    out_dir: str | Path | None = None,
) -> list[ControlResult]:
    """
    For each alpha, inject alpha*vector at layer_key and record ratings.

    Args:
        model / processor:  loaded HuggingFace model+processor
        images:             list of {"filename", "img_path"} dicts
        prompt:             persona prompt string (fixed for all images)
        vector:             unit-normalised concept direction, shape (D,)
        layer_key:          comp_key identifying the injection layer
                            (e.g. "language_29_D5120")
        alphas:             dose-response values (include 0 for baseline)
        checkpoint_every:   save intermediate CSV every N images
        out_dir:            directory for CSV output (optional)

    Returns:
        List of ControlResult objects.
    """
    from utils.hooks import HookState, register_inject_hooks
    from representation.I1_contrast_design.collect import model_response
    from utils.hpc import emergency_cleanup

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Map comp_key (e.g. language_29_D5120) to the hook name the inject hook uses.
    # register_inject_hooks uses names like "llm_layer_29_rating_token".
    hook_name = _comp_key_to_hook_name(layer_key)
    logger.info(f"Injection target: {layer_key} → hook '{hook_name}'")

    results: list[ControlResult] = []

    for alpha in alphas:
        logger.info(f"alpha={alpha:+.1f} — {len(images)} images")
        state = HookState()
        if abs(alpha) > 1e-9:
            register_inject_hooks(model, state, {hook_name: (vector, float(alpha))})
        # alpha=0: no injection; hooks fire but injection_vectors is empty

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
                score  = LABEL_SCORE.get(rating, 0)
                results.append(ControlResult(
                    filename=filename,
                    alpha=alpha,
                    layer_key=layer_key,
                    rating=rating,
                    score=score,
                    explanation=data.get("explanation", ""),
                ))
            except Exception as e:
                logger.warning(f"  [{alpha:+.1f}] {filename}: {e} — skipping")
                emergency_cleanup(embeddings_dict=state.embeddings)

            if out_dir and idx % checkpoint_every == 0:
                _save_results(results, out_dir / "control_results_partial.csv")

        state.remove_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if out_dir:
        _save_results(results, out_dir / "control_results.csv")
        logger.info(f"Control results saved → {out_dir / 'control_results.csv'}")

    return results


def _comp_key_to_hook_name(layer_key: str) -> str:
    """
    Convert a comp_key like 'language_29_D5120' to the hook name
    registered by register_inject_hooks: 'llm_layer_29_rating_token'.
    """
    import re
    m = re.search(r"language_(\d+)", layer_key)
    if m:
        return f"llm_layer_{m.group(1)}_rating_token"
    m = re.search(r"vision_(\d+)", layer_key)
    if m:
        return f"vision_layer_{m.group(1)}_cls"
    raise ValueError(f"Cannot map layer_key '{layer_key}' to a hook name.")


def _save_results(results: list[ControlResult], path: Path) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "alpha", "layer_key", "rating", "score", "explanation"])
        for r in results:
            w.writerow([r.filename, r.alpha, r.layer_key, r.rating, r.score, r.explanation])


def summarise_and_plot(
    results: list[ControlResult],
    baseline_alpha: float = 0.0,
    out_dir: str | Path | None = None,
) -> "pd.DataFrame":
    """
    Compute per-alpha mean score and plot dose-response curve.

    Returns a summary DataFrame with columns [alpha, mean_score, std_score, n].
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame([
        {"filename": r.filename, "alpha": r.alpha,
         "score": r.score, "rating": r.rating}
        for r in results
    ])

    summary = (df.groupby("alpha")["score"]
               .agg(mean_score="mean", std_score="std", n="count")
               .reset_index().round(4))

    print("\nDose-response summary:")
    print(summary.to_string(index=False))

    # Rating distribution per alpha
    dist = df.groupby(["alpha", "rating"]).size().unstack(fill_value=0)
    dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.errorbar(summary.alpha, summary.mean_score, yerr=summary.std_score,
                fmt="o-", color="steelblue", lw=2, capsize=4, ms=7)
    ax.axvline(baseline_alpha, color="gray", ls="--", lw=1, alpha=0.6, label="baseline")
    ax.set_xlabel("Injection alpha"); ax.set_ylabel("Mean interestingness score (1–5)")
    ax.set_title("Dose-response: mean rating vs injection strength")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    label_order = list(LABEL_SCORE.keys())
    colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]
    dist_pct.reindex(columns=[c for c in label_order if c in dist_pct.columns])\
             .plot(kind="bar", stacked=True, ax=ax, color=colors[:len(dist_pct.columns)],
                   width=0.7, legend=True)
    ax.set_xlabel("Alpha"); ax.set_ylabel("% images")
    ax.set_title("Rating distribution per alpha")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=0)

    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "dose_response.png", dpi=150, bbox_inches="tight")
    plt.show()

    return summary
