"""
III. Lexicon-based causal validation for concept vectors (Branch B / Branch C).

Shared utilities for injection/ablation experiments whose causal signature is
measured via a logit-lens-derived lexicon-count shift in generated
explanations (see logit_lens.py: derive_lexicon / count_lexicon_occurrences),
alongside the interestingness rating distribution (1-5, the same output
variable for every experiment regardless of which vector is injected).

Used by:
    - Branch B (run_interest_validation): v_interest_blank @ language_29,
      lexicon = high/low-interest words.
    - Branch C (run_country_validation):  v_country_nigeria @ language_23,
      lexicon = Nigeria/African country-name words.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_lexicon(json_path: str | Path) -> tuple[list[str], list[str]]:
    """Load (positive_words, negative_words) persisted by run_logit_lens_evidence.py."""
    with open(json_path) as f:
        d = json.load(f)
    lex = d["lexicon"]
    return lex["positive_words"], lex["negative_words"]


def summarise_lexicon_results(
    results,  # list[BlankResult] from run_blank_conditions / load_blank_baseline
    lexicon: tuple[list[str], list[str]],
    out_dir: str | Path | None = None,
    title: str = "Blank-prompt condition summary",
) -> "pd.DataFrame":
    """
    Per-condition summary: n, mean rating score, mean count of logit-lens
    derived positive/negative-direction words in the generated explanation,
    and their difference.

    Plots: rating distribution per condition, lexicon-count difference per
    condition.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from control.III1_vector_control.control import LABEL_SCORE
    from control.III1_vector_control.logit_lens import count_lexicon_occurrences

    pos_words, neg_words = lexicon
    rows = []
    for r in results:
        n_pos, n_neg = count_lexicon_occurrences(r.explanation, pos_words, neg_words)
        rows.append({
            "filename": r.filename, "condition": r.condition,
            "rating": r.rating, "score": r.score,
            "n_pos_words": n_pos, "n_neg_words": n_neg,
            "lexicon_diff": n_pos - n_neg,
        })
    df = pd.DataFrame(rows)

    summary = (df.groupby("condition")
                 .agg(score_mean=("score", "mean"), score_std=("score", "std"),
                      score_count=("score", "count"),
                      n_pos_words_mean=("n_pos_words", "mean"),
                      n_neg_words_mean=("n_neg_words", "mean"),
                      lexicon_diff_mean=("lexicon_diff", "mean"))
                 .round(4).reset_index())

    print(f"\n{title}:")
    print(summary.to_string(index=False))

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
    summary.plot(x="condition", y="lexicon_diff_mean", kind="bar", ax=ax,
                  color="steelblue", legend=False)
    ax.axhline(0, color="gray", lw=1)
    ax.set_ylabel("mean(n_pos_words - n_neg_words)")
    ax.set_title("Lexicon-count difference per condition")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "lexicon_validation_per_image.csv", index=False)
        summary.to_csv(out_dir / "lexicon_validation_summary.csv", index=False)
        fig.savefig(out_dir / "lexicon_validation.png", dpi=150, bbox_inches="tight")
    plt.show()

    return summary


def lexicon_count_by_alpha(
    results,  # list[ControlResult] from run_control_experiment
    lexicon: tuple[list[str], list[str]],
    out_path: str | Path | None = None,
) -> "pd.DataFrame":
    """
    Per-alpha mean/std/count of logit-lens derived positive/negative-direction
    word counts in the generated explanation (alpha-sweep dose-response measure).
    """
    import pandas as pd
    from control.III1_vector_control.logit_lens import count_lexicon_occurrences

    pos_words, neg_words = lexicon
    df = pd.DataFrame([{"alpha": r.alpha, "explanation": r.explanation} for r in results])
    counts = df["explanation"].apply(lambda t: count_lexicon_occurrences(t, pos_words, neg_words))
    df["n_pos_words"] = counts.apply(lambda c: c[0])
    df["n_neg_words"] = counts.apply(lambda c: c[1])
    df["lexicon_diff"] = df["n_pos_words"] - df["n_neg_words"]
    by_alpha = (df.groupby("alpha")[["n_pos_words", "n_neg_words", "lexicon_diff"]]
                .agg(["mean", "std", "count"]).round(4))
    print("\nLexicon-count by alpha:")
    print(by_alpha.to_string())
    if out_path:
        by_alpha.to_csv(out_path)
    return by_alpha
