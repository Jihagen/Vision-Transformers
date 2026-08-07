"""
Post-hoc analysis for Generalisation Level 1: turns predictions.csv into
summary.csv (spec section 7), task-specific dataset-reference/ground-truth
metrics, the SMID arousal analysis (section 2B), plots, and a cross-task
normalized-effect matrix for merging with the original interestingness/
relevance results (results/attack_eval_projected/).
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from attack.IV2_generalisation.metrics import condition_metrics, budget_response
from attack.IV2_generalisation.tasks import TaskSpec

logger = logging.getLogger(__name__)

_SUMMARY_FRONT_COLS = ["attack", "epsilon", "n", "clean_mean", "perturbed_mean",
                        "mean_signed_shift", "mean_abs_shift", "normalized_signed_shift",
                        "pct_up", "pct_unchanged", "pct_down", "wasserstein"]


def build_summary(predictions: pd.DataFrame, task: TaskSpec) -> pd.DataFrame:
    """Core attack x epsilon summary table (spec section 7)."""
    rows = []
    attacked = predictions[predictions["attack"] != "clean"]
    for (attack, eps), g in attacked.groupby(["attack", "epsilon"]):
        g_ok = g.dropna(subset=["delta"]).rename(columns={"clean_model_label": "clean_label"})
        m = condition_metrics(g_ok, task.valid_labels, task.scale_width)
        m["attack"] = attack
        m["epsilon"] = eps
        m["n_total_rows"] = len(g)
        m["n_parse_ok"] = int(g["parse_ok"].sum())
        rows.append(m)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    other = [c for c in summary.columns if c not in _SUMMARY_FRONT_COLS]
    return summary[_SUMMARY_FRONT_COLS + other].sort_values(["attack", "epsilon"]).reset_index(drop=True)


# ── Task 1: shopping — dataset-reference secondary metrics ────────────────

def add_shopping_reference_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["model_reference_error"] = df["model_label"] - df["reference_relevance_5"]
    clean_err = (df[df["attack"] == "clean"]
                 .set_index("sample_id")["model_reference_error"])
    df["clean_reference_error"] = df["sample_id"].map(clean_err)
    df["delta_reference_error"] = df["model_reference_error"] - df["clean_reference_error"]
    return df


# ── Task 3: MEDIC — ground-truth disagreement metrics ──────────────────────

def add_medic_gt_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["severity_error_signed"] = df["model_label"] - df["gt_damage_severity"]
    df["severity_error_absolute"] = df["severity_error_signed"].abs()
    clean_signed = (df[df["attack"] == "clean"].set_index("sample_id")["severity_error_signed"])
    clean_abs = (df[df["attack"] == "clean"].set_index("sample_id")["severity_error_absolute"])
    df["clean_signed_error"] = df["sample_id"].map(clean_signed)
    df["clean_absolute_error"] = df["sample_id"].map(clean_abs)
    df["delta_signed_error"] = df["severity_error_signed"] - df["clean_signed_error"]
    df["delta_absolute_error"] = df["severity_error_absolute"] - df["clean_absolute_error"]
    return df


def medic_gt_summary(predictions_with_gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    attacked = predictions_with_gt[predictions_with_gt["attack"] != "clean"]
    for (attack, eps), g in attacked.groupby(["attack", "epsilon"]):
        g_ok = g.dropna(subset=["delta"])
        if g_ok.empty:
            continue
        exact_acc = float((g_ok["model_label"] == g_ok["gt_damage_severity"]).mean())
        one_step = float((g_ok["severity_error_absolute"] == 1).mean())
        two_step = float((g_ok["severity_error_absolute"] == 2).mean())
        rows.append({
            "attack": attack, "epsilon": eps, "n": len(g_ok),
            "exact_accuracy": exact_acc,
            "one_step_error_rate": one_step,
            "two_step_error_rate": two_step,
            "mean_delta_signed_error": float(g_ok["delta_signed_error"].mean()),
            "mean_delta_absolute_error": float(g_ok["delta_absolute_error"].mean()),
            "pct_attack_induced_up": float((g_ok["delta"] > 0).mean()),
            "pct_attack_induced_down": float((g_ok["delta"] < 0).mean()),
        })
    return pd.DataFrame(rows).sort_values(["attack", "epsilon"]).reset_index(drop=True)


# ── Task 2: SMID — human-disagreement + arousal analysis (section 2B) ─────

def add_smid_bias_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["moral_bias"] = df["model_label"] - df["human_morality"]
    df["absolute_moral_error"] = df["moral_bias"].abs()
    clean_bias = (df[df["attack"] == "clean"].set_index("sample_id")["moral_bias"])
    clean_abs = (df[df["attack"] == "clean"].set_index("sample_id")["absolute_moral_error"])
    df["clean_moral_bias"] = df["sample_id"].map(clean_bias)
    df["clean_absolute_moral_error"] = df["sample_id"].map(clean_abs)
    df["delta_moral_bias"] = df["moral_bias"] - df["clean_moral_bias"]
    df["delta_absolute_moral_error"] = df["absolute_moral_error"] - df["clean_absolute_moral_error"]
    return df


def smid_arousal_analysis(predictions_with_bias: pd.DataFrame) -> pd.DataFrame:
    """
    Per attack x epsilon: correlation(human_arousal, delta_morality) and
    correlation(human_arousal, |delta_morality|), plus a
    delta_morality ~ arousal * epsilon (+ clean_morality covariate) OLS fit
    pooling all epsilons for that attack (spec section 2B minimum model).
    Requires statsmodels; degrades gracefully (NaN + warning) if unavailable.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        smf = None
        logger.warning("statsmodels not available — skipping delta_morality ~ arousal*epsilon regression")

    rows = []
    attacked = predictions_with_bias[predictions_with_bias["attack"] != "clean"]
    for (attack, eps), g in attacked.groupby(["attack", "epsilon"]):
        g_ok = g.dropna(subset=["delta", "human_arousal"])
        if len(g_ok) < 3:
            continue
        r_signed, p_signed = pearsonr(g_ok["human_arousal"], g_ok["delta"])
        r_abs, p_abs = pearsonr(g_ok["human_arousal"], g_ok["delta"].abs())
        rows.append({
            "attack": attack, "epsilon": eps, "n": len(g_ok),
            "corr_arousal_vs_delta_morality": r_signed, "p_arousal_vs_delta_morality": p_signed,
            "corr_arousal_vs_abs_delta_morality": r_abs, "p_arousal_vs_abs_delta_morality": p_abs,
        })
    corr_df = pd.DataFrame(rows).sort_values(["attack", "epsilon"]).reset_index(drop=True)

    if smf is None:
        return corr_df

    reg_rows = []
    for attack, g in attacked.groupby("attack"):
        g_ok = g.dropna(subset=["delta", "human_arousal", "clean_model_label"]).copy()
        if g_ok["epsilon"].nunique() < 2 or len(g_ok) < 10:
            continue
        g_ok = g_ok.rename(columns={"delta": "delta_morality", "clean_model_label": "clean_morality"})
        try:
            model = smf.ols(
                "delta_morality ~ human_arousal * epsilon + clean_morality", data=g_ok
            ).fit()
            for term, coef in model.params.items():
                reg_rows.append({
                    "attack": attack, "term": term, "coef": coef,
                    "p_value": model.pvalues[term], "n": len(g_ok), "r_squared": model.rsquared,
                })
        except Exception as e:
            logger.warning(f"  OLS fit failed for attack={attack}: {e}")

    reg_df = pd.DataFrame(reg_rows)
    return corr_df, reg_df


# ── plots ───────────────────────────────────────────────────────────────

def plot_budget_response(summary: pd.DataFrame, out_dir: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for attack, g in summary.groupby("attack"):
        g = g.sort_values("epsilon")
        ax.plot(g["epsilon"], g["mean_signed_shift"], marker="o", label=attack)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("epsilon (UAP L_inf budget)")
    ax.set_ylabel("mean signed shift (perturbed - clean)")
    ax.set_title(f"UAP budget-response — {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "budget_response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_arousal_stratified(predictions_with_bias: pd.DataFrame, out_dir: Path, n_strata: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    df = predictions_with_bias[predictions_with_bias["attack"] != "clean"].dropna(subset=["delta", "human_arousal"])
    if df.empty:
        return
    df = df.copy()
    df["arousal_stratum"] = pd.qcut(df["human_arousal"], q=n_strata, duplicates="drop")

    fig, ax = plt.subplots(figsize=(8, 5))
    for attack, g in df.groupby("attack"):
        means = g.groupby("arousal_stratum", observed=True)["delta"].mean()
        ax.plot(range(len(means)), means.values, marker="o", label=attack)
    ax.set_xticks(range(len(df["arousal_stratum"].cat.categories)))
    ax.set_xticklabels([str(c) for c in df["arousal_stratum"].cat.categories], rotation=20, ha="right")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("human normative arousal (low -> high)")
    ax.set_ylabel("mean delta_morality")
    ax.set_title("Moral-shift susceptibility by normative arousal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "arousal_stratified.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── cross-task merge (spec section 7) ─────────────────────────────────────

def cross_task_matrix(summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """summaries: {task_name: summary_df}. One row per task x attack x epsilon."""
    rows = []
    for task_name, summary in summaries.items():
        if summary is None or summary.empty:
            continue
        for _, r in summary.iterrows():
            rows.append({
                "task": task_name, "attack": r["attack"], "epsilon": r["epsilon"],
                "normalized_signed_shift": r["normalized_signed_shift"],
                "mean_abs_shift": r["mean_abs_shift"], "n": r["n"],
            })
    return pd.DataFrame(rows)
