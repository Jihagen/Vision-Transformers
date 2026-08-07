"""
Common attack-effect metrics for Generalisation Level 1 (spec section 2).

condition_metrics() operates on one (dataset, task, attack, epsilon) slice —
rows already paired against their own clean response. budget_response()
operates across the four independently-trained epsilon budgets for one
attack; this is a UAP BUDGET-response relationship, not an activation-vector
dose-response (these are separately trained deltas, not points on one sweep).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance


def condition_metrics(df: pd.DataFrame, valid_labels: list[int], scale_width: float) -> dict:
    """
    df: rows for one (attack, epsilon) condition, already restricted to
    parse_ok==True on BOTH the clean and perturbed response for that sample
    (delta is only meaningful when both sides parsed).
    Columns required: clean_label, model_label, delta.
    """
    n = len(df)
    if n == 0:
        return {"n": 0}

    clean = df["clean_label"].astype(float)
    pert = df["model_label"].astype(float)
    delta = df["delta"].astype(float)

    out = {
        "n": n,
        "clean_mean": float(clean.mean()),
        "perturbed_mean": float(pert.mean()),
        "mean_signed_shift": float(delta.mean()),
        "mean_abs_shift": float(delta.abs().mean()),
        "normalized_signed_shift": float(delta.mean() / scale_width),
        "pct_up": float((delta > 0).mean()),
        "pct_unchanged": float((delta == 0).mean()),
        "pct_down": float((delta < 0).mean()),
        "wasserstein": float(wasserstein_distance(clean, pert)),
    }

    clean_counts = clean.value_counts()
    pert_counts = pert.value_counts()
    for lbl in valid_labels:
        out[f"clean_n_{lbl}"] = int(clean_counts.get(lbl, 0))
        out[f"perturbed_n_{lbl}"] = int(pert_counts.get(lbl, 0))

    return out


def budget_response(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    summary_df: one row per (attack, epsilon) with a 'mean_signed_shift'
    column (the output of condition_metrics, tabulated). Descriptive Spearman
    correlation between epsilon and mean signed shift, per attack.
    """
    rows = []
    for attack, g in summary_df.groupby("attack"):
        g = g.sort_values("epsilon")
        if g["epsilon"].nunique() >= 3 and g["mean_signed_shift"].notna().sum() >= 3:
            rho, p = spearmanr(g["epsilon"], g["mean_signed_shift"])
        else:
            rho, p = float("nan"), float("nan")
        rows.append({
            "attack": attack,
            "n_epsilons": int(g["epsilon"].nunique()),
            "spearman_rho_eps_vs_meandelta": rho,
            "spearman_p": p,
            "eps2_mean_signed_shift": g.loc[g["epsilon"] == 2.0, "mean_signed_shift"].squeeze()
                if (g["epsilon"] == 2.0).any() else np.nan,
        })
    return pd.DataFrame(rows)
