import pandas as pd

from repuap.generalise import condition_metrics, budget_response


def test_condition_metrics_basic():
    df = pd.DataFrame({
        "clean_model_label": [1, 2, 3, 4, 5, 1, 2, 3],
        "model_label": [2, 2, 4, 4, 5, 1, 3, 3],
        "clean_model_score": [1, 2, 3, 4, 5, 1, 2, 3],
        "model_score": [2, 2, 4, 4, 5, 1, 3, 3],
    })
    df["delta"] = df["model_score"] - df["clean_model_score"]
    m = condition_metrics(df, [1, 2, 3, 4, 5], scale_width=4.0)

    assert m["n"] == 8
    assert m["clean_mean"] == 2.625
    assert m["perturbed_mean"] == 3.0
    assert round(m["mean_signed_shift"], 3) == 0.375
    assert round(m["normalized_signed_shift"], 5) == round(0.375 / 4.0, 5)
    assert m["pct_down"] == 0.0
    assert m["clean_n_1"] == 2
    assert m["perturbed_n_1"] == 1


def test_condition_metrics_empty():
    df = pd.DataFrame(columns=["clean_model_label", "model_label", "clean_model_score", "model_score", "delta"])
    assert condition_metrics(df, [1, 2, 3], 2.0) == {"n": 0}


def test_budget_response_monotonic_detected():
    summary = pd.DataFrame([
        {"attack": "a", "epsilon": 0.1, "mean_signed_shift": -0.1},
        {"attack": "a", "epsilon": 0.5, "mean_signed_shift": -0.3},
        {"attack": "a", "epsilon": 1.0, "mean_signed_shift": -0.6},
        {"attack": "a", "epsilon": 2.0, "mean_signed_shift": -1.2},
    ])
    result = budget_response(summary)
    row = result[result["attack"] == "a"].iloc[0]
    assert row["spearman_rho"] == -1.0
    assert row["spearman_p"] == 0.0


def test_budget_response_too_few_points_is_nan():
    summary = pd.DataFrame([
        {"attack": "a", "epsilon": 0.1, "mean_signed_shift": -0.1},
        {"attack": "a", "epsilon": 0.5, "mean_signed_shift": -0.3},
    ])
    result = budget_response(summary)
    row = result[result["attack"] == "a"].iloc[0]
    assert pd.isna(row["spearman_rho"])
