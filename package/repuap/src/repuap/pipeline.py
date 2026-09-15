"""
Thin orchestration over the four pipeline stages. Each function here is a
convenience wrapper around discover.py / causal.py / uap.py / generalise.py --
call those modules directly for anything these wrappers don't expose.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .tasks import JudgmentTask
from . import discover, causal, uap, generalise


def discover_direction(
    model, processor, task: JudgmentTask,
    positive_prompt: str, negative_prompt: str,
    positive_images: list[dict], negative_images: list[dict],
    auc_threshold: float = 0.7, probe_threshold: float = 0.6,
) -> dict:
    """
    Stage I end-to-end: collect activations under two conditions (e.g. a
    "positive persona" prompt and a "negative persona" prompt over the SAME
    images, or two different image sets under one shared prompt -- whichever
    defines your contrast), then compute + evaluate a candidate direction at
    every layer.

    Returns {"vectors": {layer_key: mean_diff_vector}, "reports": {layer_key: VectorReport},
             "best_layer": str or None}.
    """
    pos_records = discover.collect_condition_activations(model, processor, task, positive_prompt, positive_images)
    neg_records = discover.collect_condition_activations(model, processor, task, negative_prompt, negative_images)

    matrix_pos = discover.build_layer_matrix(pos_records)
    matrix_neg = discover.build_layer_matrix(neg_records)

    vectors = discover.compute_contrast_vectors(matrix_pos, matrix_neg)
    reports = discover.evaluate_layers(matrix_pos, matrix_neg, auc_threshold, probe_threshold)

    kept = [(k, r) for k, r in reports.items() if r.keep]
    best_layer = max(kept, key=lambda kv: kv[1].projection_auc)[0] if kept else None

    return {"vectors": vectors, "reports": reports, "best_layer": best_layer,
            "positive_records": pos_records, "negative_records": neg_records}


def validate_causally(
    model, processor, task: JudgmentTask,
    prompt: str, vector: np.ndarray, layer_key: str,
    images: list[dict], alphas: list[float] | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Stage III end-to-end: dose-response sweep + summary table."""
    alphas = alphas or [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    results = causal.run_dose_response(model, processor, task, images, prompt,
                                        vector, layer_key, alphas, out_dir=out_dir)
    summary = causal.summarise_dose_response(results)
    return {"results": results, "summary": summary}


def train_uap(
    model, processor, task: JudgmentTask,
    prompt: str, direction_vector: np.ndarray, layer_key: str,
    train_images: list[str], eval_images: list[str],
    epsilons: list[float] | None = None, n_epochs: int = 20,
    lr: float | None = None, max_side: int = 336,
    out_dir: str | Path | None = None,
) -> dict[float, dict]:
    """Stage IV.1 end-to-end: sweep epsilons, return {epsilon: gradient_match_universal result}."""
    epsilons = epsilons or [0.1, 0.5, 1.0, 2.0]
    out: dict[float, dict] = {}
    for eps in epsilons:
        eps_lr = lr if lr is not None else eps / n_epochs
        eps_dir = Path(out_dir) / f"eps{eps:.2f}" if out_dir else None
        if eps_dir:
            eps_dir.mkdir(parents=True, exist_ok=True)
        result = uap.gradient_match_universal(
            model, processor, task, train_images, eval_images, prompt, layer_key,
            direction_vector, epsilon=eps, lr=eps_lr, n_epochs=n_epochs,
            max_side=max_side, out_dir=eps_dir,
        )
        if eps_dir:
            np.save(eps_dir / "delta.npy", result["delta"])
        out[eps] = result
    return out


def evaluate_generalisation(
    model, processor, task: JudgmentTask,
    manifest: pd.DataFrame, uap_conditions: list[tuple[str, np.ndarray]],
    **kwargs,
) -> dict:
    """
    Stage IV.2 end-to-end: run clean + every (name, delta) in uap_conditions
    over the manifest, then build the standard summary table.
    kwargs forwarded to generalise.evaluate_generalisation (max_side,
    max_new_tokens, prompt_ctx_cols, extra_manifest_cols, limit, ...).
    """
    conditions = [("clean", None)] + list(uap_conditions)
    predictions = generalise.evaluate_generalisation(model, processor, task, manifest, conditions, **kwargs)

    rows = []
    attacked = predictions[predictions["condition"] != "clean"]
    for name, g in attacked.groupby("condition"):
        g_ok = g.dropna(subset=["delta"])
        m = generalise.condition_metrics(g_ok, task.valid_labels, task.scale_width)
        m["condition"] = name
        rows.append(m)
    summary = pd.DataFrame(rows)
    return {"predictions": predictions, "summary": summary}
