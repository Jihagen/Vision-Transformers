"""
Stage IV.2 — Generalisation testing.

Apply an already-trained UAP delta (from uap.py; never retrained or scaled
here) to a new set of images under a new JudgmentTask, and measure the
shift relative to each sample's own clean (unperturbed) response. Also
works with any other single-image perturbation you supply as a plain
numpy array matching the model's pixel_values shape -- delta doesn't have
to come from uap.py.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance

from .model_io import preprocess_image
from .generation import build_inputs, generate_text
from .tasks import JudgmentTask, parse_response

logger = logging.getLogger(__name__)


# ── UAP application ──────────────────────────────────────────────────────────

def apply_delta(pixel_values, delta_np: np.ndarray, device, delta_t_cache: dict):
    """Add delta_np to pixel_values and clamp to the model's valid pixel range
    ([-1, 1] for Llama-4/SigLIP-style processors -- adjust if yours differs).
    delta_t_cache is a caller-owned dict used to cache the uploaded delta
    tensor across calls, keyed by id(delta_np)."""
    import torch
    ref_shape = tuple(delta_np.shape)
    if tuple(pixel_values.shape) != ref_shape:
        raise ValueError(f"pixel_values shape {tuple(pixel_values.shape)} != delta shape {ref_shape} "
                          "(image did not resolve to the same processor tiling used to train the delta)")
    key = id(delta_np)
    if key not in delta_t_cache:
        delta_t_cache[key] = torch.tensor(delta_np, device=device)
    return (pixel_values.to(device) + delta_t_cache[key]).clamp(-1.0, 1.0)


def generate_response(model, processor, image, prompt: str, delta_np: np.ndarray | None,
                       delta_t_cache: dict, max_new_tokens: int = 128) -> str:
    """One model.generate() call, optionally with delta_np added to pixel_values."""
    device = next(model.parameters()).device
    inputs = build_inputs(processor, image, prompt, device)
    pixel_values = inputs["pixel_values"]
    if delta_np is not None:
        pixel_values = apply_delta(pixel_values, delta_np, device, delta_t_cache)
    else:
        pixel_values = pixel_values.to(device)
    return generate_text(model, processor, inputs["input_ids"], inputs["attention_mask"],
                          pixel_values, max_new_tokens=max_new_tokens)


# ── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate_generalisation(
    model, processor, task: JudgmentTask,
    manifest: pd.DataFrame,       # must have columns: sample_id, image_id, img_path
    conditions: list[tuple[str, np.ndarray | None]],   # [("clean", None), ("my_uap_eps1.0", delta_np), ...]
    prompt_ctx_cols: list[str] | None = None,           # manifest columns passed to task.prompt_builder(row)
    extra_manifest_cols: list[str] | None = None,       # manifest columns carried through into predictions
    max_side: int = 336,
    max_new_tokens: int = 128,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Run every (sample, condition) pair, pairing each perturbed response
    against that same sample's clean response. First entry in `conditions`
    should be ("clean", None) -- if it's not, delta will be computed against
    whichever condition name is literally "clean".

    Returns one row per (sample, condition):
        sample_id, image_id, condition, model_label, clean_model_label,
        delta, explanation, raw_output, parse_ok, used_fallback_parser
        + any extra_manifest_cols
    """
    rows_df = manifest.head(limit) if limit else manifest
    delta_t_cache: dict = {}
    extra_manifest_cols = extra_manifest_cols or []
    records = []

    for i, (_, row) in enumerate(rows_df.iterrows()):
        try:
            image = preprocess_image(row["img_path"], max_side=max_side)
        except Exception as e:
            logger.warning(f"  [{i+1}] skip {row.get('sample_id')}: image load failed -- {e}")
            continue

        ctx = row if prompt_ctx_cols is None else {c: row[c] for c in prompt_ctx_cols}
        prompt = task.prompt_builder(ctx)
        clean_label = None

        for condition_name, delta_np in conditions:
            raw = generate_response(model, processor, image, prompt, delta_np, delta_t_cache,
                                     max_new_tokens=max_new_tokens)
            parsed = parse_response(raw, task)
            if condition_name == "clean":
                clean_label = parsed.label
            delta_val = (task.score(parsed.label) - task.score(clean_label)
                         if (parsed.label is not None and clean_label is not None) else None)

            record = {
                "sample_id": row["sample_id"], "image_id": row["image_id"],
                "condition": condition_name,
                "model_label": parsed.label, "clean_model_label": clean_label,
                "model_score": task.score(parsed.label) if parsed.label is not None else None,
                "clean_model_score": task.score(clean_label) if clean_label is not None else None,
                "delta": delta_val, "explanation": parsed.explanation,
                "raw_output": parsed.raw_output, "parse_ok": parsed.parse_ok,
                "used_fallback_parser": parsed.used_fallback,
            }
            for col in extra_manifest_cols:
                record[col] = row.get(col)
            records.append(record)

    return pd.DataFrame(records)


# ── Metrics ───────────────────────────────────────────────────────────────────

def condition_metrics(df: pd.DataFrame, valid_labels: list, scale_width: float) -> dict:
    """df: rows for one condition, restricted to parse_ok on both clean and
    perturbed for that sample (delta non-null). Numeric ops use the
    model_score/clean_model_score columns (works for str-valued labels via
    JudgmentTask.label_to_score); label-distribution counts use the raw
    model_label/clean_model_label columns, keyed against valid_labels."""
    n = len(df)
    if n == 0:
        return {"n": 0}
    clean_score = df["clean_model_score"].astype(float)
    pert_score = df["model_score"].astype(float)
    delta = df["delta"].astype(float)
    out = {
        "n": n, "clean_mean": float(clean_score.mean()), "perturbed_mean": float(pert_score.mean()),
        "mean_signed_shift": float(delta.mean()), "mean_abs_shift": float(delta.abs().mean()),
        "normalized_signed_shift": float(delta.mean() / scale_width),
        "pct_up": float((delta > 0).mean()), "pct_unchanged": float((delta == 0).mean()),
        "pct_down": float((delta < 0).mean()),
        "wasserstein": float(wasserstein_distance(clean_score, pert_score)),
    }
    clean_counts = df["clean_model_label"].value_counts()
    pert_counts = df["model_label"].value_counts()
    for lbl in valid_labels:
        out[f"clean_n_{lbl}"] = int(clean_counts.get(lbl, 0))
        out[f"perturbed_n_{lbl}"] = int(pert_counts.get(lbl, 0))
    return out


def budget_response(summary_df: pd.DataFrame, group_col: str = "attack", x_col: str = "epsilon") -> pd.DataFrame:
    """
    summary_df: one row per (group, x) with a 'mean_signed_shift' column.
    Descriptive Spearman correlation between x (e.g. perturbation budget) and
    mean signed shift, per group (e.g. per attack). This is a BUDGET-response
    across independently-produced perturbations at different x, not a
    dose-response over one continuously-scaled perturbation (see causal.py
    for that).
    """
    rows = []
    for group, g in summary_df.groupby(group_col):
        g = g.sort_values(x_col)
        if g[x_col].nunique() >= 3 and g["mean_signed_shift"].notna().sum() >= 3:
            rho, p = spearmanr(g[x_col], g["mean_signed_shift"])
        else:
            rho, p = float("nan"), float("nan")
        rows.append({group_col: group, f"n_{x_col}s": int(g[x_col].nunique()),
                     "spearman_rho": rho, "spearman_p": p})
    return pd.DataFrame(rows)
