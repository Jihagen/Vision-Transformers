"""
Stage I — Representation Discovery.

1. collect_condition_activations(): run a JudgmentTask over a set of images
   under one condition (e.g. one persona, or "positive"/"negative" of a
   contrast), capturing per-layer hidden states at the judgment-label token.
2. compute_mean_difference_vector() / compute_contrast_vectors(): the
   simplest candidate direction — mean(H_pos) - mean(H_neg) per layer.
3. train_linear_probe(): a logistic-regression alternative (CAV) with
   cross-validated accuracy as a decodability score.
4. evaluate_layers(): score every layer's candidate direction (projection
   AUROC, mean-diff/probe alignment) and recommend which to keep.
"""

from __future__ import annotations
import gc
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from .hooks import HookState, register_extract_hooks
from .model_io import preprocess_image, remove_batch_dimension, DOWNSCALE_LADDER
from .tasks import JudgmentTask, ParsedResponse, parse_response

logger = logging.getLogger(__name__)


# ── Two-stage judgment + activation capture ─────────────────────────────────

def run_judgment_two_stage(
    task: JudgmentTask,
    prompt: str,
    image_path: str | Path,
    model,
    processor,
    state: HookState,
    image_max_size: int = 800,
    max_new_tokens: int = 64,
) -> tuple[str, ParsedResponse]:
    """
    Two-stage inference for one image x prompt pair:

    Stage 1: full generation — obtain the judgment label + explanation.
             Vision hooks fire during this pass (phase="vision_once").
    Stage 2: teacher-forced single step, forcing the sequence up through
             task.anchor() — captures LLM layer hidden states at the label
             token position (phase="rating_step"), the position a
             mean-difference/probe direction and a gradient-matching UAP
             target should be defined at.

    Returns (decoded_text, ParsedResponse). Raises torch.cuda.OutOfMemoryError
    (caller handles via downscale) if generation OOMs.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image = preprocess_image(image_path, max_side=image_max_size)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat_input = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=chat_input, images=image, return_tensors="pt",
        add_special_tokens=True, truncation=False, max_length=512,
    )

    target_device = next(model.parameters()).device
    _keep = {"input_ids", "attention_mask", "pixel_values"}
    model_inputs = {
        k: v.to(target_device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items() if k in _keep
    }

    gen_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True, do_sample=False,
                       pad_token_id=processor.tokenizer.eos_token_id)

    state.phase = "vision_once"
    try:
        with torch.no_grad():
            gen_ids = model.generate(**model_inputs, **gen_kwargs)
    finally:
        state.phase = "idle"

    decoded = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    parsed = parse_response(decoded, task)

    # Stage 2: teacher-forced one step up through the label anchor.
    full_seq_ids = gen_ids[0].tolist()
    input_len = model_inputs["input_ids"].shape[1]
    forced_ids = processor.tokenizer.encode(task.anchor(), add_special_tokens=False)
    prefix_ids = full_seq_ids[:input_len] + forced_ids

    local = {
        "input_ids": torch.tensor([prefix_ids], device=model_inputs["input_ids"].device),
        "attention_mask": torch.ones(1, len(prefix_ids),
                                      device=model_inputs["attention_mask"].device,
                                      dtype=model_inputs["attention_mask"].dtype),
        "pixel_values": model_inputs["pixel_values"],
    }
    state.phase = "rating_step"
    with torch.no_grad():
        model.generate(**local, max_new_tokens=1, do_sample=False, use_cache=True,
                        pad_token_id=processor.tokenizer.eos_token_id)
    state.phase = "idle"

    del gen_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return decoded, parsed


def run_judgment_with_retries(
    task: JudgmentTask,
    prompt: str,
    image_path: str | Path,
    model,
    processor,
    state: HookState,
    initial_max_side: int = 800,
) -> tuple[str, ParsedResponse, int]:
    """
    Wrap run_judgment_two_stage with an OOM-downscale retry ladder and a
    JSON-parse-failure token-budget retry. Returns (decoded, parsed, image_max_size_used).
    """
    try:
        state.reset_embeddings()
        decoded, parsed = run_judgment_two_stage(task, prompt, image_path, model, processor,
                                                  state, image_max_size=initial_max_side)
        if parsed.parse_ok:
            return decoded, parsed, initial_max_side
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        idx = DOWNSCALE_LADDER.index(initial_max_side) if initial_max_side in DOWNSCALE_LADDER else 0
        for oom_size in DOWNSCALE_LADDER[idx + 1:]:
            logger.warning(f"OOM; retrying with image_max_size={oom_size}")
            state.reset_embeddings()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                decoded, parsed = run_judgment_two_stage(task, prompt, image_path, model, processor,
                                                          state, image_max_size=oom_size)
                return decoded, parsed, oom_size
            except RuntimeError as e2:
                if "out of memory" not in str(e2).lower():
                    raise
        raise

    # Parse failed but no OOM — retry with a larger token budget.
    for n_tokens in (96, 128):
        state.reset_embeddings()
        decoded, parsed = run_judgment_two_stage(task, prompt, image_path, model, processor,
                                                  state, image_max_size=initial_max_side,
                                                  max_new_tokens=n_tokens)
        if parsed.parse_ok:
            return decoded, parsed, initial_max_side
    return decoded, parsed, initial_max_side


def collect_condition_activations(
    model,
    processor,
    task: JudgmentTask,
    prompt: str,
    images: list[dict],   # [{"filename": str, "img_path": str}, ...]
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    """
    Run one condition (a fixed prompt, e.g. one persona) over a list of
    images, returning per-image records with parsed judgment + captured
    per-layer activations at the judgment-label token:

        {"filename", "img_path", "label", "explanation", "parse_ok",
         "embeddings": {layer_key: np.ndarray}}

    on_result: optional callback invoked with each record as it's produced —
    use it to checkpoint to disk on long runs (this function keeps no
    on-disk state of its own).
    """
    state = HookState()
    register_extract_hooks(model, state)
    records = []
    try:
        for img in images:
            img_path = img["img_path"]
            try:
                decoded, parsed, size_used = run_judgment_with_retries(
                    task, prompt, img_path, model, processor, state
                )
                embeds = {k: remove_batch_dimension(v) for k, v in state.embeddings.items()}
                rec = {
                    "filename": img.get("filename", Path(img_path).name),
                    "img_path": img_path,
                    "label": parsed.label,
                    "explanation": parsed.explanation,
                    "parse_ok": parsed.parse_ok,
                    "image_max_size": size_used,
                    "embeddings": embeds,
                }
            except Exception as e:
                logger.warning(f"Skipping {img_path}: {e}")
                continue
            records.append(rec)
            if on_result is not None:
                on_result(rec)
    finally:
        state.remove_hooks()
    return records


# ── Layer-matrix helper ──────────────────────────────────────────────────────

def _to_comp_key(raw_key: str, dim: int) -> str:
    """
    Translate a raw hook name ("llm_layer_29_rating_token",
    "vision_layer_0_cls") into the "language_29_D5120" / "vision_0_D1408"
    comp-key convention that causal.py / uap.py expect as `layer_key` --
    dimension-annotated, and matching what register_grad_extract_hook's
    "language_(\\d+)" pattern and causal.py's _comp_key_to_hook_name expect.
    """
    m = re.match(r"llm_layer_(\d+)_rating_token", raw_key)
    if m:
        return f"language_{m.group(1)}_D{dim}"
    m = re.match(r"vision_layer_(\d+)_cls", raw_key)
    if m:
        return f"vision_{m.group(1)}_D{dim}"
    return f"{raw_key}_D{dim}"   # unrecognised hook name: keep it, just annotate dim


def build_layer_matrix(records: list[dict]) -> dict[str, np.ndarray]:
    """records: output of collect_condition_activations (must have parse_ok=True
    to be included). Returns {comp_key: stacked (N, D) activation matrix},
    comp_key e.g. "language_29_D5120" -- ready to pass straight to
    causal.run_dose_response / uap.gradient_match_universal as layer_key."""
    by_layer: dict[str, list[np.ndarray]] = {}
    for r in records:
        if not r.get("parse_ok", False):
            continue
        for raw_key, vec in r["embeddings"].items():
            arr = np.asarray(vec).ravel()
            comp_key = _to_comp_key(raw_key, arr.shape[-1])
            by_layer.setdefault(comp_key, []).append(arr)
    return {k: np.stack(v, axis=0) for k, v in by_layer.items()}


# ── Mean-difference vectors ──────────────────────────────────────────────────

def compute_mean_difference_vector(H_pos: np.ndarray, H_neg: np.ndarray, normalize: bool = True) -> np.ndarray:
    """v = mean(H_pos) - mean(H_neg), optionally L2-normalized."""
    v = H_pos.mean(axis=0) - H_neg.mean(axis=0)
    if normalize:
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v = v / norm
    return v.astype(np.float32)


def compute_contrast_vectors(
    matrix_pos: dict[str, np.ndarray],
    matrix_neg: dict[str, np.ndarray],
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Mean-difference vector per layer common to both conditions."""
    common = sorted(set(matrix_pos) & set(matrix_neg))
    return {k: compute_mean_difference_vector(matrix_pos[k], matrix_neg[k], normalize) for k in common}


# ── Linear probe (CAV) ────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    layer_key: str
    accuracy_cv: float
    accuracy_std: float
    cav_vector: np.ndarray
    n_pos: int
    n_neg: int


def train_linear_probe(
    H_pos: np.ndarray, H_neg: np.ndarray,
    cv_folds: int = 5, max_iter: int = 1000,
    normalize_vector: bool = True, layer_key: str = "",
) -> ProbeResult:
    """Logistic-regression CAV direction + cross-validated separability."""
    X = np.concatenate([H_pos, H_neg], axis=0)
    y = np.concatenate([np.ones(len(H_pos)), np.zeros(len(H_neg))])
    n_splits = min(cv_folds, min(len(H_pos), len(H_neg)))
    clf = LogisticRegression(max_iter=max_iter)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        scores = cross_val_score(clf, X, y, cv=cv)
        acc, std = float(scores.mean()), float(scores.std())
    else:
        acc, std = float("nan"), float("nan")
    clf.fit(X, y)
    v = clf.coef_[0].astype(np.float32)
    if normalize_vector:
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v = v / norm
    return ProbeResult(layer_key, acc, std, v, len(H_pos), len(H_neg))


# ── Vector evaluation ─────────────────────────────────────────────────────────

@dataclass
class VectorReport:
    layer_key: str
    projection_auc: float
    md_cav_cosine: float
    probe_accuracy_cv: float
    keep: bool
    notes: str = ""
    extra: dict = field(default_factory=dict)


def projection_separation(H_pos: np.ndarray, H_neg: np.ndarray, vector: np.ndarray) -> float:
    """AUROC separating pos/neg by scalar projection onto `vector` (0.5=chance, 1.0=perfect)."""
    v_unit = vector / (np.linalg.norm(vector) + 1e-12)
    proj_pos, proj_neg = H_pos @ v_unit, H_neg @ v_unit
    scores = np.concatenate([proj_pos, proj_neg])
    labels = np.concatenate([np.ones(len(H_pos)), np.zeros(len(H_neg))])
    try:
        auc = roc_auc_score(labels, scores)
        return float(max(auc, 1.0 - auc))
    except Exception:
        return 0.5


def alignment_score(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two direction vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def evaluate_layers(
    matrix_pos: dict[str, np.ndarray],
    matrix_neg: dict[str, np.ndarray],
    auc_threshold: float = 0.7,
    probe_threshold: float = 0.6,
) -> dict[str, VectorReport]:
    """
    For every layer common to both conditions: compute the mean-difference
    vector, a linear probe, their alignment, and an AUROC separation score —
    then recommend keep=True/False based on the two thresholds.
    """
    reports = {}
    common = sorted(set(matrix_pos) & set(matrix_neg))
    for key in common:
        H_pos, H_neg = matrix_pos[key], matrix_neg[key]
        md_vec = compute_mean_difference_vector(H_pos, H_neg)
        probe = train_linear_probe(H_pos, H_neg, layer_key=key)
        auc = projection_separation(H_pos, H_neg, md_vec)
        cos = alignment_score(md_vec, probe.cav_vector)
        keep = auc >= auc_threshold and (np.isnan(probe.accuracy_cv) or probe.accuracy_cv >= probe_threshold)
        notes = "" if keep else "below AUC/probe threshold"
        reports[key] = VectorReport(key, auc, cos, probe.accuracy_cv, keep, notes,
                                     extra={"md_vector": md_vec, "cav_vector": probe.cav_vector})
    return reports
