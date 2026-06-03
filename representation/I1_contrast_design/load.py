"""
Loading and preprocessing of stored activation result files.

Also contains the layer-ordering and activation-pooling helpers that are shared
across all analytics modules (GDV, UMAP, t-SNE, PCA, probes).

Loading logic derived from gdv_summary.ipynb and metrics/metrics.py.
"""

from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


# ── Layer-key utilities (shared across all analytics) ────────────────────────

MOD_PRIORITY = {"vision": 0, "projector": 1, "language": 2, "unknown": 3}


def _modality_from_key_and_dim(key: str, d: int) -> str:
    s = str(key).lower()
    if any(t in s for t in ["vision", "vit", "image", "visual", "patch", "encoder"]):
        return "vision"
    if any(t in s for t in ["proj", "projector", "connector", "bridge"]):
        return "projector"
    if any(t in s for t in ["lang", "llm", "text", "decoder", "gpt", "lm_head"]):
        return "language"
    if d in (512, 768, 960, 1024, 1152, 1280, 1408, 1536, 1664, 1792):
        return "vision"
    if d in (2048, 4096, 5120, 6144, 8192):
        return "language"
    return "unknown"


def _extract_layer_number(key: Any) -> int | None:
    if isinstance(key, int):
        return key
    m = re.search(r"\d+", str(key))
    return int(m.group()) if m else None


def _sort_items_by_mod_layer_width(items: list[tuple]) -> list[tuple]:
    """Sort (layer_key, activation_array) pairs by modality → layer index → width."""
    def sk(kv):
        k, v = kv
        arr = np.array(v)
        D   = int(arr.shape[-1]) if arr.ndim >= 1 else 0
        mod = _modality_from_key_and_dim(k, D)
        num = _extract_layer_number(k)
        num = num if num is not None else 10**9
        return (MOD_PRIORITY.get(mod, 3), num, D, str(k))
    return sorted(items, key=sk)


def sort_layer_keys(layer_keys: list[str]) -> list[str]:
    """Sort a list of comp_keys like 'vision_0_D1408' or 'language_45_D5120'."""
    def sk(k: str):
        parts = k.split("_D")
        base  = parts[0]
        mod, num_str = base.split("_", 1)
        D = int(parts[1]) if len(parts) > 1 else 0
        return (MOD_PRIORITY.get(mod, 3), int(num_str), D)
    return sorted(layer_keys, key=sk)


# ── Activation pooling ────────────────────────────────────────────────────────

def pool_activation(
    arr: np.ndarray,
    modality: str,
    prefer_first_token: bool = True,
    prefer_cls_for_vision: bool = True,
) -> tuple[np.ndarray, str]:
    """
    Reduce a (possibly multi-token) activation array to a single vector.

    Vision: take CLS token (index 0).
    LLM:    take first token (index 0, which is the label-token position after
            teacher-forcing).  Falls back to mean if array is 1D.

    Returns:
        (vector, pooling_method_str)
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        return a, "Vector"

    if modality == "vision":
        if prefer_cls_for_vision:
            try:
                return a[0] if a.ndim == 2 else a.reshape(-1, a.shape[-1])[0], "CLS"
            except Exception:
                return a.reshape(-1, a.shape[-1]).mean(axis=0), "Mean (vision fallback)"
        return a.reshape(-1, a.shape[-1]).mean(axis=0), "Mean (vision)"

    if modality in ("language", "projector"):
        if prefer_first_token:
            try:
                vec = a[0] if a.ndim == 2 else a.reshape(-1, a.shape[-1])[0]
                return vec, "FirstToken"
            except Exception:
                return a.reshape(-1, a.shape[-1]).mean(axis=0), "Mean (fallback)"
        return a.reshape(-1, a.shape[-1]).mean(axis=0), "Mean"

    return a.reshape(-1, a.shape[-1]).mean(axis=0), "Mean (unknown)"


# ── Loading ───────────────────────────────────────────────────────────────────

def load_activation_results(path: str | Path) -> dict:
    """
    Load a results_<persona_key>.npy file.

    Returns the top-level dict with keys:
        version, persona_key, selected_images, results
    """
    return np.load(path, allow_pickle=True).item()


def get_results_list(data: dict) -> list[dict]:
    """Extract the list of per-image result dicts from a loaded data dict."""
    return data["results"]


def get_layer_keys(results: list[dict]) -> list[str]:
    """Return sorted list of embedding layer keys from the first valid result."""
    for r in results:
        if r.get("embeddings"):
            return sorted(r["embeddings"].keys())
    return []


def filter_by_label_support(
    results: list[dict],
    min_fraction: float = 0.10,
    label_field: str = "interestingness",
) -> list[dict]:
    """
    Drop results from labels that have fewer than min_fraction of total samples.

    Mirrors the filtering in the original metrics.py to maintain consistency.
    Passing min_fraction=0.0 keeps all classes.
    """
    labels = [r[label_field] for r in results]
    if not labels:
        return results

    vals, counts = np.unique(labels, return_counts=True)
    thresh       = max(1, int(np.floor(min_fraction * len(labels))))
    keep         = {v for v, c in zip(vals, counts) if c >= thresh}
    return [r for r in results if r[label_field] in keep]


def _detect_label_key(results: list[dict]) -> str:
    for r in results:
        if "interestingness_label" in r:
            return "interestingness_label"
        if "interestingness" in r:
            return "interestingness"
    raise KeyError("No interestingness label key found in results")


def stack_activations(
    results: list[dict],
    layer_key: str,
    label_field: str = "interestingness",
) -> tuple[np.ndarray, list[str]]:
    """
    Stack per-image activation vectors for a given layer into arrays.

    Returns:
        X: np.ndarray of shape (N, D)
        y: list of str labels, length N
    """
    vecs, labels = [], []
    for r in results:
        emb = r.get("embeddings", {})
        if layer_key not in emb:
            continue
        arr = np.asarray(emb[layer_key])
        if arr.ndim > 1:
            # already pooled by hooks (CLS / last token), but flatten just in case
            arr = arr.reshape(-1, arr.shape[-1])[0]
        vecs.append(arr)
        labels.append(r[label_field])
    return np.vstack(vecs), labels


def _hook_selection_label(raw_key: str, modality: str) -> str:
    """Infer which hook selector captured this layer (for display only)."""
    k = str(raw_key).lower()
    if "cls" in k:
        return "cls"
    if "rating" in k or "label" in k:
        return "rating_token"
    return "cls" if modality == "vision" else "rating_token"


def build_layer_matrix(
    results: list[dict],
    label_field: str = "interestingness",
    min_fraction: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """
    Build a comp_key-indexed dict of {X, y, hook_selection, modality, layer_num, D}
    ready for analytics.

    comp_key format: "{modality}_{layer_num}_D{D}"  e.g. "language_45_D5120"

    Embeddings are stored as 1D vectors by the hook selectors (CLS for vision,
    rating-token position for LLM) — no pooling is applied here.
    """
    results = filter_by_label_support(results, min_fraction=min_fraction, label_field=label_field)
    label_key = _detect_label_key(results)

    layer_acts: dict[str, list] = defaultdict(list)
    layer_sidx: dict[str, list] = defaultdict(list)
    layer_rawkey: dict[str, str] = {}

    for s_idx, result in enumerate(results):
        items = _sort_items_by_mod_layer_width(list(result["embeddings"].items()))
        for pos, (k, act) in enumerate(items):
            arr = np.array(act)
            D   = int(arr.shape[-1]) if arr.ndim >= 1 else 0
            mod = _modality_from_key_and_dim(k, D)
            num = _extract_layer_number(k)
            if num is None:
                num = pos
            comp_key = f"{mod}_{num}_D{D}"
            layer_acts[comp_key].append(arr)
            layer_sidx[comp_key].append(s_idx)
            layer_rawkey.setdefault(comp_key, str(k))

    labels_all = np.array([r[label_key] for r in results])

    out = {}
    for comp_key in sort_layer_keys(list(layer_acts.keys())):
        arrs = layer_acts[comp_key]
        sidx = layer_sidx[comp_key]

        base, d_str = comp_key.split("_D")
        mod, num_str = base.split("_", 1)
        D   = int(d_str)
        num = int(num_str)

        # Embeddings are already 1D — stored directly from hook selectors.
        vecs = [np.asarray(a).ravel() for a in arrs]
        X = np.vstack(vecs)
        y = labels_all[sidx]

        hook_sel = _hook_selection_label(layer_rawkey.get(comp_key, ""), mod)

        out[comp_key] = {
            "X":              X,
            "y":              y.tolist(),
            "hook_selection": hook_sel,
            "modality":       mod,
            "layer_num":      num,
            "D":              D,
            "n_samples":      int(X.shape[0]),
        }

    return out


def load_multiple(paths: list[str | Path]) -> dict[str, dict]:
    """
    Load several result files and return {persona_key: data_dict}.
    persona_key is read from the stored dict, not the filename.
    """
    out = {}
    for p in paths:
        data = load_activation_results(p)
        out[data["persona_key"]] = data
    return out


def merge_conditions(
    data_list: list[dict],
    merged_key: str = "merged",
) -> dict:
    """
    Concatenate results from multiple loaded data dicts into one pseudo-condition.

    Useful for gender-averaged emotion contrasts: merge female_anger + male_anger
    into a single data dict so the existing I.2–I.5 pipeline can treat it as one
    condition without modification.

    Args:
        data_list:  list of loaded data dicts (each with a "results" list)
        merged_key: persona_key stored in the returned dict

    Returns:
        A data dict with concatenated results, usable anywhere a single condition
        dict is expected.
    """
    all_results = []
    for d in data_list:
        all_results.extend(d.get("results", []))
    return {
        "version":    "merged",
        "persona_key": merged_key,
        "results":    all_results,
    }
