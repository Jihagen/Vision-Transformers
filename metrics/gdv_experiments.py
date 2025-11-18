#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from itertools import combinations
import csv
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import pickle
from collections import defaultdict
from typing import Optional

from metrics.tsne import run_tsne_for_layer
from metrics.umap import run_umap_for_layer
from metrics.gdv import compute_gdv_both, compute_gdv_metric

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS: class filter, pooling (vector+matrix), modality/layer sorting
# ───────────────────────────────────────────────────────────────────────────────

def filter_results_by_min_support(results_list, label_key="interestingness", min_frac=0.10):
    """
    Keep only samples whose class has at least `min_frac` of the total samples.
    Returns: filtered_results, kept_labels, dropped_labels, counts_dict
    """
    labels = [r[label_key] for r in results_list]
    if len(labels) == 0:
        return results_list, set(), set(), {}

    vals, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    thresh = max(1, int(np.floor(min_frac * n)))
    keep_labels = {v for v, c in zip(vals, counts) if c >= thresh}
    drop_labels = set(vals) - keep_labels

    filtered = [r for r in results_list if r[label_key] in keep_labels]
    counts_dict = {v: int(c) for v, c in zip(vals, counts)}
    return filtered, keep_labels, drop_labels, counts_dict


def _safe_to_vector(a: np.ndarray) -> np.ndarray:
    """
    Robustly reduce any activation 'a' to a single 1D vector.
    Handles shapes: (D), (B,D), (T,D), (B,T,D) by averaging all non-last dims.
    """
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr
    flat = arr.reshape(-1, arr.shape[-1])
    return flat.mean(axis=0)


def _pool_matrix(a: np.ndarray, which: str) -> np.ndarray:
    """
    Pool a 2D or 3D activation into a single vector according to `which`.
    - which in {"first","mean","last"}.
    Works with (T,D), (B,D), (B,T,D). Anything awkward falls back to safe mean.
    """
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr

    # Normalize to (N,D)
    if arr.ndim == 2:
        M, D = arr.shape
        if which == "first":
            return arr[0]
        elif which == "last":
            return arr[-1]
        else:
            return arr.mean(axis=0)

    # arr.ndim >= 3 → collapse all leading dims except last
    flat = arr.reshape(-1, arr.shape[-1])
    if which == "first":
        return flat[0]
    elif which == "last":
        return flat[-1]
    else:
        return flat.mean(axis=0)


def pool_activation(arr: np.ndarray, modality: str,
                    prefer_first_token: bool = True,
                    prefer_cls_for_vision: bool = True) -> tuple[np.ndarray, str]:
    """
    General pooling that supports both vector and matrix inputs.
    Returns (vector, pooling_method_str).
    """
    a = np.asarray(arr)
    # If already a single vector, just return it.
    if a.ndim == 1:
        return a, "Vector"

    # Vision: prefer CLS (row 0) → fallback to mean
    if modality == "vision":
        if prefer_cls_for_vision:
            try:
                return _pool_matrix(a, "first"), "CLS"
            except Exception:
                return _safe_to_vector(a), "Mean (vision fallback)"
        else:
            return _safe_to_vector(a), "Mean (vision)"

    # Language / Projector: prefer first token as "CLS-proxy" if requested
    if modality in ("language", "projector"):
        if prefer_first_token:
            try:
                return _pool_matrix(a, "first"), "FirstToken (CLS-proxy)"
            except Exception:
                return _safe_to_vector(a), "Mean (fallback)"
        else:
            return _safe_to_vector(a), "Mean"

    # Unknown modality: be safe
    return _safe_to_vector(a), "Mean (unknown)"


MOD_PRIORITY = {"vision": 0, "projector": 1, "language": 2, "unknown": 3}

def _extract_layer_number(key):
    """Return an int from keys like 12, '12', 'layer_12', 'block.12.attn'; else None."""
    if isinstance(key, int):
        return key
    s = str(key)
    m = re.search(r'\d+', s)
    return int(m.group()) if m else None

def _modality_from_key_and_dim(key, d: int) -> str:
    """Heuristic modality from key text + hidden width."""
    s = str(key).lower()
    if any(t in s for t in ["vision", "vit", "image", "visual", "patch", "encoder"]):
        return "vision"
    if any(t in s for t in ["proj", "projector", "connector", "bridge"]):
        return "projector"
    if any(t in s for t in ["lang", "llm", "text", "decoder", "gpt", "lm_head"]):
        return "language"
    # fallback by common widths
    if d in (512, 768, 960, 1024, 1152, 1280, 1408, 1536, 1664, 1792):
        return "vision"
    if d in (2048, 4096, 5120, 6144, 8192):
        return "language"
    return "unknown"

def _sort_items_by_mod_layer_width(items):
    """
    items: list[(key, value)] where value ~ np.ndarray
    Sort by modality → numeric layer index → width (D) → string key.
    """
    def sk(kv):
        k, v = kv
        arr = np.array(v)
        D = int(arr.shape[-1]) if arr.ndim >= 1 else 0
        mod = _modality_from_key_and_dim(k, D)
        num = _extract_layer_number(k)
        num = num if num is not None else 10**9  # non-numeric go last within modality
        return (MOD_PRIORITY.get(mod, 3), num, D, str(k))
    return sorted(items, key=sk)

def _sort_layer_key_full(k: str):
    """
    Composite keys look like 'vision_12_D1408'.
    Sort by modality → layer num → width.
    """
    base, d_str = str(k).split("_D")
    mod, num_str = base.split("_", 1)
    return (MOD_PRIORITY.get(mod, 3), int(num_str), int(d_str))

# ───────────────────────────────────────────────────────────────────────────────
# SCATTER/PCA PLOTS
# ───────────────────────────────────────────────────────────────────────────────

def plot_layer_activations(activations: np.ndarray, labels: np.ndarray, layer_id, gdv_value: float,
                           output_dir='results/_gdv/plots', pooling_method=""):
    labels = np.array(labels)
    pca = PCA(n_components=2)
    activations_2d = pca.fit_transform(activations)

    plt.figure(figsize=(6, 6))
    unique_groups = np.unique(labels)
    for group in unique_groups:
        group_indices = np.where(labels == group)[0]
        plt.scatter(activations_2d[group_indices, 0], activations_2d[group_indices, 1], label=str(group))
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"{layer_id} ({pooling_method})\nGDV = {gdv_value:.4f}")
    plt.legend(); plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    safe_layer = re.sub(r'[^A-Za-z0-9_\-+=.]', '_', str(layer_id))
    plot_path = os.path.join(output_dir, f"layer_{safe_layer}_activations.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()

def _save_scatter(X2, labels, title, out_png):
    plt.figure(figsize=(6, 6))
    for g in np.unique(labels):
        m = labels == g
        plt.scatter(X2[m, 0], X2[m, 1], label=str(g))
    plt.xlabel("PCx"); plt.ylabel("PCy")
    plt.title(title)
    plt.legend(); plt.grid(True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_png}")

def plot_layer_activations_multi_pca(
    activations: np.ndarray,
    labels: np.ndarray,
    layer_id,
    gdv_euc: float,
    gdv_cos: float,
    output_dir='results/_gdv/plots',
    pooling_method="",
    pca_setups=(2, 4),
):
    labels = np.array(labels)
    safe_layer = re.sub(r'[^A-Za-z0-9_\-+=.]', '_', str(layer_id))
    base_dir = os.path.join(output_dir, safe_layer)
    os.makedirs(base_dir, exist_ok=True)

    for n in pca_setups:
        n = int(n)
        pca = PCA(n_components=n)
        Xn = pca.fit_transform(activations)
        # define which pairs to plot
        pairs = []
        if n == 2:
            pairs = [(0, 1)]
        elif n >= 4:
            pairs = [(0, 1), (1, 2), (2, 3)]
        else:
            pairs = [(i, i+1) for i in range(n-1)]

        for (a, b) in pairs:
            X2 = Xn[:, [a, b]]
            title = f"{layer_id} ({pooling_method})\nGDV (Euc)={gdv_euc:.4f} | (Cos)={gdv_cos:.4f}\nPC{a+1} vs PC{b+1} (PCA {n})"
            out_png = os.path.join(base_dir, f"PC{a+1}_vs_PC{b+1}__PCA{n}.png")
            _save_scatter(X2, labels, title, out_png)

# ───────────────────────────────────────────────────────────────────────────────
# ANIMATION
# ───────────────────────────────────────────────────────────────────────────────

def compute_gdv(X: np.ndarray, labels: np.ndarray) -> float:
    return compute_gdv_metric(X, labels, metric="euclidean")["gdv"] 

def animate_layers_smooth(activations_dict: dict, semantic_labels: np.ndarray,
                          hold_count: int = 3, interp_count: int = 5, interval: int = 500):
    sorted_layers = sorted(activations_dict.keys())
    pca_data = {}
    gdv_data = {}
    for layer in sorted_layers:
        X = activations_dict[layer].numpy()
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        pca_data[layer] = X_2d
        gdv_data[layer] = compute_gdv(X, semantic_labels)

    schedule = []
    L = len(sorted_layers)
    for i in range(L - 1):
        schedule += [("recorded", i, 0)] * hold_count
        for j in range(1, interp_count):
            frac = j / (interp_count - 1)
            schedule.append(("interp", i, frac))
    schedule += [("recorded", L - 1, 0)] * hold_count

    unique_groups = np.unique(semantic_labels)
    colors = plt.cm.get_cmap("tab10", len(unique_groups))

    fig, ax = plt.subplots(figsize=(6, 6))
    scatters = {}
    for idx, group in enumerate(unique_groups):
        scat = ax.scatter([], [], color=colors(idx), label=f"Group {group}")
        scatters[group] = scat
    title = ax.set_title("")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.legend()

    all_data = np.concatenate(list(pca_data.values()), axis=0)
    x_min, x_max = all_data[:,0].min(), all_data[:,0].max()
    y_min, y_max = all_data[:,1].min(), all_data[:,1].max()
    ax.set_xlim(x_min - 0.1*(x_max - x_min), x_max + 0.1*(x_max - x_min))
    ax.set_ylim(y_min - 0.1*(y_max - y_min), y_max + 0.1*(y_max - y_min))

    def init():
        for group in unique_groups:
            scatters[group].set_offsets(np.empty((0, 2)))
        title.set_text("")
        return list(scatters.values()) + [title]

    schedule_len = len(schedule)
    def update(frame):
        mode, base_idx, frac = schedule[frame % schedule_len]
        if mode == "recorded":
            cur_layer = sorted_layers[base_idx]
            data_2d = pca_data[cur_layer]
            cur_gdv = gdv_data[cur_layer]
            title_text = f"Layer {cur_layer}: GDV = {cur_gdv:.4f}"
        else:
            layer_a = sorted_layers[base_idx]
            layer_b = sorted_layers[base_idx+1]
            data_2d = (1-frac)*pca_data[layer_a] + frac*pca_data[layer_b]
            title_text = f"Transition: {layer_a}→{layer_b} (t={frac:.2f})"
        for group in unique_groups:
            mask = semantic_labels == group
            scatters[group].set_offsets(data_2d[mask])
        title.set_text(title_text)
        return list(scatters.values()) + [title]

    ani = animation.FuncAnimation(fig, update, frames=len(schedule), init_func=init,
                                  interval=interval, blit=False, repeat=True)
    plt.show()
    return ani

# ───────────────────────────────────────────────────────────────────────────────
# UMAP safety
# ───────────────────────────────────────────────────────────────────────────────

def ensure_umap_setups(umap_setups: Optional[list], n_samples: int) -> list:
    """
    Provide valid UMAP setups and filter out those that violate n_neighbors < n_samples.
    """
    if not umap_setups:
        # conservative defaults that tend to work
        base = [
            {"n_neighbors": max(3, min(15, n_samples - 1)), "min_dist": 0.1, "metric": "cosine",    "n_components": 2},
            {"n_neighbors": max(3, min(30, n_samples - 1)), "min_dist": 0.0, "metric": "euclidean", "n_components": 2},
        ]
    else:
        base = umap_setups

    safe = []
    for s in base:
        nn = int(s.get("n_neighbors", 15))
        if nn < n_samples:   # UMAP requires n_neighbors < n_samples
            safe.append(s)
    return safe

# ───────────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────────

def run_metrics(
    data_path,
    vision_use_cls: bool = True,
    language_use_first_token: bool = True,  # try "CLS proxy" for language; fallback to mean
    output_root: str = 'results/_gdv',
    do_umap: bool = True,
    do_tsne: bool = True,
    vision_layer_threshold: Optional[int] = None,
    umap_setups: Optional[list] = None,
    tsne_setups: Optional[list] = None,
):
    # Load precomputed activations
    loaded_results = np.load(data_path, allow_pickle=True).item()
    results_list = loaded_results['results']

    # ── Global class-support filter (<10% are dropped) ──────────────────────────
    orig_n = len(results_list)
    results_list, kept_labels, dropped_labels, label_counts = filter_results_by_min_support(
        results_list, label_key="interestingness", min_frac=0.10
    )
    if len(results_list) == 0:
        raise ValueError("All samples were filtered out by min support; relax the threshold or add data.")

    print(f"[class filter] total={orig_n}  kept={len(results_list)}  "
          f"kept_labels={sorted(kept_labels)}  dropped_labels={sorted(dropped_labels)}")
    print(f"[class counts] " + ", ".join(f"{k}:{label_counts[k]}" for k in sorted(label_counts)))

    # Aggregate BEFORE pooling; keep sample indices
    layer_acts = {}          # composite_key -> [np.ndarray ...]
    layer_sample_idx = {}    # composite_key -> [int ...]
    # Dimension trackers
    dim_counts_by_mod_layer = defaultdict(int)    # (mod, num, D) -> samples
    dims_by_modality = defaultdict(set)           # mod -> {D,...}

    for s_idx, result in enumerate(results_list):
        embeddings = result['embeddings']  # dict: layer_key -> activations
        items = list(embeddings.items())
        items = _sort_items_by_mod_layer_width(items)  # stable, model-faithful order

        for pos, (k, act) in enumerate(items):
            arr = np.array(act)
            D = int(arr.shape[-1]) if arr.ndim >= 1 else 0
            mod = _modality_from_key_and_dim(k, D)
            num = _extract_layer_number(k)
            if num is None:
                num = pos  # fallback to position if no number in key
            # composite key separates modality and width so we don't mix 1408 vs 5120
            comp_key = f"{mod}_{num}_D{D}"
            layer_acts.setdefault(comp_key, []).append(arr)
            layer_sample_idx.setdefault(comp_key, []).append(s_idx)
            dim_counts_by_mod_layer[(mod, num, D)] += 1
            dims_by_modality[mod].add(D)

    labels_all = np.array([r['interestingness'] for r in results_list])
    sents_all  = np.array([r['explanation'] for r in results_list])

    # Output dirs
    plots_root = os.path.join(output_root, 'plots')
    os.makedirs(plots_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    # Compute GDV per composite layer, with requested pooling rules
    gdv_all = {}
    layer_info = {}

    # Counters for how language pooling was resolved (info only)
    lang_used_first = 0
    lang_used_mean  = 0

    for comp_key in sorted(layer_acts.keys(), key=_sort_layer_key_full):
        arrs = layer_acts[comp_key]
        sidx = layer_sample_idx[comp_key]

        base, d_str = comp_key.split("_D")
        mod, num_str = base.split("_", 1)
        D = int(d_str)
        num = int(num_str)

        # ── Pool to one vector per sample, supporting vector or matrix inputs ──
        pooled = []
        pooling_method = None
        for a in arrs:
            vec, method = pool_activation(
                a, modality=mod,
                prefer_first_token=language_use_first_token,
                prefer_cls_for_vision=vision_use_cls
            )
            pooled.append(vec)
            pooling_method = method  # last one; methods should be consistent per comp_key

            # bookkeeping for language mode usage (informative counters, optional)
            if mod == "language":
                if np.asarray(a).ndim == 1:
                    lang_used_first += 1   # treat 1D as "already the decision vector"
                else:
                    if method.startswith("FirstToken"):
                        lang_used_first += 1
                    else:
                        lang_used_mean  += 1

        X = np.vstack(pooled)         # (M, D)
        y = labels_all[sidx]          # (M,)
        texts = sents_all[sidx]       # (M,)

        # --- GDV for both metrics (macro by default) ---
        gdv_both = compute_gdv_both(X, y, weighting="macro")
        gdv_all_euc = gdv_both["euclidean"]["gdv"]
        gdv_all_cos = gdv_both["cosine"]["gdv"]

        gdv_all[comp_key] = gdv_all_euc  # keep 'gdv_all' for backward-compat (euclidean)
        # store more rich info per-layer for dashboards etc.
        layer_info[comp_key] = {
            'pooling':  pooling_method,
            'modality': mod,
            'layer_num': num,
            'width_D':  D,
            'samples_used': int(X.shape[0]),
            'labels': y.tolist(),
            'texts': texts.tolist(),
            'gdv_euclidean': gdv_all_euc,
            'gdv_cosine': gdv_all_cos,
            'intra_euclidean': gdv_both["euclidean"]["intra"],
            'inter_euclidean': gdv_both["euclidean"]["inter"],
            'intra_cosine': gdv_both["cosine"]["intra"],
            'inter_cosine': gdv_both["cosine"]["inter"],
        }

        # --- PCA plots (2 & 4 comps; 1-2, 2-3, 3-4) ---
        mod_dir = os.path.join(plots_root, mod)
        os.makedirs(mod_dir, exist_ok=True)
        plot_layer_activations_multi_pca(
            activations=X,
            labels=y,
            layer_id=f"{mod} {num} (D={D})",
            gdv_euc=gdv_all_euc,
            gdv_cos=gdv_all_cos,
            output_dir=mod_dir,
            pooling_method=pooling_method,
            pca_setups=(2, 4),
        )

        # keep a 2D for any legacy dashboards, if you still need it
        pca2 = PCA(n_components=2).fit_transform(X)
        layer_info[comp_key].update({
            'x': pca2[:, 0].tolist(),
            'y': pca2[:, 1].tolist(),
        })

        # --- Dimensionality reduction runs (robust UMAP; optional robust t-SNE) ---
        if do_umap:
            try:
                safe_setups = ensure_umap_setups(umap_setups, n_samples=X.shape[0])
                if len(safe_setups) == 0 or X.shape[0] < 5:
                    layer_info[comp_key]['umap_runs'] = []
                else:
                    layer_info[comp_key]['umap_runs'] = run_umap_for_layer(
                        X=X, y=y,
                        layer_id=f"{mod} {num} (D={D})",
                        out_dir=mod_dir,
                        pooling_method=pooling_method,
                        setups=safe_setups,
                    )
            except Exception as e:
                layer_info[comp_key]['umap_runs'] = []
                layer_info[comp_key]['umap_error'] = str(e)
                print(f"[UMAP skipped] {comp_key}: {e}")

        if do_tsne:
            try:
                layer_info[comp_key]['tsne_runs'] = run_tsne_for_layer(
                    X=X, y=y,
                    layer_id=f"{mod} {num} (D={D})",
                    out_dir=mod_dir,
                    pooling_method=pooling_method,
                    setups=tsne_setups,   # None = defaults inside the module
                )
            except Exception as e:
                layer_info[comp_key]['tsne_runs'] = []
                layer_info[comp_key]['tsne_error'] = str(e)
                print(f"[t-SNE skipped] {comp_key}: {e}")

    # ── Save GDV, layer info, and dimension reports ────────────────────────────
    sorted_layers = sorted(gdv_all.keys(), key=_sort_layer_key_full)
    max_gdv_layer = max(gdv_all, key=gdv_all.get)
    meta = {
        'labels':       labels_all.tolist(),
        'total_layers': len(sorted_layers),
        'max_gdv_layer': max_gdv_layer,
        'language_first_token_used': lang_used_first,
        'language_mean_fallback_used': lang_used_mean,
        'dropped_labels_lt_25pct': sorted(list(dropped_labels)),
        'kept_labels_ge_25pct':     sorted(list(kept_labels)),
    }

    # ── Aggregate UMAP / t-SNE summaries across layers ─────────────────────────
    umap_rows = []
    tsne_rows = []
    best_umap_by_layer = {}
    best_tsne_by_layer = {}

    for k in sorted_layers:
        mod, rest = k.split("_", 1)
        num_str, d_str = rest.split("_D")
        info = layer_info[k]

        # ---- UMAP rows ----
        if do_umap and 'umap_runs' in info and info['umap_runs']:
            def _score_u(r):
                tw = r.get("trustworthiness", float("-inf"))
                sil = r.get("silhouette_on_2D", None)
                sil = -1e9 if sil is None else sil
                return (tw, sil)

            best_umap = max(info['umap_runs'], key=_score_u)
            best_umap_by_layer[k] = best_umap

            for r in info['umap_runs']:
                s = r["setup"]
                tag = f"UMAP_n{s.get('n_neighbors')}__md{s.get('min_dist')}__{s.get('metric')}__nc{s.get('n_components', 2)}"
                umap_rows.append([
                    mod, int(num_str), int(d_str), k, tag,
                    "UMAP", s.get("n_neighbors"), s.get("min_dist"), s.get("metric"), s.get("n_components", 2),
                    r.get("trustworthiness"), r.get("silhouette_on_2D"),
                    r.get("coords_csv"), r.get("plot"),
                ])

        # ---- t-SNE rows ----
        if do_tsne and 'tsne_runs' in info and info['tsne_runs']:
            def _score_t(r):
                tw = r.get("trustworthiness", float("-inf"))
                sil = r.get("silhouette_on_2D", None)
                sil = -1e9 if sil is None else sil
                return (tw, sil)

            best_tsne = max(info['tsne_runs'], key=_score_t)
            best_tsne_by_layer[k] = best_tsne

            for r in info['tsne_runs']:
                s = r["setup"]
                tag = f"TSNE_p{s.get('perplexity')}__lr{s.get('learning_rate')}__{s.get('metric')}__init{s.get('init')}"
                tsne_rows.append([
                    mod, int(num_str), int(d_str), k, tag,
                    "TSNE", s.get("perplexity"), s.get("learning_rate"), s.get("metric"), s.get("init"),
                    r.get("trustworthiness"), r.get("silhouette_on_2D"),
                    r.get("coords_csv"), r.get("plot"),
                ])

    # Write UMAP summary CSV
    if umap_rows:
        umap_csv = os.path.join(output_root, 'umap_summary.csv')
        with open(umap_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'Modality','Layer_Num','Width_D','LayerKey','Setup_Tag',
                'Algo','n_neighbors','min_dist','metric','n_components',
                'Trustworthiness','Silhouette2D','Coords_CSV','Plot_PNG'
            ])
            w.writerows(umap_rows)
        print(f"Saved UMAP summary to {umap_csv}")

    # Write t-SNE summary CSV
    if tsne_rows:
        tsne_csv = os.path.join(output_root, 'tsne_summary.csv')
        with open(tsne_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'Modality','Layer_Num','Width_D','LayerKey','Setup_Tag',
                'Algo','perplexity','learning_rate','metric','init',
                'Trustworthiness','Silhouette2D','Coords_CSV','Plot_PNG'
            ])
            w.writerows(tsne_rows)
        print(f"Saved t-SNE summary to {tsne_csv}")

    # GDV CSV
    csv_path = os.path.join(output_root, 'gdv_values.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow([
            'Modality','Layer_Num','Width_D','LayerKey',
            'GDV_Euclidean','Intra_Euclidean','Inter_Euclidean',
            'GDV_Cosine','Intra_Cosine','Inter_Cosine',
            'Pooling','Samples_Used'
        ])
        for k in sorted_layers:
            mod, rest = k.split("_", 1)
            num_str, d_str = rest.split("_D")
            info = layer_info[k]
            w.writerow([
                mod, int(num_str), int(d_str), k,
                info['gdv_euclidean'], info['intra_euclidean'], info['inter_euclidean'],
                info['gdv_cosine'],    info['intra_cosine'],    info['inter_cosine'],
                info['pooling'],       info['samples_used']
            ])
    
    meta.update({
        'best_umap_by_layer': best_umap_by_layer,   # dict: layerKey -> run dict (setup, trust, silhouette, paths)
        'best_tsne_by_layer': best_tsne_by_layer,
    })

    # Dimensions by (modality, layer, D)
    dims_csv = os.path.join(output_root, 'dimensions_by_layer.csv')
    with open(dims_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Modality', 'Layer_Num', 'Width_D', 'Samples_Used'])
        for (mod, num, D), cnt in sorted(dim_counts_by_mod_layer.items(),
                                         key=lambda x: (MOD_PRIORITY.get(x[0][0],3), x[0][1], x[0][2])):
            w.writerow([mod, num, D, cnt])

    # Unique D per modality
    mods_csv = os.path.join(output_root, 'dimensions_by_modality.csv')
    with open(mods_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Modality', 'Unique_Widths_D'])
        for mod in sorted(dims_by_modality.keys(), key=lambda m: MOD_PRIORITY.get(m, 3)):
            w.writerow([mod, ";".join(str(d) for d in sorted(dims_by_modality[mod]))])

    # PKL
    pkl_path = os.path.join(output_root, 'gdv.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'sorted_layers': sorted_layers,
            'layer_data':    layer_info,                 # has both metrics already
            'gdv_per_layer': {k: layer_info[k]['gdv_euclidean'] for k in sorted_layers},
            'gdv_per_layer_cosine': {k: layer_info[k]['gdv_cosine'] for k in sorted_layers},
            'meta':          meta
        }, f)

    # Console summaries
    print(f"\nLanguage pooling usage: FirstToken={lang_used_first}, MeanFallback={lang_used_mean}")
    for mod in sorted(dims_by_modality.keys(), key=lambda m: MOD_PRIORITY.get(m, 3)):
        print(f"Modality '{mod}' widths: {sorted(dims_by_modality[mod])}")

    print(f"\nSaved GDV CSV to {csv_path}")
    print(f"Saved dimensions per layer to {dims_csv}")
    print(f"Saved dimensions per modality to {mods_csv}")
    print(f"Saved GDV and layer data to {pkl_path}")
    print(f"Plots under: {plots_root}")

    return gdv_all
