#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import csv
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import pickle
from collections import defaultdict

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS: layer key → numeric index, modality, robust sorting
# ───────────────────────────────────────────────────────────────────────────────

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
# GDV FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

def compute_mean_intra_class_distance(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.array([str(label).strip().lower() for label in labels])
    unique_labels, _ = np.unique(labels, return_counts=True)
    intra_dists = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) < 2:
            continue
        subset = X[idx]
        dists = pdist(subset, metric="euclidean")
        intra_dists.append(np.mean(dists))
    return float(np.mean(intra_dists)) if intra_dists else 0.0

def compute_mean_inter_class_distance(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.array([str(label).strip().lower() for label in labels])
    unique_labels, _ = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        return 0.0
    inter_dists = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            idx1 = np.where(labels == unique_labels[i])[0]
            idx2 = np.where(labels == unique_labels[j])[0]
            if len(idx1) == 0 or len(idx2) == 0:
                continue
            diff = X[idx1][:, None, :] - X[idx2][None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            inter_dists.append(np.mean(dists))
    return float(np.mean(inter_dists)) if inter_dists else 0.0

def compute_gdv(X: np.ndarray, labels: np.ndarray) -> float:
    # z-score per dim, scale by 1/2 (per paper)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    Xz = (X - mu) / sigma
    Xz *= 0.5

    intra = compute_mean_intra_class_distance(Xz, labels)
    inter = compute_mean_inter_class_distance(Xz, labels)

    D = Xz.shape[1]
    K = len(np.unique(labels))
    if K < 2:
        return 0.0
    gdv = (1 / np.sqrt(D)) * ((1 / K) * intra - (2 / (K * (K - 1))) * inter)
    return float(gdv)

# ───────────────────────────────────────────────────────────────────────────────
# PLOTTING
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

# ───────────────────────────────────────────────────────────────────────────────
# ANIMATION (unchanged)
# ───────────────────────────────────────────────────────────────────────────────

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
# MAIN: enforce correct order + per-sample pooling + composite keys
# ───────────────────────────────────────────────────────────────────────────────

def run_gdv_experiment(
    data_path,
    vision_use_cls: bool = True,
    language_use_first_token: bool = True,  # try "CLS proxy" for language; fallback to mean
    output_root: str = 'results/_gdv'
):
    # Load precomputed activations
    loaded_results = np.load(data_path, allow_pickle=True).item()
    results_list = loaded_results['results']

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

    # Counters for how language pooling was resolved
    lang_used_first = 0
    lang_used_mean  = 0

    for comp_key in sorted(layer_acts.keys(), key=_sort_layer_key_full):
        arrs = layer_acts[comp_key]
        sidx = layer_sample_idx[comp_key]

        base, d_str = comp_key.split("_D")
        mod, num_str = base.split("_", 1)
        D = int(d_str)
        num = int(num_str)

        pooled = []
        if mod == "vision":
            # Strict CLS for Vision (as requested); fallback to mean if something odd shows up
            pooling_method = "CLS"
            for a in arrs:
                if a.ndim == 1:        # already pooled
                    vec = a
                    pooling_method = "CLS (input-1D)"
                else:
                    try:
                        vec = a[0]
                    except Exception:
                        vec = a.mean(axis=0)
                        pooling_method = "Mean (vision fallback)"
                pooled.append(vec)
        else:
            # Language / Projector: prefer first token as CLS-proxy if tokens exist; else mean
            # We also count how often we used each variant.
            if language_use_first_token:
                pooling_method = "FirstToken (CLS-proxy)"
                for a in arrs:
                    if a.ndim == 1:
                        vec = a
                        pooling_method = "FirstToken (input-1D)"
                        lang_used_mean += 1  # treat as non-token case
                    else:
                        try:
                            vec = a[0]
                            lang_used_first += 1
                        except Exception:
                            vec = a.mean(axis=0)
                            lang_used_mean += 1
                            pooling_method = "Mean (fallback)"
                    pooled.append(vec)
            else:
                pooling_method = "Mean"
                for a in arrs:
                    vec = a if a.ndim == 1 else a.mean(axis=0)
                    pooled.append(vec)
                # we won't modify counters in this branch

        X = np.vstack(pooled)         # (M, D)
        y = labels_all[sidx]          # (M,)
        texts = sents_all[sidx]       # (M,)

        gdv = compute_gdv(X, y)
        gdv_all[comp_key] = gdv

        # Plots into modality subfolder
        mod_dir = os.path.join(plots_root, mod)
        os.makedirs(mod_dir, exist_ok=True)
        plot_layer_activations(
            activations=X,
            labels=y,
            layer_id=f"{mod} {num} (D={D})",
            gdv_value=gdv,
            output_dir=mod_dir,
            pooling_method=pooling_method
        )

        # 2D for dashboards
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        layer_info[comp_key] = {
            'x':        X2[:, 0].tolist(),
            'y':        X2[:, 1].tolist(),
            'group':    y.tolist(),
            'sentence': texts.tolist(),
            'pooling':  pooling_method,
            'modality': mod,
            'layer_num': num,
            'width_D':  D,
            'samples_used': int(X.shape[0]),
        }

    # ── Save GDV, layer info, and dimension reports ────────────────────────────
    sorted_layers = sorted(gdv_all.keys(), key=_sort_layer_key_full)
    max_gdv_layer = max(gdv_all, key=gdv_all.get)
    meta = {
        'labels':       labels_all.tolist(),
        'total_layers': len(sorted_layers),
        'max_gdv_layer': max_gdv_layer,
        'language_first_token_used': lang_used_first,
        'language_mean_fallback_used': lang_used_mean,
    }

    # GDV CSV
    csv_path = os.path.join(output_root, 'gdv_values.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['Modality', 'Layer_Num', 'Width_D', 'LayerKey', 'GDV', 'Pooling', 'Samples_Used'])
        for k in sorted_layers:
            mod, rest = k.split("_", 1)        # e.g. "vision", "12_D1408"
            num_str, d_str = rest.split("_D")
            w.writerow([mod, int(num_str), int(d_str), k, gdv_all[k],
                        layer_info[k]['pooling'], layer_info[k]['samples_used']])

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
            'sorted_layers': sorted_layers,   # composite keys in correct order
            'layer_data':    layer_info,
            'gdv_per_layer': gdv_all,
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
