#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional

# ---------- config ----------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(BASE_DIR, "res")     # adjust if needed
RUN_GLOB     = os.path.join(RESULTS_ROOT, "gdv_*")
OUT_SUBDIR   = "plots_overview"

MOD_PRIORITY = {"vision": 0, "projector": 1, "language": 2, "unknown": 3}

# ---------- safe unpickler for numpy._core ----------
class RemappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

def load_pickle_remap(path: str):
    with open(path, "rb") as f:
        return RemappingUnpickler(f).load()

# ---------- helpers ----------
def _ensure_out_dir(run_dir: str) -> str:
    out = os.path.join(run_dir, OUT_SUBDIR)
    os.makedirs(out, exist_ok=True)
    return out

def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    header = rows[0]
    out = []
    for r in rows[1:]:
        d = {}
        for k, v in zip(header, r):
            d[k] = v
        out.append(d)
    return out

def _sort_key(modality: str, layer_num: int, width_d: int, layer_key: str):
    return (MOD_PRIORITY.get(modality, 3), int(layer_num), int(width_d), str(layer_key))

def _build_order_from_csv(rows: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """
    Returns ordered list of (LayerKey, Modality) derived from a summary CSV
    """
    uniq = {}
    for d in rows:
        try:
            mod = d["Modality"]
            num = int(d["Layer_Num"])
            D   = int(d["Width_D"])
            key = d["LayerKey"]
        except Exception:
            continue
        uniq[key] = (mod, num, D, key)
    ordered = sorted(uniq.values(), key=lambda t: _sort_key(t[0], t[1], t[2], t[3]))
    return [(k, m) for (m, _, _, k) in ordered]

def _build_order_from_pkl(pkl_path: str) -> Optional[List[Tuple[str, str]]]:
    """
    Prefer exact model order from gdv.pkl: returns list of (key, modality)
    """
    if not os.path.exists(pkl_path):
        return None
    obj = load_pickle_remap(pkl_path)
    if not all(k in obj for k in ["sorted_layers", "layer_data"]):
        return None
    ordered = []
    for k in obj["sorted_layers"]:
        try:
            base, d = str(k).split("_D")
            mod, _num = base.split("_", 1)
        except Exception:
            mod = "unknown"
        ordered.append((k, mod))
    return ordered

def _find_switch_index(order: List[Tuple[str, str]]) -> Optional[float]:
    """
    Returns x-position (float) where modality first becomes 'language'.
    We place the line between the last non-language and first language layer (i+0.5).
    """
    last_non_lang = -1
    for i, (_, mod) in enumerate(order):
        if mod.lower() == "language":
            last_non_lang = i - 1
            return max(0, last_non_lang + 0.5)
    return None

def _collect_sil_by_setup(rows: List[Dict[str, str]], order: List[Tuple[str, str]]) -> Dict[str, List[Optional[float]]]:
    """
    Build per-setup silhouette arrays aligned to 'order'.
    """
    idx = {k: i for i, (k, _) in enumerate(order)}
    by_setup: Dict[str, Dict[int, float]] = defaultdict(dict)

    for d in rows:
        key = d.get("LayerKey")
        if key not in idx:
            continue
        tag = d.get("Setup_Tag", "")  # UMAP_n... or TSNE_p...
        val = d.get("Silhouette2D", "")
        try:
            sil = float(val) if val != "" and val.lower() != "nan" else np.nan
        except Exception:
            sil = np.nan
        by_setup[tag][idx[key]] = sil

    out: Dict[str, List[Optional[float]]] = {}
    L = len(order)
    for tag, m in by_setup.items():
        arr = [np.nan] * L
        for i, v in m.items():
            arr[i] = v
        out[tag] = arr
    return out

def _collect_gdv(rows: List[Dict[str, str]], order: List[Tuple[str, str]]) -> List[Optional[float]]:
    """
    Pull GDV_Euclidean per layer aligned to 'order' (single curve).
    """
    idx = {k: i for i, (k, _) in enumerate(order)}
    arr = [np.nan] * len(order)
    for d in rows:
        key = d.get("LayerKey")
        if key not in idx:
            continue
        try:
            arr[idx[key]] = float(d.get("GDV_Euclidean", "nan"))
        except Exception:
            pass
    return arr

def _plot_curves(
    x: List[int],
    curves: Dict[str, List[Optional[float]]],
    switch_x: Optional[float],
    title: str,
    ylabel: str,
    out_path: str,
    legend_ncol: int = 2,
):
    plt.figure(figsize=(12, 5))
    for tag, y in sorted(curves.items(), key=lambda kv: kv[0]):
        y_arr = np.array(y, dtype=float)
        plt.plot(x, y_arr, marker="o", linewidth=1.5, label=tag)
    if switch_x is not None:
        plt.axvline(switch_x, color="k", linestyle="--", linewidth=1.2, alpha=0.8)
        plt.text(switch_x, plt.ylim()[1], " vision → language", rotation=90,
                 va="top", ha="left", fontsize=9, color="k", alpha=0.8)
    plt.title(title)
    plt.xlabel("Layer order")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if curves:
        plt.legend(loc="best", fontsize=8, ncol=legend_ncol)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved: {out_path}")

def process_run(run_dir: str):
    run_name = os.path.basename(run_dir)
    out_dir  = _ensure_out_dir(run_dir)

    pkl_path = os.path.join(run_dir, "gdv.pkl")
    umap_csv = os.path.join(run_dir, "umap_summary.csv")
    tsne_csv = os.path.join(run_dir, "tsne_summary.csv")
    gdv_csv  = os.path.join(run_dir, "gdv_values.csv")

    # Build layer order (prefer exact order from pkl)
    order = _build_order_from_pkl(pkl_path)
    if order is None:
        # fallback: try umap, else tsne, else gdv
        rows = _read_csv(umap_csv) or _read_csv(tsne_csv) or _read_csv(gdv_csv)
        if not rows:
            print(f"[{run_name}] No CSVs found; skipping.")
            return
        order = _build_order_from_csv(rows)

    x = list(range(1, len(order) + 1))
    switch_x = _find_switch_index(order)

    # UMAP curves (Silhouette2D per setup)
    u_rows = _read_csv(umap_csv)
    if u_rows:
        umap_curves = _collect_sil_by_setup(u_rows, order)
        if umap_curves:
            _plot_curves(
                x, umap_curves, switch_x,
                title=f"{run_name} — UMAP separability (Silhouette on 2D)",
                ylabel="Silhouette score (2D)",
                out_path=os.path.join(out_dir, "umap_separability_all_setups.png"),
            )
            # per-setup files
            for tag, arr in umap_curves.items():
                _plot_curves(
                    x, {tag: arr}, switch_x,
                    title=f"{run_name} — UMAP separability: {tag}",
                    ylabel="Silhouette score (2D)",
                    out_path=os.path.join(out_dir, f"umap_separability__{tag}.png"),
                    legend_ncol=1,
                )

    # t-SNE curves (Silhouette2D per setup)
    t_rows = _read_csv(tsne_csv)
    if t_rows:
        tsne_curves = _collect_sil_by_setup(t_rows, order)
        if tsne_curves:
            _plot_curves(
                x, tsne_curves, switch_x,
                title=f"{run_name} — t-SNE separability (Silhouette on 2D)",
                ylabel="Silhouette score (2D)",
                out_path=os.path.join(out_dir, "tsne_separability_all_setups.png"),
            )
            # per-setup files
            for tag, arr in tsne_curves.items():
                _plot_curves(
                    x, {tag: arr}, switch_x,
                    title=f"{run_name} — t-SNE separability: {tag}",
                    ylabel="Silhouette score (2D)",
                    out_path=os.path.join(out_dir, f"tsne_separability__{tag}.png"),
                    legend_ncol=1,
                )

    # PCA / GDV curve (single)
    g_rows = _read_csv(gdv_csv)
    if g_rows:
        gdv_curve = _collect_gdv(g_rows, order)
        if any(np.isfinite(v) for v in gdv_curve):
            _plot_curves(
                x, {"GDV (Euclidean)": gdv_curve}, switch_x,
                title=f"{run_name} — PCA baseline separability: GDV across layers",
                ylabel="GDV (Euclidean)",
                out_path=os.path.join(out_dir, "gdv_euclidean_across_layers.png"),
                legend_ncol=1,
            )

def main():
    run_dirs = sorted([p for p in glob.glob(RUN_GLOB) if os.path.isdir(p)])
    if not run_dirs:
        raise FileNotFoundError(f"No runs found under: {RUN_GLOB}")
    for rd in run_dirs:
        process_run(rd)

if __name__ == "__main__":
    main()
