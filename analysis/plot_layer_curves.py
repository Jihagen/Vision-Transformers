#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot GDV, UMAP trustworthiness, and t-SNE trustworthiness across all layers
for each Gender × Emotion experiment.  Reads pre-computed gdv.pkl / *_summary.csv
so no re-computation is needed.

Outputs under results/experiments/gender_emotion_analysis/layer_curves/
"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

METRICS_DIR = 'results/experiments/metrics'
OUT_DIR     = 'results/experiments/gender_emotion_analysis/layer_curves'

EXPERIMENTS = [
    'female_anger', 'female_fear', 'female_disgust', 'female_sad',
    'female_amusement', 'female_awe', 'female_contentment',
    'female_excitement', 'male_anger',
]
SHORT = {
    'female_anger': 'F-Anger',    'female_fear': 'F-Fear',
    'female_disgust': 'F-Disgust','female_sad': 'F-Sad',
    'female_amusement': 'F-Amus', 'female_awe': 'F-Awe',
    'female_contentment': 'F-Con','female_excitement': 'F-Exc',
    'male_anger': 'M-Anger',
}

MOD_ORDER = {'vision': 0, 'projector': 1, 'language': 2, 'unknown': 3}

def _sort_key(layer_key: str):
    parts = layer_key.split('_D')
    base  = parts[0]
    mod, num_str = base.split('_', 1)
    return (MOD_ORDER.get(mod, 3), int(num_str))


def load_gdv(exp: str) -> pd.DataFrame:
    key = f'gender_{exp}'
    path = os.path.join(METRICS_DIR, key, 'gdv_values.csv')
    df = pd.read_csv(path)
    df['sort_key'] = df['LayerKey'].apply(_sort_key)
    df = df.sort_values('sort_key').reset_index(drop=True)
    df['layer_pos'] = range(len(df))
    return df


def load_umap(exp: str) -> pd.DataFrame:
    key  = f'gender_{exp}'
    path = os.path.join(METRICS_DIR, key, 'umap_summary.csv')
    df   = pd.read_csv(path)
    # Best run per layer = max trustworthiness
    best = df.groupby('LayerKey')['Trustworthiness'].max().reset_index()
    best.columns = ['LayerKey', 'TW_best']
    best['sort_key'] = best['LayerKey'].apply(_sort_key)
    return best.sort_values('sort_key').reset_index(drop=True)


def load_tsne(exp: str) -> pd.DataFrame:
    key  = f'gender_{exp}'
    path = os.path.join(METRICS_DIR, key, 'tsne_summary.csv')
    df   = pd.read_csv(path)
    best = df.groupby('LayerKey')['Trustworthiness'].max().reset_index()
    best.columns = ['LayerKey', 'TW_best']
    best['sort_key'] = best['LayerKey'].apply(_sort_key)
    return best.sort_values('sort_key').reset_index(drop=True)


def _modality_spans(gdv_df: pd.DataFrame):
    """Return list of (modality, start_pos, end_pos) for shading."""
    spans = []
    cur_mod, start = None, 0
    for _, row in gdv_df.iterrows():
        mod = row['Modality']
        pos = row['layer_pos']
        if mod != cur_mod:
            if cur_mod is not None:
                spans.append((cur_mod, start, pos - 1))
            cur_mod, start = mod, pos
    if cur_mod is not None:
        spans.append((cur_mod, start, gdv_df['layer_pos'].max()))
    return spans


MOD_COLORS = {'vision': '#e8f4fd', 'projector': '#fff3cd', 'language': '#e8f8e8'}


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: GDV across all layers, all experiments — one panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_gdv_all(out_dir: str):
    fig, ax = plt.subplots(figsize=(16, 5))
    cmap = plt.colormaps['tab10']

    # Use common layer ordering from first experiment
    ref_df = load_gdv(EXPERIMENTS[0])
    layer_keys = ref_df['LayerKey'].tolist()
    layer_pos  = list(range(len(layer_keys)))

    # Modality shading
    spans = _modality_spans(ref_df)
    for mod, s, e in spans:
        color = MOD_COLORS.get(mod, '#f0f0f0')
        ax.axvspan(s - 0.5, e + 0.5, color=color, alpha=0.4, zorder=0)

    for i, exp in enumerate(EXPERIMENTS):
        df = load_gdv(exp)
        style = '--' if 'male' in exp else '-'
        lw    = 2.0 if exp in ('female_anger', 'male_anger') else 1.1
        gdv_vals = df.set_index('LayerKey').reindex(layer_keys)['GDV_Euclidean'].values
        ax.plot(layer_pos, gdv_vals, style, color=cmap(i / len(EXPERIMENTS)),
                linewidth=lw, label=SHORT[exp], alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.7, linestyle=':', alpha=0.5)
    ax.set_xticks(layer_pos[::4])
    ax.set_xticklabels([layer_keys[p] for p in layer_pos[::4]], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('GDV (Euclidean)', fontsize=10)
    ax.set_title('GDV Across All Layers — Gender × Emotion Experiments\n'
                 '(negative = better label separation; shading: vision / projector / language)',
                 fontsize=11)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Legend for modality shading
    patches = [mlines.Line2D([], [], color=MOD_COLORS[m], linewidth=8,
                              label=m.capitalize(), alpha=0.6)
               for m in ('vision', 'projector', 'language') if m in MOD_COLORS]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0],
              labels=['Vision', 'Projector', 'Language'] + ax.get_legend_handles_labels()[1],
              bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    out = os.path.join(out_dir, 'gdv_all_layers_all_experiments.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: GDV by modality section — 3-panel (vision / projector / language)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gdv_by_modality(out_dir: str):
    cmap = plt.colormaps['tab10']
    ref_df = load_gdv(EXPERIMENTS[0])

    for mod in ('vision', 'language'):
        mod_df  = ref_df[ref_df['Modality'] == mod]
        if mod_df.empty:
            continue
        lkeys = mod_df['LayerKey'].tolist()
        xpos  = list(range(len(lkeys)))

        fig, ax = plt.subplots(figsize=(12, 4))
        for i, exp in enumerate(EXPERIMENTS):
            df   = load_gdv(exp)
            vals = df.set_index('LayerKey').reindex(lkeys)['GDV_Euclidean'].values
            style = '--' if 'male' in exp else '-'
            lw    = 2.0 if exp in ('female_anger', 'male_anger') else 1.1
            ax.plot(xpos, vals, style, color=cmap(i / len(EXPERIMENTS)),
                    linewidth=lw, label=SHORT[exp], alpha=0.85)

        ax.axhline(0, color='black', linewidth=0.7, linestyle=':', alpha=0.5)
        ax.set_xticks(xpos[::2] if len(xpos) > 10 else xpos)
        ax.set_xticklabels([lkeys[p].split('_')[1] for p in (xpos[::2] if len(xpos) > 10 else xpos)],
                           rotation=0, fontsize=8)
        ax.set_xlabel(f'{mod.capitalize()} layer number', fontsize=9)
        ax.set_ylabel('GDV (Euclidean)', fontsize=10)
        ax.set_title(f'GDV — {mod.capitalize()} Layers', fontsize=11)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        out = os.path.join(out_dir, f'gdv_{mod}_layers.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: UMAP trustworthiness across all layers
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap_trustworthiness(out_dir: str):
    fig, ax = plt.subplots(figsize=(16, 5))
    cmap = plt.colormaps['tab10']

    ref_gdv = load_gdv(EXPERIMENTS[0])
    layer_keys = ref_gdv['LayerKey'].tolist()
    spans = _modality_spans(ref_gdv)
    for mod, s, e in spans:
        ax.axvspan(s - 0.5, e + 0.5, color=MOD_COLORS.get(mod, '#f0f0f0'), alpha=0.35, zorder=0)

    for i, exp in enumerate(EXPERIMENTS):
        try:
            umap_df = load_umap(exp)
        except Exception:
            continue
        style = '--' if 'male' in exp else '-'
        lw    = 2.0 if exp in ('female_anger', 'male_anger') else 1.1
        umap_pos = [layer_keys.index(k) for k in umap_df['LayerKey'] if k in layer_keys]
        umap_tw  = [umap_df.loc[umap_df['LayerKey'] == layer_keys[p], 'TW_best'].values[0]
                    for p in umap_pos]
        ax.plot(umap_pos, umap_tw, style, color=cmap(i / len(EXPERIMENTS)),
                linewidth=lw, label=SHORT[exp], alpha=0.85)

    ax.set_xticks(list(range(0, len(layer_keys), 4)))
    ax.set_xticklabels([layer_keys[p] for p in range(0, len(layer_keys), 4)],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('UMAP Trustworthiness (best setup)', fontsize=10)
    ax.set_title('UMAP Trustworthiness Across All Layers', fontsize=11)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, 'umap_trustworthiness_all_layers.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: GDV vs UMAP trustworthiness — scatter per experiment
# ─────────────────────────────────────────────────────────────────────────────

def plot_gdv_vs_umap(out_dir: str):
    ncols = 3
    nrows = (len(EXPERIMENTS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()
    cmap = plt.colormaps['tab10']

    for i, exp in enumerate(EXPERIMENTS):
        ax = axes[i]
        try:
            gdv_df  = load_gdv(exp)
            umap_df = load_umap(exp)
        except Exception as e:
            ax.set_title(f'{SHORT[exp]} — data missing'); continue

        merged = gdv_df.merge(umap_df[['LayerKey', 'TW_best']], on='LayerKey', how='inner')
        colors = [cmap(MOD_ORDER.get(m, 3) / 4) for m in merged['Modality']]
        sc = ax.scatter(merged['GDV_Euclidean'], merged['TW_best'],
                        c=colors, s=25, alpha=0.7, edgecolors='white', linewidths=0.4)
        ax.set_xlabel('GDV (Euclidean)', fontsize=8)
        ax.set_ylabel('UMAP Trustworthiness', fontsize=8)
        ax.set_title(SHORT[exp], fontsize=9)
        ax.grid(alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    for ax in axes[len(EXPERIMENTS):]:
        ax.set_visible(False)

    # Modality legend
    handles = [plt.scatter([], [], c=cmap(MOD_ORDER[m] / 4), s=30, label=m.capitalize())
               for m in ('vision', 'language')]
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(1, 0), fontsize=8)
    plt.suptitle('GDV vs UMAP Trustworthiness per Layer', fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, 'gdv_vs_umap_scatter.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: Per-experiment panel — GDV + UMAP on twin axes, all layers
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_experiment_panels(out_dir: str):
    ncols = 3
    nrows = (len(EXPERIMENTS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 3.5))
    axes = axes.flatten()

    ref_gdv    = load_gdv(EXPERIMENTS[0])
    layer_keys = ref_gdv['LayerKey'].tolist()

    for i, exp in enumerate(EXPERIMENTS):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        try:
            gdv_df  = load_gdv(exp)
            umap_df = load_umap(exp)
        except Exception:
            ax1.set_title(f'{SHORT[exp]} — missing'); continue

        gdv_vals = gdv_df.set_index('LayerKey').reindex(layer_keys)['GDV_Euclidean'].values
        xpos     = list(range(len(layer_keys)))

        # Modality shading
        spans = _modality_spans(ref_gdv)
        for mod, s, e in spans:
            ax1.axvspan(s - 0.5, e + 0.5, color=MOD_COLORS.get(mod, '#f0f0f0'), alpha=0.35)

        ax1.plot(xpos, gdv_vals, '-', color='#1f77b4', linewidth=1.2, label='GDV')
        ax1.axhline(0, color='#1f77b4', linewidth=0.5, linestyle=':', alpha=0.4)
        ax1.set_ylabel('GDV', fontsize=8, color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=7)

        umap_pos = [layer_keys.index(k) for k in umap_df['LayerKey'] if k in layer_keys]
        umap_tw  = [umap_df.loc[umap_df['LayerKey'] == layer_keys[p], 'TW_best'].values[0]
                    for p in umap_pos]
        ax2.plot(umap_pos, umap_tw, '-', color='#d62728', linewidth=1.2,
                 label='UMAP TW', alpha=0.8)
        ax2.set_ylabel('UMAP Trustworthiness', fontsize=8, color='#d62728')
        ax2.tick_params(axis='y', labelcolor='#d62728', labelsize=7)

        ax1.set_xticks(xpos[::8])
        ax1.set_xticklabels([layer_keys[p].split('_')[1] for p in xpos[::8]], fontsize=6)
        ax1.set_title(SHORT[exp], fontsize=9, fontweight='bold')
        ax1.grid(axis='y', alpha=0.15)

    for ax in axes[len(EXPERIMENTS):]:
        ax.set_visible(False)

    plt.suptitle('GDV (blue) & UMAP Trustworthiness (red) per Layer\nShading: vision / language',
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, 'per_experiment_gdv_umap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Plotting GDV across all layers...")
    plot_gdv_all(OUT_DIR)
    plot_gdv_by_modality(OUT_DIR)
    print("Plotting UMAP trustworthiness...")
    plot_umap_trustworthiness(OUT_DIR)
    print("Plotting GDV vs UMAP scatter...")
    plot_gdv_vs_umap(OUT_DIR)
    print("Plotting per-experiment panels...")
    plot_per_experiment_panels(OUT_DIR)
    print(f"\nAll plots in: {OUT_DIR}")
