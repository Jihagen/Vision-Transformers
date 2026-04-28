#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""
Gender × Emotion analysis — four analysis tasks:
  1. Label distribution table (quantitative comparison)
  2. Qualitative sample inspection (same image, different conditions)
  3. GDV numerical comparison across experiments and layers
  4. Thematic cluster analysis (keyword extraction per label cluster)
  5. Emotion ↔ metric correlation analysis
"""

import os
import csv
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from collections import Counter, defaultdict
from scipy import stats

# ───────────────────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────────────────

_GE_DIR  = 'data/experiments/gender_emotion'
_MET_DIR = 'results/experiments/metrics'
OUT_ROOT = 'results/experiments/gender_emotion_analysis'

EXPERIMENTS = {
    'female_anger':       f'{_GE_DIR}/results_female_anger.npy',
    'female_fear':        f'{_GE_DIR}/results_female_fear.npy',
    'female_disgust':     f'{_GE_DIR}/results_female_disgust.npy',
    'female_sad':         f'{_GE_DIR}/results_female_sad.npy',
    'female_amusement':   f'{_GE_DIR}/results_female_amusement.npy',
    'female_awe':         f'{_GE_DIR}/results_female_awe.npy',
    'female_contentment': f'{_GE_DIR}/results_female_contentment.npy',
    'female_excitement':  f'{_GE_DIR}/results_female_excitement.npy',
    'male_anger':         f'{_GE_DIR}/results_male_anger.npy',
}

# Canonical label order (negative → positive)
LABEL_ORDER = [
    'Not Interesting',
    'Slightly Interesting',
    'Moderately Interesting',
    'Very Interesting',
    'Extremely Interesting',
]

LABEL_SCORE = {l: i for i, l in enumerate(LABEL_ORDER)}  # 0-4

LABEL_COLORS = {
    'Not Interesting':        '#d62728',
    'Slightly Interesting':   '#ff7f0e',
    'Moderately Interesting': '#1f77b4',
    'Very Interesting':       '#2ca02c',
    'Extremely Interesting':  '#9467bd',
}

# Emotion valence — used for correlation analysis
# Negative: anger, fear, disgust, sad | Positive: amusement, awe, contentment, excitement
EMOTION_VALENCE = {
    'female_anger':       -1.00,
    'female_disgust':     -0.90,
    'female_fear':        -0.75,
    'female_sad':         -0.50,
    'female_contentment':  0.50,
    'female_amusement':    0.60,
    'female_excitement':   0.80,
    'female_awe':          0.90,
    'male_anger':         -1.00,  # same emotion, different gender
}

EMOTION_AROUSAL = {
    'female_anger':        0.80,
    'female_disgust':      0.50,
    'female_fear':         0.75,
    'female_sad':         -0.50,
    'female_contentment':  0.10,
    'female_amusement':    0.60,
    'female_excitement':   0.90,
    'female_awe':          0.70,
    'male_anger':          0.80,
}

# Simple English stopwords
_STOP = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
    'it', 'its', 'they', 'them', 'their', 'there', 'which', 'who', 'what',
    'as', 'if', 'not', 'no', 'so', 'such', 'very', 'also', 'both', 'other',
    'more', 'most', 'some', 'any', 'each', 'all', 'into', 'out', 'up',
    'about', 'than', 'then', 'when', 'while', 'where', 'how', 'image',
    'shows', 'show', 'showing', 'shown', 'appears', 'appear', 'seems',
    'scene', 'photo', 'picture', 'view', 'seen', 'see', 'depict', 'depicts',
    'featuring', 'features', 'featured', 'taken', 'captured',
}


# ───────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ───────────────────────────────────────────────────────────────────────────────

def _load_results():
    """Load all experiment result lists.  Returns dict name → list[dict]."""
    data = {}
    for name, path in EXPERIMENTS.items():
        d = np.load(path, allow_pickle=True).item()
        data[name] = d['results']
    return data


def _label_key(results_list):
    r = results_list[0]
    return 'interestingness_label' if 'interestingness_label' in r else 'interestingness'


def _label_df(data: dict) -> pd.DataFrame:
    """Return DataFrame of label percentages: rows=experiments, cols=labels."""
    rows = {}
    for name, results in data.items():
        key = _label_key(results)
        counts = Counter(r[key] for r in results)
        total = len(results)
        rows[name] = {l: 100.0 * counts.get(l, 0) / total for l in LABEL_ORDER}
    return pd.DataFrame(rows, index=LABEL_ORDER).T  # shape: (exp, label)


def _mean_label_score(percentages: dict) -> float:
    return sum(LABEL_SCORE[l] * pct / 100 for l, pct in percentages.items())


def _load_gdv_csv(exp_key: str) -> pd.DataFrame:
    """Load gdv_values.csv for one experiment."""
    # exp_key may be 'female_anger' or 'gender_female_anger'
    key = exp_key if exp_key.startswith('gender_') else f'gender_{exp_key}'
    path = os.path.join(_MET_DIR, key, 'gdv_values.csv')
    return pd.read_csv(path)


def _load_gdv_pkl(exp_key: str) -> dict:
    key = exp_key if exp_key.startswith('gender_') else f'gender_{exp_key}'
    path = os.path.join(_MET_DIR, key, 'gdv.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


# ───────────────────────────────────────────────────────────────────────────────
# TASK 1 — Label distribution table + heatmap
# ───────────────────────────────────────────────────────────────────────────────

def task1_label_distribution_table(data: dict, out_dir: str):
    print("=" * 60)
    print("TASK 1  —  Label Distribution Table")
    print("=" * 60)

    df = _label_df(data)

    # Add derived columns
    df['Mean Score (0-4)'] = df.apply(lambda row: _mean_label_score(row.to_dict()), axis=1)
    df['% High (≥Very)']   = df['Very Interesting'] + df['Extremely Interesting']
    df['% Low (≤Slight)']  = df['Not Interesting'] + df['Slightly Interesting']

    # Pretty-print to terminal
    pd.set_option('display.float_format', '{:.1f}'.format)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 160)
    print(df.to_string())
    print()

    # Save CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'label_distribution_table.csv')
    df.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    # ── Heatmap (percentages, 5 label columns) ──────────────────────────────
    pct_df = df[LABEL_ORDER].copy()
    short_names = {
        'female_anger': 'F-Anger', 'female_fear': 'F-Fear',
        'female_disgust': 'F-Disgust', 'female_sad': 'F-Sad',
        'female_amusement': 'F-Amusement', 'female_awe': 'F-Awe',
        'female_contentment': 'F-Contentment', 'female_excitement': 'F-Excitement',
        'male_anger': 'M-Anger',
    }
    pct_df.index = [short_names.get(i, i) for i in pct_df.index]
    short_labels = ['Not Int.', 'Slightly', 'Moderately', 'Very', 'Extremely']
    pct_df.columns = short_labels

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        pct_df, annot=True, fmt='.1f', cmap='YlOrRd',
        linewidths=0.5, ax=ax, cbar_kws={'label': '% samples'},
        annot_kws={'size': 9},
    )
    ax.set_title('Label Distribution (%) — Gender × Emotion Conditions', fontsize=13, pad=12)
    ax.set_xlabel('Interestingness Label', fontsize=10)
    ax.set_ylabel('Condition', fontsize=10)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()
    hm_path = os.path.join(out_dir, 'label_distribution_heatmap.png')
    plt.savefig(hm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {hm_path}")

    # ── Stacked bar ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    bar_colors = [LABEL_COLORS[l] for l in LABEL_ORDER]
    bottom = np.zeros(len(pct_df))
    xs = np.arange(len(pct_df))
    for li, (lbl, col) in enumerate(zip(short_labels, bar_colors)):
        vals = pct_df.iloc[:, li].values
        ax.bar(xs, vals, bottom=bottom, color=col, label=lbl, width=0.7, edgecolor='white')
        bottom += vals
    ax.set_xticks(xs)
    ax.set_xticklabels(pct_df.index, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Share (%)')
    ax.set_ylim(0, 100)
    ax.legend(title='Label', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.set_title('Label Distribution (Stacked) — Gender × Emotion', fontsize=12, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    sb_path = os.path.join(out_dir, 'label_distribution_stacked.png')
    plt.savefig(sb_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {sb_path}")
    print()

    return df


# ───────────────────────────────────────────────────────────────────────────────
# TASK 2 — Qualitative sample inspection
# ───────────────────────────────────────────────────────────────────────────────

def _text_divergence(texts: list[str]) -> float:
    """Mean pairwise word-overlap distance (1 - Jaccard) between explanations."""
    def tokens(t):
        return set(re.findall(r'\b[a-z]+\b', t.lower())) - _STOP
    from itertools import combinations
    pairs = list(combinations(range(len(texts)), 2))
    if not pairs:
        return 0.0
    dists = []
    for i, j in pairs:
        a, b = tokens(texts[i]), tokens(texts[j])
        if not a and not b:
            dists.append(0.0)
        else:
            dists.append(1.0 - len(a & b) / len(a | b))
    return float(np.mean(dists))


def _label_variance(labels: list[str]) -> float:
    """Variance of numeric label scores across conditions."""
    scores = [LABEL_SCORE.get(l, 2) for l in labels]
    return float(np.var(scores))


def task2_qualitative_comparison(data: dict, out_dir: str, n_top: int = 10):
    print("=" * 60)
    print("TASK 2  —  Qualitative Sample Comparison")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)
    names = list(data.keys())
    n_samples = len(data[names[0]])
    lkey = {name: _label_key(data[name]) for name in names}

    # Build per-image records
    records = []
    for i in range(n_samples):
        fname = data[names[0]][i]['filename']
        texts  = {n: data[n][i]['explanation'] for n in names}
        labels = {n: data[n][i][lkey[n]] for n in names}

        # Scores restricted to female conditions (for intra-emotion variation)
        female_names = [n for n in names if n.startswith('female_')]
        female_texts  = [texts[n]  for n in female_names]
        female_labels = [labels[n] for n in female_names]

        div  = _text_divergence(female_texts)
        lvar = _label_variance(female_labels)
        records.append({
            'filename': fname,
            'idx': i,
            'text_divergence': div,
            'label_variance': lvar,
            'texts': texts,
            'labels': labels,
        })

    # Also compute female_anger vs male_anger comparison
    fa_ma_div = []
    fa_ma_lvar = []
    for rec in records:
        div_pair = _text_divergence([rec['texts']['female_anger'], rec['texts']['male_anger']])
        lvar_pair = _label_variance([rec['labels']['female_anger'], rec['labels']['male_anger']])
        fa_ma_div.append(div_pair)
        fa_ma_lvar.append(lvar_pair)

    # Sort by divergence among female conditions
    records_sorted_div  = sorted(records, key=lambda r: r['text_divergence'], reverse=True)
    records_sorted_lvar = sorted(records, key=lambda r: r['label_variance'],  reverse=True)

    # Write text report
    report_path = os.path.join(out_dir, 'qualitative_report.txt')
    with open(report_path, 'w') as f:
        _write_section = lambda s: f.write(s + '\n')

        _write_section("=" * 80)
        _write_section("QUALITATIVE SAMPLE COMPARISON — Gender × Emotion")
        _write_section("=" * 80)
        _write_section("")

        # ── Top-N by explanation divergence ────────────────────────────────
        _write_section(f"TOP {n_top} IMAGES BY EXPLANATION DIVERGENCE (across female conditions)")
        _write_section("-" * 80)
        for rec in records_sorted_div[:n_top]:
            _write_section(f"\nImage: {rec['filename']}  |  "
                          f"text_div={rec['text_divergence']:.3f}  "
                          f"label_var={rec['label_variance']:.3f}")
            for name in names:
                lbl = rec['labels'][name]
                txt = rec['texts'][name]
                _write_section(f"  [{name:<22}]  ({lbl:<25})  {txt[:120]}")

        # ── Top-N by label variance ─────────────────────────────────────────
        _write_section("")
        _write_section(f"TOP {n_top} IMAGES BY LABEL VARIANCE (different interestingness per condition)")
        _write_section("-" * 80)
        for rec in records_sorted_lvar[:n_top]:
            _write_section(f"\nImage: {rec['filename']}  |  "
                          f"label_var={rec['label_variance']:.3f}  "
                          f"text_div={rec['text_divergence']:.3f}")
            for name in names:
                lbl = rec['labels'][name]
                txt = rec['texts'][name]
                _write_section(f"  [{name:<22}]  ({lbl:<25})  {txt[:120]}")

        # ── Female anger vs Male anger ──────────────────────────────────────
        _write_section("")
        _write_section("TOP 10 IMAGES — Female-Anger vs Male-Anger (text divergence)")
        _write_section("-" * 80)
        paired = sorted(
            zip(fa_ma_div, fa_ma_lvar, records),
            key=lambda x: x[0], reverse=True
        )
        for div_v, lvar_v, rec in paired[:10]:
            _write_section(f"\nImage: {rec['filename']}  |  text_div={div_v:.3f}  label_var={lvar_v:.3f}")
            for cond in ['female_anger', 'male_anger']:
                lbl = rec['labels'][cond]
                txt = rec['texts'][cond]
                _write_section(f"  [{cond:<22}]  ({lbl:<25})  {txt[:150]}")

    print(f"  Saved: {report_path}")

    # ── Distribution of divergence scores ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist([r['text_divergence'] for r in records], bins=30, color='steelblue', edgecolor='white')
    axes[0].axvline(np.mean([r['text_divergence'] for r in records]),
                    color='red', linestyle='--', label='mean')
    axes[0].set_xlabel('Text Divergence (1 − Jaccard)')
    axes[0].set_ylabel('# Images')
    axes[0].set_title('Explanation Divergence\nacross Female Conditions')
    axes[0].legend()

    axes[1].hist([r['label_variance'] for r in records], bins=15, color='darkorange', edgecolor='white')
    axes[1].axvline(np.mean([r['label_variance'] for r in records]),
                    color='red', linestyle='--', label='mean')
    axes[1].set_xlabel('Label Score Variance')
    axes[1].set_ylabel('# Images')
    axes[1].set_title('Label Variance\nacross Female Conditions')
    axes[1].legend()

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Per-Image Divergence Distributions', fontsize=12, y=1.02)
    plt.tight_layout()
    hist_path = os.path.join(out_dir, 'divergence_distributions.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {hist_path}")
    print()

    return records


# ───────────────────────────────────────────────────────────────────────────────
# TASK 3 — GDV numerical comparison
# ───────────────────────────────────────────────────────────────────────────────

def task3_gdv_comparison(out_dir: str):
    print("=" * 60)
    print("TASK 3  —  GDV Numerical Comparison")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)

    # Load GDV per layer for all experiments
    gdv_frames = {}
    for name in EXPERIMENTS:
        df = _load_gdv_csv(name)
        gdv_frames[name] = df.set_index('LayerKey')[['GDV_Euclidean', 'GDV_Cosine', 'Modality']].copy()

    # Common layers (should be all 82)
    common_layers = sorted(
        set.intersection(*[set(df.index) for df in gdv_frames.values()]),
        key=lambda k: (
            {'vision': 0, 'projector': 1, 'language': 2}.get(k.split('_')[0], 3),
            int(re.search(r'\d+', k.split('_')[1]).group()),
        )
    )

    # Build wide GDV table
    euc_data = {name: gdv_frames[name].loc[common_layers, 'GDV_Euclidean'] for name in EXPERIMENTS}
    gdv_wide = pd.DataFrame(euc_data, index=common_layers)

    # Summary stats per experiment
    summary_rows = []
    for name in EXPERIMENTS:
        col = gdv_wide[name]
        summary_rows.append({
            'Experiment':   name,
            'Best GDV':      col.min(),
            'Best Layer':    col.idxmin(),
            'Mean GDV':     col.mean(),
            'Median GDV':   col.median(),
            'Std GDV':      col.std(),
            '# Layers > 0.05': (col > 0.05).sum(),
            '# Layers > 0.07': (col > 0.07).sum(),
        })
    summary = pd.DataFrame(summary_rows).set_index('Experiment')

    print(summary.to_string(float_format='{:.4f}'.format))
    print()

    summary.to_csv(os.path.join(out_dir, 'gdv_summary.csv'))
    gdv_wide.to_csv(os.path.join(out_dir, 'gdv_all_layers.csv'))
    print(f"  Saved: {os.path.join(out_dir, 'gdv_summary.csv')}")
    print(f"  Saved: {os.path.join(out_dir, 'gdv_all_layers.csv')}")

    # ── Heatmap: experiments × layers ──────────────────────────────────────
    short_names = {
        'female_anger': 'F-Anger', 'female_fear': 'F-Fear',
        'female_disgust': 'F-Disgust', 'female_sad': 'F-Sad',
        'female_amusement': 'F-Amusement', 'female_awe': 'F-Awe',
        'female_contentment': 'F-Contentment', 'female_excitement': 'F-Excitement',
        'male_anger': 'M-Anger',
    }
    gdv_plot = gdv_wide.T.copy()
    gdv_plot.index = [short_names.get(i, i) for i in gdv_plot.index]

    fig, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(
        gdv_plot, cmap='viridis', ax=ax, cbar_kws={'label': 'GDV (Euclidean)'},
        xticklabels=4, yticklabels=True, linewidths=0,
    )
    ax.set_title('GDV per Layer per Experiment — Gender × Emotion', fontsize=12, pad=10)
    ax.set_xlabel('Layer Key', fontsize=9)
    ax.set_ylabel('Condition', fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()
    hm_path = os.path.join(out_dir, 'gdv_heatmap.png')
    plt.savefig(hm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {hm_path}")

    # ── Line plot: GDV profile per experiment (vision layers only) ──────────
    vision_layers = [l for l in common_layers if l.startswith('vision')]
    v_data = gdv_wide.loc[vision_layers]
    v_nums = [int(re.search(r'\d+', l.split('_')[1]).group()) for l in vision_layers]

    cmap = plt.colormaps['tab10']
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, name in enumerate(EXPERIMENTS):
        style = '--' if name == 'male_anger' else '-'
        lw    = 2.2 if name in ('female_anger', 'male_anger') else 1.2
        ax.plot(v_nums, v_data[name].values, style, linewidth=lw,
                label=short_names.get(name, name), color=cmap(i), alpha=0.85)
    ax.set_xlabel('Vision Layer Number', fontsize=10)
    ax.set_ylabel('GDV (Euclidean)', fontsize=10)
    ax.set_title('GDV Profile — Vision Layers', fontsize=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    lp_path = os.path.join(out_dir, 'gdv_vision_lineplot.png')
    plt.savefig(lp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {lp_path}")

    # ── Bar chart: max GDV per experiment ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    exps = list(summary.index)
    best_gdvs = summary['Best GDV'].values
    bar_colors = ['#1f77b4' if 'female' in e else '#d62728' for e in exps]
    ax.bar(
        [short_names.get(e, e) for e in exps], best_gdvs,
        color=bar_colors, edgecolor='white', width=0.6
    )
    for i, (e, v) in enumerate(zip(exps, best_gdvs)):
        ax.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Best GDV (most negative = best clustering)')
    ax.set_title('Best GDV per Condition — Gender × Emotion', fontsize=12)
    ax.tick_params(axis='x', rotation=35, labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, 'gdv_max_barplot.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {bar_path}")
    print()

    return gdv_wide, summary


# ───────────────────────────────────────────────────────────────────────────────
# TASK 4 — Thematic cluster analysis
# ───────────────────────────────────────────────────────────────────────────────

def _top_words(texts: list[str], n: int = 12) -> list[tuple]:
    """Return top-n (word, count) pairs, bigrams included."""
    words = []
    for t in texts:
        toks = re.findall(r'\b[a-z]{3,}\b', t.lower())
        toks = [w for w in toks if w not in _STOP]
        words.extend(toks)
        # bigrams
        for a, b in zip(toks, toks[1:]):
            words.append(f'{a} {b}')
    return Counter(words).most_common(n)


def task4_thematic_clusters(data: dict, out_dir: str, gdv_threshold: float = -0.15):
    """For layers where GDV < threshold (more negative = better), extract keyword themes per label cluster."""
    print("=" * 60)
    print(f"TASK 4  —  Thematic Cluster Analysis  (GDV threshold={gdv_threshold})")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)
    lkey = {name: _label_key(data[name]) for name in data}
    names = list(data.keys())
    report_lines = []

    for name in names:
        pkl = _load_gdv_pkl(name)
        results = data[name]
        key = lkey[name]

        # Find layers below threshold (more negative = better clustering)
        high_gdv = {
            layer: gdv
            for layer, gdv in pkl['gdv_per_layer'].items()
            if gdv <= gdv_threshold
        }

        report_lines.append(f"\n{'='*70}")
        report_lines.append(f"Experiment: {name}  |  {len(high_gdv)} layers with GDV ≤ {gdv_threshold}")
        if not high_gdv:
            report_lines.append(f"  (no layers below threshold — try raising gdv_threshold toward 0)")
            continue

        # Use the single best layer (most negative GDV) for thematic analysis
        best_layer = min(high_gdv, key=high_gdv.get)
        best_gdv   = high_gdv[best_layer]
        layer_data = pkl['layer_data'][best_layer]
        sample_labels = layer_data['samples']['labels']
        sample_idxs   = list(range(len(sample_labels)))

        # Map sample positions back to results (same order)
        label_groups: dict[str, list[str]] = defaultdict(list)
        for idx, lbl in zip(sample_idxs, sample_labels):
            label_groups[lbl].append(results[idx]['explanation'])

        report_lines.append(f"  Best layer: {best_layer}  (GDV={best_gdv:.4f})")
        report_lines.append(f"  Layers below threshold: {sorted(high_gdv, key=high_gdv.get)[:5]}")
        report_lines.append("")

        for lbl in LABEL_ORDER:
            if lbl not in label_groups:
                continue
            group_texts = label_groups[lbl]
            top = _top_words(group_texts, n=10)
            keywords = ', '.join(f'{w}({c})' for w, c in top)
            report_lines.append(f"  [{lbl:<25}]  n={len(group_texts):>3}  |  {keywords}")

        # Build keyword frequency comparison across labels for this experiment
        # (small heatmap: labels × top words)
        all_words = Counter()
        for lbl in LABEL_ORDER:
            for w, c in _top_words(label_groups.get(lbl, []), n=20):
                if ' ' not in w:  # unigrams only for heatmap
                    all_words[w] += c
        top_vocab = [w for w, _ in all_words.most_common(15) if ' ' not in w]

        heat_rows = {}
        for lbl in LABEL_ORDER:
            if lbl not in label_groups:
                continue
            texts = label_groups[lbl]
            wc = Counter()
            for t in texts:
                toks = [w for w in re.findall(r'\b[a-z]{3,}\b', t.lower()) if w not in _STOP]
                wc.update(toks)
            total = max(1, sum(wc.values()))
            heat_rows[lbl[:12]] = {w: 100.0 * wc.get(w, 0) / total for w in top_vocab}

        if len(heat_rows) >= 2 and len(top_vocab) >= 3:
            heat_df = pd.DataFrame(heat_rows, index=top_vocab).T
            fig, ax = plt.subplots(figsize=(max(10, len(top_vocab) * 0.8), 3.5))
            sns.heatmap(heat_df, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                        linewidths=0.3, cbar_kws={'label': '% usage'},
                        annot_kws={'size': 7})
            ax.set_title(f'Keyword Frequency by Label Cluster\n{name} | best layer: {best_layer} (GDV={best_gdv:.4f})',
                         fontsize=10, pad=8)
            ax.set_xlabel('Top Keywords', fontsize=9)
            ax.set_ylabel('Label', fontsize=9)
            ax.tick_params(axis='x', rotation=40, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8)
            plt.tight_layout()
            fig_path = os.path.join(out_dir, f'theme_heatmap_{name}.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            report_lines.append(f"  → Keyword heatmap: {fig_path}")

    report_lines.append("")
    report_path = os.path.join(out_dir, 'thematic_clusters_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"  Saved: {report_path}")
    print('\n'.join(report_lines[:60]))  # preview
    print()


# ───────────────────────────────────────────────────────────────────────────────
# TASK 5 — Emotion ↔ metric correlation analysis
# ───────────────────────────────────────────────────────────────────────────────

def task5_emotion_correlations(data: dict, gdv_summary: pd.DataFrame, label_df: pd.DataFrame, out_dir: str):
    print("=" * 60)
    print("TASK 5  —  Emotion × Metric Correlations")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)

    # Build correlation table (female conditions only for clean emotion-valence signal)
    female_names = [n for n in EXPERIMENTS if n.startswith('female_')]

    rows = []
    for name in female_names:
        lbl_row = label_df.loc[name]
        mean_score = _mean_label_score(lbl_row[LABEL_ORDER].to_dict())
        pct_high   = lbl_row['Very Interesting'] + lbl_row['Extremely Interesting']
        pct_low    = lbl_row['Not Interesting']  + lbl_row['Slightly Interesting']
        rows.append({
            'Condition':   name,
            'Emotion':     name.replace('female_', ''),
            'Valence':     EMOTION_VALENCE[name],
            'Arousal':     EMOTION_AROUSAL[name],
            'Best GDV':     gdv_summary.loc[name, 'Best GDV'],
            'Mean GDV':    gdv_summary.loc[name, 'Mean GDV'],
            'Mean Score':  mean_score,
            '% High':      pct_high,
            '% Low':       pct_low,
            '% Moderate':  lbl_row['Moderately Interesting'],
        })

    corr_df = pd.DataFrame(rows).set_index('Condition')
    print(corr_df[['Valence', 'Arousal', 'Best GDV', 'Mean Score', '% High', '% Low']].to_string(
        float_format='{:.3f}'.format))
    print()

    # Pearson correlations between valence/arousal and metrics
    metrics = ['Best GDV', 'Mean GDV', 'Mean Score', '% High', '% Low']
    predictors = ['Valence', 'Arousal']

    print("Pearson correlations (r, p):")
    print(f"{'Metric':<15}", end='')
    for pred in predictors:
        print(f"  {pred} (r / p)", end='')
    print()
    print("-" * 55)
    for m in metrics:
        print(f"{m:<15}", end='')
        for pred in predictors:
            r, p = stats.pearsonr(corr_df[pred], corr_df[m])
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {r:+.3f} / {p:.3f}{stars}", end='')
        print()
    print()

    corr_df.to_csv(os.path.join(out_dir, 'emotion_metric_table.csv'))
    print(f"  Saved: {os.path.join(out_dir, 'emotion_metric_table.csv')}")

    # ── Female anger vs Male anger direct comparison ─────────────────────
    print("\nFemale-Anger vs Male-Anger:")
    fa = {m: corr_df.loc['female_anger', m] if 'female_anger' in corr_df.index else None for m in metrics}
    ma_rows = [n for n in EXPERIMENTS if n == 'male_anger']
    if ma_rows:
        ma_row = label_df.loc['male_anger']
        ma_mean = _mean_label_score(ma_row[LABEL_ORDER].to_dict())
        ma_high = ma_row['Very Interesting'] + ma_row['Extremely Interesting']
        ma_low  = ma_row['Not Interesting']  + ma_row['Slightly Interesting']
        ma_dict = {
            'Best GDV': gdv_summary.loc['male_anger', 'Best GDV'],
            'Mean GDV': gdv_summary.loc['male_anger', 'Mean GDV'],
            'Mean Score': ma_mean,
            '% High': ma_high,
            '% Low': ma_low,
        }
        print(f"  {'Metric':<15}  {'Female Anger':>14}  {'Male Anger':>12}  {'Δ (M-F)':>10}")
        print(f"  {'-'*55}")
        for m in metrics:
            fv = fa.get(m)
            mv = ma_dict.get(m)
            if fv is not None and mv is not None:
                print(f"  {m:<15}  {fv:>14.4f}  {mv:>12.4f}  {mv - fv:>+10.4f}")
    print()

    # ── Scatter plots: valence vs each metric ───────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    scatter_metrics = ['Best GDV', 'Mean Score', '% High', '% Low', '% Moderate', 'Mean GDV']
    emotions = corr_df['Emotion'].tolist()

    for ax, m in zip(axes, scatter_metrics):
        x = corr_df['Valence'].values
        y = corr_df[m].values

        # Regression line
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xfit = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)
        ax.plot(xfit, slope * xfit + intercept, 'k--', alpha=0.5, linewidth=1.0)

        sc = ax.scatter(x, y, c=x, cmap='RdYlGn', s=90, zorder=3,
                        vmin=-1, vmax=1, edgecolors='white', linewidths=0.5)
        for xi, yi, em in zip(x, y, emotions):
            ax.annotate(em, (xi, yi), fontsize=7, ha='left', va='bottom',
                        textcoords='offset points', xytext=(4, 2))
        ax.set_xlabel('Emotion Valence', fontsize=9)
        ax.set_ylabel(m, fontsize=9)
        ax.set_title(f'{m}\nr={r:+.3f}, p={p:.3f}', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2)

    plt.suptitle('Emotion Valence vs Metrics (Female Conditions)', fontsize=12, y=1.01)
    plt.tight_layout()
    scat_path = os.path.join(out_dir, 'valence_metric_scatter.png')
    plt.savefig(scat_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {scat_path}")

    # ── Radar / spider chart comparing all conditions on key metrics ─────────
    # Normalize metrics to 0–1 for radar
    radar_metrics = ['Best GDV', 'Mean Score', '% High']
    all_conditions = list(EXPERIMENTS.keys())

    all_rows = []
    for name in all_conditions:
        lbl_row = label_df.loc[name]
        mean_score = _mean_label_score(lbl_row[LABEL_ORDER].to_dict())
        pct_high   = lbl_row['Very Interesting'] + lbl_row['Extremely Interesting']
        all_rows.append({
            'name': name,
            'Best GDV': gdv_summary.loc[name, 'Best GDV'],
            'Mean Score': mean_score,
            '% High': pct_high,
        })
    radar_df = pd.DataFrame(all_rows).set_index('name')

    # Simple bar chart instead of radar (more readable for 9 conditions)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    short_names = {
        'female_anger': 'F-Anger', 'female_fear': 'F-Fear',
        'female_disgust': 'F-Disgust', 'female_sad': 'F-Sad',
        'female_amusement': 'F-Amusement', 'female_awe': 'F-Awe',
        'female_contentment': 'F-Contentment', 'female_excitement': 'F-Excitement',
        'male_anger': 'M-Anger',
    }
    bar_colors = ['#d62728' if 'male' in n else '#1f77b4' for n in radar_df.index]
    xticks = [short_names.get(n, n) for n in radar_df.index]

    for ax, m in zip(axes, radar_metrics):
        vals = radar_df[m].values
        ax.bar(xticks, vals, color=bar_colors, edgecolor='white', width=0.65)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.01, f'{v:.3f}', ha='center', fontsize=7)
        ax.set_title(m, fontsize=10)
        ax.tick_params(axis='x', rotation=40, labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25)

    plt.suptitle('Key Metrics per Condition (blue=female, red=male)', fontsize=11, y=1.02)
    plt.tight_layout()
    comp_path = os.path.join(out_dir, 'condition_comparison_bars.png')
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comp_path}")
    print()

    return corr_df


# ───────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading experiment data...")
    data = _load_results()
    print(f"  {len(data)} experiments loaded, {len(data['female_anger'])} samples each.\n")

    t1_dir  = os.path.join(OUT_ROOT, 'label_distributions')
    t2_dir  = os.path.join(OUT_ROOT, 'qualitative')
    t3_dir  = os.path.join(OUT_ROOT, 'gdv_comparison')
    t4_dir  = os.path.join(OUT_ROOT, 'thematic_clusters')
    t5_dir  = os.path.join(OUT_ROOT, 'correlations')

    label_df_ext = task1_label_distribution_table(data, t1_dir)
    _           = task2_qualitative_comparison(data, t2_dir, n_top=10)
    gdv_wide, gdv_summary = task3_gdv_comparison(t3_dir)
    task4_thematic_clusters(data, t4_dir, gdv_threshold=-0.15)
    task5_emotion_correlations(data, gdv_summary, label_df_ext, t5_dir)

    print("All tasks complete.  Output root:", OUT_ROOT)
