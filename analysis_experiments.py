#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from metrics import run_metrics


# ───────────────────────────────────────────────────────────────────────────────
# EXPERIMENT FILE REGISTRY
# ───────────────────────────────────────────────────────────────────────────────

_GE_DIR  = 'data/experiments/gender_emotion'
_SYN_DIR = 'data/experiments/synonym_test'

DATA_FILES = {
    # Gender × Emotion
    'gender_female_anger':       f'{_GE_DIR}/results_female_anger.npy',
    'gender_female_fear':        f'{_GE_DIR}/results_female_fear.npy',
    'gender_female_disgust':     f'{_GE_DIR}/results_female_disgust.npy',
    'gender_female_sad':         f'{_GE_DIR}/results_female_sad.npy',
    'gender_female_amusement':   f'{_GE_DIR}/results_female_amusement.npy',
    'gender_female_awe':         f'{_GE_DIR}/results_female_awe.npy',
    'gender_female_contentment': f'{_GE_DIR}/results_female_contentment.npy',
    'gender_female_excitement':  f'{_GE_DIR}/results_female_excitement.npy',
    'gender_male_anger':         f'{_GE_DIR}/results_male_anger.npy',
    # Synonym Test
    'synonym_5504':              f'{_SYN_DIR}/results_5504_synonym.npy',
    'synonym_7319':              f'{_SYN_DIR}/results_7319_synonym.npy',
}


# ───────────────────────────────────────────────────────────────────────────────
# EXPERIMENT METADATA
# ───────────────────────────────────────────────────────────────────────────────

_GE_BASE = {
    'Age range':       '30–45',
    'Employment':      'full-time employed',
    'Mental workload': 'moderate',
}


def _ge_meta(gender: str, emotion: str) -> dict:
    return {
        'title':      f'Gender × Emotion — {gender.capitalize()} / {emotion.capitalize()}',
        'experiment': 'gender_emotion',
        'persona':    {**_GE_BASE, 'Gender': gender, 'Emotion': emotion},
    }


EXPERIMENT_META = {
    'gender_female_anger':       _ge_meta('female', 'anger'),
    'gender_female_fear':        _ge_meta('female', 'fear'),
    'gender_female_disgust':     _ge_meta('female', 'disgust'),
    'gender_female_sad':         _ge_meta('female', 'sad'),
    'gender_female_amusement':   _ge_meta('female', 'amusement'),
    'gender_female_awe':         _ge_meta('female', 'awe'),
    'gender_female_contentment': _ge_meta('female', 'contentment'),
    'gender_female_excitement':  _ge_meta('female', 'excitement'),
    'gender_male_anger':         _ge_meta('male', 'anger'),
    'synonym_5504': {
        'title':      'Synonym Test — Persona 5504 (Extreme Negative)',
        'experiment': 'synonym_test',
        'persona': {
            'Age':             '25–29',
            'Gender':          'male',
            'Country':         'Hungary',
            'Continent':       'Europe',
            'Occupation':      'Law and Public Service',
            'Mental workload': 'overwhelming  (originally "extreme")',
            'Emotion':         'anger',
        },
    },
    'synonym_7319': {
        'title':      'Synonym Test — Persona 7319 (Extreme Positive)',
        'experiment': 'synonym_test',
        'persona': {
            'Age':             '15–19',
            'Gender':          'male',
            'Country':         'Greece',
            'Continent':       'Europe',
            'Occupation':      'Retail and Customer Service',
            'Mental workload': 'overwhelming  (originally "extreme")',
            'Emotion':         'excitement',
        },
    },
}


# ───────────────────────────────────────────────────────────────────────────────
# LABEL COLOURS & ORDER  (consistent across all charts)
# ───────────────────────────────────────────────────────────────────────────────

LABEL_ORDER = [
    'Not Interesting',
    'Slightly Interesting',
    'Moderately Interesting',
    'Very Interesting',
    'Extremely Interesting',
]

LABEL_COLORS = {
    'Not Interesting':        '#d62728',
    'Slightly Interesting':   '#ff7f0e',
    'Moderately Interesting': '#1f77b4',
    'Very Interesting':       '#2ca02c',
    'Extremely Interesting':  '#9467bd',
}


# ───────────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────────

def _detect_label_key(results_list: list) -> str:
    for r in results_list:
        if 'interestingness_label' in r:
            return 'interestingness_label'
        if 'interestingness' in r:
            return 'interestingness'
    raise KeyError("No interestingness label key found in results.")


def _format_persona(persona: dict) -> str:
    """Format persona dict as two balanced lines of key: value pairs."""
    items = [f'{k}: {v}' for k, v in persona.items()]
    mid = (len(items) + 1) // 2
    line1 = '  ·  '.join(items[:mid])
    line2 = '  ·  '.join(items[mid:])
    return f'{line1}\n{line2}' if line2 else line1


# ───────────────────────────────────────────────────────────────────────────────
# TASK #1 — RESULT COUNTS
# New experiments always run all 500 images, so this replaces the
# missing-entry report from main.py.
# ───────────────────────────────────────────────────────────────────────────────

def report_result_counts():
    print("=" * 60)
    print("TASK #1  —  Result Counts")
    print("=" * 60)
    print()
    for name, path in DATA_FILES.items():
        d = np.load(path, allow_pickle=True).item()
        n = len(d['results'])
        meta = EXPERIMENT_META[name]
        status = "OK" if n == 500 else f"INCOMPLETE ({n}/500)"
        print(f"  {name:35s}  {n:3d} / 500   [{status}]")
    print()


# ───────────────────────────────────────────────────────────────────────────────
# TASK #2 — LABEL DISTRIBUTION: PRINT + BAR CHARTS
# ───────────────────────────────────────────────────────────────────────────────

def print_label_distribution(name: str, results_list: list):
    label_key = _detect_label_key(results_list)
    labels = [r[label_key] for r in results_list]
    counts = Counter(labels)
    total = len(labels)
    meta = EXPERIMENT_META[name]

    print(f"  {meta['title']}  (n={total})")
    for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = '█' * int(round(30 * cnt / total))
        print(f"    {lbl:30s}  {cnt:4d}  ({100 * cnt / total:5.1f}%)  {bar}")
    print()


def save_bar_chart(name: str, results_list: list, out_dir: str):
    """Bar chart of interestingness label distribution.

    All canonical labels are always shown.  Labels absent from the data are
    rendered with a dimmed x-tick.  Persona fields are shown in an annotation
    box at the bottom instead of a hypothesis string.
    """
    meta = EXPERIMENT_META[name]
    label_key = _detect_label_key(results_list)
    raw = [r[label_key] for r in results_list]
    counts = Counter(raw)
    total = len(raw)

    unknown = sorted(l for l in counts if l not in LABEL_ORDER)
    all_labels = LABEL_ORDER + unknown

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x_pos = np.arange(len(all_labels))
    BAR_W = 0.58

    for i, lbl in enumerate(all_labels):
        cnt = counts.get(lbl, 0)
        pct = 100.0 * cnt / total
        if cnt > 0:
            ax.bar(i, pct, width=BAR_W,
                   color=LABEL_COLORS.get(lbl, '#aec7e8'),
                   edgecolor='white', linewidth=1.0, zorder=3)
            ax.text(i, pct + 1.2,
                    f"{pct:.1f}%\n({cnt} samples)",
                    ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold', color='#222222', zorder=4)

    def _wrap(s):
        parts = s.split()
        mid = len(parts) // 2
        return ' '.join(parts[:mid]) + '\n' + ' '.join(parts[mid:])

    ax.set_xticks(x_pos)
    ax.set_xticklabels([_wrap(l) for l in all_labels], fontsize=9)
    for tick, lbl in zip(ax.get_xticklabels(), all_labels):
        tick.set_color('#aaaaaa' if counts.get(lbl, 0) == 0 else '#222222')

    ax.set_ylim(0, 100)
    ax.set_ylabel('Share of samples (%)', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(
        f"Interestingness Distribution — {meta['title']}  (n={total})",
        fontsize=12, fontweight='bold', pad=14,
    )

    persona_text = _format_persona(meta['persona'])
    fig.text(0.5, 0.01, persona_text,
             ha='center', va='bottom', fontsize=8.5,
             transform=fig.transFigure,
             bbox=dict(boxstyle='round,pad=0.5',
                       facecolor='#f5f5f5', edgecolor='#bbbbbb'))
    plt.tight_layout(rect=[0, 0.12, 1, 1.0])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'label_dist_{name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def report_label_distributions(out_dir: str):
    print("=" * 60)
    print("TASK #2  —  Interestingness Label Distributions")
    print("=" * 60)
    print()
    for name, path in DATA_FILES.items():
        d = np.load(path, allow_pickle=True).item()
        print_label_distribution(name, d['results'])
        save_bar_chart(name, d['results'], out_dir)
    print()


# ───────────────────────────────────────────────────────────────────────────────
# TASK #3 — RUN METRICS ON ALL EXPERIMENT FILES
# ───────────────────────────────────────────────────────────────────────────────

def run_all_metrics(output_root: str, min_frac: float = 0.0):
    """Run GDV / UMAP / t-SNE metrics for every experiment results file.

    min_frac controls the minimum relative class frequency passed to
    filter_results_by_min_support inside run_metrics.  0.0 keeps all classes.
    """
    print("=" * 60)
    print("TASK #3  —  Computing Metrics (GDV + UMAP + t-SNE)")
    print(f"           min_frac={min_frac}  (0.0 = all classes kept)")
    print("=" * 60)
    print()

    for name, path in DATA_FILES.items():
        out = os.path.join(output_root, name)
        meta = EXPERIMENT_META[name]
        print(f"--- {meta['title']}  [{name}]")
        print(f"    data : {path}")
        print(f"    out  : {out}")

        gdv_results = run_metrics(
            data_path=path,
            vision_layer_threshold=33,
            output_root=out,
            min_frac=min_frac,
        )

        print(f"    layers analysed : {len(gdv_results)}")
        if gdv_results:
            best = max(gdv_results, key=gdv_results.get)
            print(f"    highest GDV     : {best}  (GDV = {gdv_results[best]:.4f})")
        print()


# ───────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DIST_DIR     = 'results/experiments/label_distributions'
    METRICS_ROOT = 'results/experiments/metrics'

    report_result_counts()
    report_label_distributions(out_dir=DIST_DIR)
    run_all_metrics(output_root=METRICS_ROOT, min_frac=0.0)
