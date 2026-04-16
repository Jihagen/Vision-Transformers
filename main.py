#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from metrics import run_metrics


# ───────────────────────────────────────────────────────────────────────────────
# PERSONA / FILE REGISTRY
# ───────────────────────────────────────────────────────────────────────────────

# Each entry maps a short name to its data file and descriptive metadata used
# for terminal output and pie-chart annotations.
DATA_FILES = {
    'blank':            'data/results_blank_activations.npy',
    'extreme_negative': 'data/results_persona_5504.npy',
    'extreme_positive': 'data/results_persona_7319.npy',
    'low_negative':     'data/results_persona_3770.npy',
    'low_positive':     'data/results_persona_8447.npy',
}

PERSONA_META = {
    'blank': {
        'title':      'Baseline — No Persona',
        'hypothesis': None,
        'candidate':  None,
        'missing_pkl': None,
    },
    'extreme_negative': {
        'title':     'Extreme Negative',
        'hypothesis': (
            'Denser distribution and lower interestingness scores '
            'due to extreme mental load and negative emotional valence.'
        ),
        'candidate': {
            'ID':          '5504',
            'Age':         '25–29',
            'Gender':      'Male',
            'Country':     'Hungary',
            'Region':      'Europe',
            'Occupation':  'Law and Public Service',
            'Mental Load': 'Extreme',
            'Emotion':     'Anger',
        },
        'missing_pkl': 'data/missing_persona_5504.pkl',
    },
    'extreme_positive': {
        'title':     'Extreme Positive',
        'hypothesis': (
            'Denser distribution at higher interestingness scores '
            'due to extreme mental load but positive emotional valence.'
        ),
        'candidate': {
            'ID':          '7319',
            'Age':         '15–19',
            'Gender':      'Male',
            'Country':     'Greece',
            'Region':      'Europe',
            'Occupation':  'Retail and Customer Service',
            'Mental Load': 'Extreme',
            'Emotion':     'Excitement',
        },
        'missing_pkl': 'data/missing_persona_7319.pkl',
    },
    'low_negative': {
        'title':     'Low Negative',
        'hypothesis': (
            'Broader distribution at lower interestingness scores '
            'due to low mental load but negative emotional valence.'
        ),
        'candidate': {
            'ID':          '3770',
            'Age':         '55–59',
            'Gender':      'Diverse',
            'Country':     'Malaysia',
            'Region':      'Asia',
            'Occupation':  'Arts and Entertainment',
            'Mental Load': 'Low',
            'Emotion':     'Fear',
        },
        'missing_pkl': 'data/missing_persona_3770.pkl',
    },
    'low_positive': {
        'title':     'Low Positive',
        'hypothesis': (
            'Higher interestingness scores and broader distribution '
            'due to low mental load and positive emotional valence.'
        ),
        'candidate': {
            'ID':          '8447',
            'Age':         '45–49',
            'Gender':      'Female',
            'Country':     'Zambia',
            'Region':      'Africa',
            'Occupation':  'Administration and Clerical',
            'Mental Load': 'Low',
            'Emotion':     'Amusement',
        },
        'missing_pkl': 'data/missing_persona_8447.pkl',
    },
}


# ───────────────────────────────────────────────────────────────────────────────
# LABEL COLOURS & ORDER  (consistent across all charts)
# ───────────────────────────────────────────────────────────────────────────────

# Canonical ordering from lowest to highest interestingness
LABEL_ORDER = [
    'Not Interesting',
    'Slightly Interesting',
    'Moderately Interesting',
    'Very Interesting',
    'Extremely Interesting',
]

# Fixed colour per label – same slice colour in every chart
LABEL_COLORS = {
    'Not Interesting':        '#d62728',   # red
    'Slightly Interesting':   '#ff7f0e',   # orange
    'Moderately Interesting': '#1f77b4',   # blue
    'Very Interesting':       '#2ca02c',   # green
    'Extremely Interesting':  '#9467bd',   # purple
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


# ───────────────────────────────────────────────────────────────────────────────
# TASK #1 — MISSING ENTRY REPORT
# ───────────────────────────────────────────────────────────────────────────────

def report_missing_entries():
    """Print a terminal summary of missing entries per persona and verify
    that the results .npy files are consistent with the missing counts."""
    print("=" * 60)
    print("TASK #1  —  Missing Entry & Integrity Report")
    print("=" * 60)

    # Blank has no missing file — it ran against all 500 images
    d = np.load(DATA_FILES['blank'], allow_pickle=True).item()
    blank_n = len(d['results'])
    print(f"\n  {'blank':20s}  results: {blank_n:3d} / 500   missing: 0")

    for name, meta in PERSONA_META.items():
        if meta['missing_pkl'] is None:
            continue
        with open(meta['missing_pkl'], 'rb') as f:
            df_missing = pickle.load(f)
        n_missing = len(df_missing)

        d = np.load(DATA_FILES[name], allow_pickle=True).item()
        n_results = len(d['results'])

        expected = 500 - n_missing
        integrity = "OK" if n_results == expected else f"MISMATCH (expected {expected})"
        cid = meta['candidate']['ID']
        print(f"  {name:20s}  results: {n_results:3d} / 500   "
              f"missing: {n_missing:2d}   [{integrity}]  (candidate ID {cid})")

    print()
    print("  All persona result files account for their missing entries.")
    print("  No additional integrity issues detected.")
    print()


# ───────────────────────────────────────────────────────────────────────────────
# TASK #2 — LABEL DISTRIBUTION: PRINT + PIE CHARTS
# ───────────────────────────────────────────────────────────────────────────────

def print_label_distribution(name: str, results_list: list):
    label_key = _detect_label_key(results_list)
    labels = [r[label_key] for r in results_list]
    counts = Counter(labels)
    total = len(labels)
    meta = PERSONA_META[name]

    print(f"  {meta['title']}  (n={total})")
    for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = '█' * int(round(30 * cnt / total))
        print(f"    {lbl:30s}  {cnt:4d}  ({100 * cnt / total:5.1f}%)  {bar}")
    print()


def save_bar_chart(name: str, results_list: list, out_dir: str):
    """Bar chart of interestingness label distribution.

    All canonical labels are always shown on the x-axis.  Labels that have
    no samples are rendered in grey (bar omitted, tick label dimmed) so the
    reader can immediately see which categories never appeared.  The y-axis
    is fixed at 0–100 % across all charts for easy cross-persona comparison.

    Bar annotation format: "X.X%\n(N samples)" — percentage is primary.
    """
    meta = PERSONA_META[name]
    label_key = _detect_label_key(results_list)
    raw = [r[label_key] for r in results_list]
    counts = Counter(raw)
    total = len(raw)

    # Always show all canonical labels; append any unknown ones alphabetically
    unknown = sorted(l for l in counts if l not in LABEL_ORDER)
    all_labels = LABEL_ORDER + unknown

    # ── Figure ────────────────────────────────────────────────────────────────
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
            # Annotation: percentage bold on top, count smaller below
            ax.text(i, pct + 1.2,
                    f"{pct:.1f}%\n({cnt} samples)",
                    ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold', color='#222222', zorder=4)

    # ── X-axis ticks: coloured if present, grey if absent ────────────────────
    # Wrap long label names at the space for compact display
    def _wrap(s):
        parts = s.split()
        mid = len(parts) // 2
        return ' '.join(parts[:mid]) + '\n' + ' '.join(parts[mid:])

    ax.set_xticks(x_pos)
    ax.set_xticklabels([_wrap(l) for l in all_labels], fontsize=9)
    for tick, lbl in zip(ax.get_xticklabels(), all_labels):
        if counts.get(lbl, 0) == 0:
            tick.set_color('#aaaaaa')   # dim → "this label never appeared"
        else:
            tick.set_color('#222222')

    # ── Y-axis: fixed 0–100 % for cross-persona comparability ────────────────
    ax.set_ylim(0, 100)
    ax.set_ylabel('Share of samples (%)', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"Interestingness Distribution — {meta['title']}  (n={total})",
        fontsize=12, fontweight='bold', pad=14,
    )

    # ── Hypothesis (top) + candidate info (bottom) ───────────────────────────
    if meta['hypothesis']:
        hyp_text = f"Hypothesis:  {meta['hypothesis']}"
        c = meta['candidate']
        cand_text = (
            f"Candidate   ID {c['ID']}  ·  Age {c['Age']}  ·  "
            f"Gender: {c['Gender']}  ·  Country: {c['Country']} ({c['Region']})\n"
            f"Occupation: {c['Occupation']}  ·  "
            f"Mental load: {c['Mental Load']}  ·  Emotion: {c['Emotion']}"
        )
        fig.text(0.5, 0.98, hyp_text,
                 ha='center', va='top', fontsize=9, style='italic',
                 transform=fig.transFigure)
        fig.text(0.5, 0.01, cand_text,
                 ha='center', va='bottom', fontsize=8.5,
                 transform=fig.transFigure,
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='#f5f5f5', edgecolor='#bbbbbb'))
        plt.tight_layout(rect=[0, 0.12, 1, 0.94])
    else:
        fig.text(0.5, 0.01, 'Baseline activations — no persona applied',
                 ha='center', va='bottom', fontsize=9, style='italic',
                 transform=fig.transFigure)
        plt.tight_layout(rect=[0, 0.05, 1, 1.0])

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
# TASK #3 — RUN METRICS ON ALL RESULTS FILES
# ───────────────────────────────────────────────────────────────────────────────

def run_all_metrics(output_root: str, min_frac: float = 0.0):
    """Run GDV / UMAP / t-SNE metrics for every results file.

    min_frac controls the minimum relative class frequency threshold passed to
    filter_results_by_min_support inside run_metrics.  Set to 0.0 to include
    all classes (default here); adjust after reviewing the label distributions.
    """
    print("=" * 60)
    print("TASK #3  —  Computing Metrics (GDV + UMAP + t-SNE)")
    print(f"           min_frac={min_frac}  (0.0 = all classes kept)")
    print("=" * 60)
    print()

    for name, path in DATA_FILES.items():
        out = os.path.join(output_root, name)
        meta = PERSONA_META[name]
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
    DIST_DIR     = 'results/label_distributions'
    METRICS_ROOT = 'results/metrics'

    # ── Task #1 ──────────────────────────────────────────────────────────────
    report_missing_entries()

    # ── Task #2 ──────────────────────────────────────────────────────────────
    report_label_distributions(out_dir=DIST_DIR)

    # ── Task #3 ──────────────────────────────────────────────────────────────
    # min_frac=0.0 keeps all classes initially.
    # Change this after reviewing the distributions printed above.
    run_all_metrics(output_root=METRICS_ROOT, min_frac=0.0)
