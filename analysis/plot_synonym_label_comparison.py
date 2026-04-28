#!/usr/bin/env python3
"""Stacked bar chart: label distributions for extreme_pos/synonym_7319 and extreme_neg/synonym_5504."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if os.path.basename(os.getcwd()) == 'analysis':
    os.chdir('..')

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LABEL_ORDER = [
    'Not Interesting', 'Slightly Interesting', 'Moderately Interesting',
    'Very Interesting', 'Extremely Interesting',
]
LABEL_COLORS = {
    'Not Interesting':        '#d62728',
    'Slightly Interesting':   '#ff7f0e',
    'Moderately Interesting': '#1f77b4',
    'Very Interesting':       '#2ca02c',
    'Extremely Interesting':  '#9467bd',
}
LABEL_SHORT = {
    'Not Interesting': 'Not', 'Slightly Interesting': 'Slightly',
    'Moderately Interesting': 'Moderately', 'Very Interesting': 'Very',
    'Extremely Interesting': 'Extremely',
}

BARS = [
    ('Extreme Positive\n(original)',  'data/results_persona_7319.npy'),
    ('Synonym 7319',                  'data/experiments/synonym_test/results_7319_synonym.npy'),
    ('Extreme Negative\n(original)',  'data/results_persona_5504.npy'),
    ('Synonym 5504',                  'data/experiments/synonym_test/results_5504_synonym.npy'),
]

# Pair groups — amber (positive) and slate-blue (negative).
# Teal/magenta/sienna are reserved for the experiment-set groupings in the summary.
GROUPS = [
    (0, 1, '#F9A825', '#E65100'),   # amber / dark-amber  — positive pair
    (2, 3, '#78909C', '#37474F'),   # slate-blue / charcoal — negative pair
]


def get_label_pcts(path):
    d    = np.load(path, allow_pickle=True).item()
    res  = d['results']
    lkey = 'interestingness_label' if 'interestingness_label' in res[0] else 'interestingness'
    cnt  = Counter(r[lkey] for r in res)
    tot  = len(res)
    return {l: 100.0 * cnt.get(l, 0) / tot for l in LABEL_ORDER}


pcts = [(label, get_label_pcts(path)) for label, path in BARS]

fig, ax = plt.subplots(figsize=(9, 5.5))
# Leave headroom above the axes for pair labels + title without crowding
fig.subplots_adjust(top=0.78)
xs = np.arange(len(pcts))

# Background shading per pair
for lo, hi, shade_col, _ in GROUPS:
    ax.axvspan(lo - 0.45, hi + 0.45, color=shade_col, alpha=0.10, zorder=0)

# Stacked bars
bottom = np.zeros(len(pcts))
for lbl in LABEL_ORDER:
    vals = np.array([p[lbl] for _, p in pcts])
    ax.bar(xs, vals, bottom=bottom, color=LABEL_COLORS[lbl],
           label=LABEL_SHORT[lbl], width=0.62, edgecolor='white', linewidth=0.5, zorder=2)
    bottom += vals

# Pair separator
ax.axvline(1.5, color='#aaaaaa', linewidth=1.0, linestyle='--', zorder=1)

# Pair labels sit just above the axes; title is in figure space above them
pair_labels = ['Extreme Positive pair', 'Extreme Negative pair']
pair_centers = [0.5, 2.5]
for (lo, hi, _, text_col), label, cx in zip(GROUPS, pair_labels, pair_centers):
    ax.text(cx, 1.03, label, ha='center', va='bottom',
            fontsize=9, fontweight='bold', color=text_col,
            transform=ax.get_xaxis_transform())

ax.set_xticks(xs)
ax.set_xticklabels([label for label, _ in pcts], fontsize=9)
ax.set_ylim(0, 100)
ax.set_ylabel('% of samples', fontsize=10)
# Title placed in figure coords so it sits above the pair labels
fig.text(0.5, 0.95, 'Interestingness Label Distribution — Original vs Synonym Personas',
         ha='center', va='top', fontsize=11)
ax.grid(axis='y', alpha=0.2, linestyle='--', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles = [mpatches.Patch(color=LABEL_COLORS[l], label=LABEL_SHORT[l]) for l in LABEL_ORDER]
ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left',
          fontsize=9, title='Interestingness')

out = 'results/summary_synonym_label_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.show()
