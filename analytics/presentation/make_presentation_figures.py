"""
Presentation figures, regenerated from saved CSV / NPY / image outputs only.
No model forward passes; reads results/, data/ and writes results/presentation/.

Run:  python analytics/presentation/make_presentation_figures.py
"""
from __future__ import annotations
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "presentation"

import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parent))
from palette import EMO_COLORS, GENDER_EDGES, COUNTRY_TEXT, COUNTRY_TINT, INTEREST_COLOR, WORKLOAD_COLOR, EXCITE_ANGER_COLOR
ATTACK_COLORS = {'interest': INTEREST_COLOR, 'excited_vs_angry': EXCITE_ANGER_COLOR, 'workload': WORKLOAD_COLOR}
ATTACKS = ['interest', 'excited_vs_angry', 'workload']
EPS = [0.10, 0.50, 1.00, 2.00]
TASKS = ['shopping_relevance', 'damage_severity', 'moral_evaluation']
TASK_TITLE = {'shopping_relevance': 'Shopping relevance (1-5)', 'damage_severity': 'Damage severity (0-2)',
              'moral_evaluation': 'Moral evaluation (1-5)'}


def _save(fig, name, dpi=150):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ── 01 decodability by layer (Tier I) ─────────────────────────────────────────
def fig_decodability():
    families = {
        'Gender (base personas)': ('base', 'female_'),
        'Emotion (one-vs-rest)': ('base_emotion', 'avg_'),
        'Country: Germany': ('extended_germany_country', ''),
        'Country: Nigeria': ('extended_nigeria_country', ''),
    }
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fam_color = {'Gender (base personas)': GENDER_EDGES['female'], 'Emotion (one-vs-rest)': EMO_COLORS['anger'],
                 'Country: Germany': COUNTRY_TEXT['Germany'], 'Country: Nigeria': COUNTRY_TEXT['Nigeria']}
    for name, (slug, _) in families.items():
        df = pd.read_csv(RES / "representation_discovery" / slug / "evaluation" / "vector_evaluation.csv")
        df = df[df.layer_key.str.startswith("language_")].copy()
        df["L"] = df.layer_key.str.extract(r"language_(\d+)")[0].astype(int)
        g = df.groupby("L")["projection_auc"]
        med, lo, hi = g.median(), g.min(), g.max()
        ax.plot(med.index, med.values, marker='o', ms=3.5, lw=1.8, label=name, color=fam_color[name])
        ax.fill_between(med.index, lo.values, hi.values, alpha=0.12, color=fam_color[name])
    ax.axhline(0.5, color='grey', ls=':', lw=1)
    ax.set_xlabel('language layer'); ax.set_ylabel('AUROC of mean-difference direction\n(median line, min–max band across contrasts)')
    ax.set_ylim(0.45, 1.02); ax.grid(alpha=0.25); ax.legend(fontsize=9, loc='lower left')
    ax.set_title('Concept decodability by layer — language stream only (vision layers stay at chance, AUROC ≈ 0.50)')
    fig.tight_layout()
    _save(fig, '01_decodability_by_layer.png')


# ── 05 causal dose response (Tier III) ────────────────────────────────────────
def fig_dose_response():
    runs = [
        ('Interestingness  (language_29)', RES / 'interest_validation/language_29/dose_response_blank/control_results.csv', '#1F5FBF'),
        ('Excited vs angry  (language_29)', RES / 'emotion_validation/language_29/excited_vs_angry/dose_response_blank/control_results.csv', '#DF325A'),
        ('Workload / stress  (language_24)', RES / 'workload_validation/language_24/dose_response_blank/control_results.csv', '#4A4A4A'),
        ('Country: Germany baseline  (language_23)', RES / 'country_validation/language_23/dose_response_blank/control_results.csv', '#8B4513'),
    ]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [2.2, 1]})
    for label, path, col in runs:
        df = pd.read_csv(path)
        df = df[df.score.notna()]
        g = df.groupby('alpha')['score'].agg(['mean', 'sem']).reset_index()
        axL.errorbar(g.alpha, g['mean'], yerr=g['sem'], marker='o', lw=2, capsize=3, color=col, label=label)
    axL.axvline(0, color='grey', ls='--', lw=1)
    axL.set_xlabel('injection strength α (residual stream)'); axL.set_ylabel('mean rating (1–5), ± s.e.m.')
    axL.set_title('Injected-direction dose response'); axL.grid(alpha=0.25); axL.legend(fontsize=8.5)

    gb = pd.read_csv(RES / 'blank_validation/language_29/primary/blank_validation_summary.csv')
    gb['sem'] = gb['score_std'] / np.sqrt(gb['score_count'])
    order = ['no_inject', 'inject_neg2', 'inject_pos2', 'ablate']
    gb = gb.set_index('condition').loc[order].reset_index()
    axR.bar(range(len(gb)), gb['score_mean'], yerr=gb['sem'], capsize=4, color='#9CA3AF')
    axR.set_xticks(range(len(gb))); axR.set_xticklabels(['no inject', 'α = −2', 'α = +2', 'ablate'], fontsize=9)
    axR.set_ylim(2.2, 2.8); axR.set_ylabel('mean rating (1–5)')
    axR.set_title('Gender direction (language_29)\nno measurable behavioural effect', fontsize=10.5)
    axR.grid(axis='y', alpha=0.25)
    fig.suptitle('Causal control: affective and interestingness directions move the judgement; the gender direction does not', fontsize=12)
    fig.tight_layout()
    _save(fig, '05_causal_dose_response.png')


# ── 06 UAP effect vs epsilon (Tier IV) ───────────────────────────────────────
def fig_uap_effect():
    ai = pd.read_csv(RES / 'attack_eval_projected/interestingness/summary.csv')
    ar = pd.read_csv(RES / 'attack_eval_projected/relevance/summary.csv')
    clean_p = float(ar.loc[ar.condition == 'clean', 'p_yes'].iloc[0])

    def rows(df, attack):
        out = []
        for e in EPS:
            r = df[df.condition == f'{attack}_eps{e:.2f}']
            out.append(r.iloc[0] if len(r) else None)
        return out

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for a in ['interest', 'excited_vs_angry']:
        col = ATTACK_COLORS[a]
        ri = rows(ai, a)
        axes[0].plot(EPS, [r.mean_score_delta for r in ri], marker='o', lw=2, color=col, label=a)
        axes[1].plot(EPS, [r.n_Extremely_Interesting / r.n_parsed for r in ri], marker='o', lw=2, color=col, label=a)
        rr = rows(ar, a)
        axes[2].plot(EPS, [r.p_yes - clean_p for r in rr], marker='o', lw=2, color=col, label=a)
    axes[0].set_title('Interestingness: Δ mean rating (1–5)'); axes[0].axhline(0, color='grey', ls='--', lw=1)
    axes[1].set_title('Interestingness: % images rated "Extremely Interesting"'); axes[1].set_ylim(-0.03, 1.03)
    axes[2].set_title('Binary relevance: Δ p(yes) vs clean'); axes[2].axhline(0, color='grey', ls='--', lw=1)
    for ax in axes:
        ax.set_xlabel('ε (UAP budget)'); ax.set_xticks(EPS); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    axes[1].axvspan(1.5, 2.1, color='#FEE2E2', alpha=0.6, zorder=0)
    axes[1].text(1.55, 0.5, 'ε = 2.0 collapse', fontsize=9, color='#991B1B')
    fig.suptitle('Pixel-space UAP effect vs budget — graded at ε ≤ 1.0, saturated/collapsed at ε = 2.0', fontsize=12)
    fig.tight_layout()
    _save(fig, '06_uap_effect_vs_epsilon.png')


# ── 07 UAP budget examples (Tier IV) ──────────────────────────────────────────
def _to_display(pv: np.ndarray) -> np.ndarray:
    return np.clip(pv * 0.5 + 0.5, 0, 1)


def load_at_336(img_path: str) -> np.ndarray:
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    if max(w, h) > 336:
        r = 336 / max(w, h)
        im = im.resize((int(w * r), int(h * r)), Image.Resampling.LANCZOS)
    im = im.resize((336, 336), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def fig_uap_examples():
    manifest = pd.read_pickle(ROOT / "data" / "selected_uniform_500_manifest.pkl")
    eval_mask = [i % 5 == 4 for i in range(len(manifest))]     # same held-out split as UAP eval
    img_path = manifest[eval_mask].iloc[0]["img_path"]

    pv = (load_at_336(img_path) - 0.5) / 0.5                    # HxWx3
    deltas = {}
    for e in EPS:
        d = np.load(RES / f"universal_perturbation_projected/interest/eps{e:.2f}/delta.npy")[0]  # 3x336x336
        deltas[e] = d.transpose(1, 2, 0)

    fig, axes = plt.subplots(2, 5, figsize=(17, 7.4))
    axes[0, 0].imshow(_to_display(pv)); axes[0, 0].set_title('original', fontsize=11)
    axes[1, 0].axis('off')
    for j, e in enumerate(EPS, start=1):
        att = np.clip(pv + deltas[e], -1, 1)
        axes[0, j].imshow(_to_display(att)); axes[0, j].set_title(f'interest UAP  ε = {e:.2f}', fontsize=11)
        amp = deltas[e] / (2 * e) + 0.5                        # δ rescaled to full [0,1] range
        axes[1, j].imshow(np.clip(amp, 0, 1)); axes[1, j].set_title(f'δ at ε = {e:.2f} (amplified)', fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Reconstructed inputs at each budget (model-input space, 336×336). Top: image + δ. Bottom: δ alone, rescaled for visibility.',
                 fontsize=11)
    fig.subplots_adjust(hspace=0.16, wspace=0.05, top=0.90, bottom=0.03, left=0.02, right=0.98)
    _save(fig, '07_uap_budget_examples.png', dpi=90)


# ── 08 generalisation transfer matrix (Tier IV.2) ─────────────────────────────
def fig_generalisation_transfer():
    df = pd.concat([pd.read_csv(RES / 'generalisation' / t / 'summary.csv').assign(task=t) for t in TASKS])
    cols = [(t, e) for t in TASKS for e in EPS]
    mat = np.full((len(ATTACKS), len(cols)), np.nan)
    for i, a in enumerate(ATTACKS):
        for j, (t, e) in enumerate(cols):
            r = df[(df.task == t) & (df.attack == a) & (np.isclose(df.epsilon, e))]
            if len(r):
                mat[i, j] = r.normalized_signed_shift.iloc[0]
    fig, ax = plt.subplots(figsize=(14, 3.8))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    for i in range(len(ATTACKS)):
        for j in range(len(cols)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i, j]:+.2f}', ha='center', va='center', fontsize=8.5)
    ax.set_yticks(range(len(ATTACKS))); ax.set_yticklabels(ATTACKS)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f'ε={e:.1f}' for _, e in cols], fontsize=8.5)
    for k, t in enumerate(TASKS):
        ax.text(k * len(EPS) + (len(EPS) - 1) / 2, -0.95, TASK_TITLE[t], ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(len(ATTACKS) - 0.5, -1.25)
    for k in range(1, len(TASKS)):
        ax.axvline(k * len(EPS) - 0.5, color='white', lw=3)
    fig.colorbar(im, ax=ax, label='normalized shift')
    ax.set_title('Same UAP, no retraining, three unseen tasks: normalized shift by attack × ε', pad=10)
    fig.tight_layout()
    _save(fig, '08_generalisation_transfer.png')


# ── 09 generalisation task examples ────────────────────────────────────────────
def fig_generalisation_examples():
    pick = {
        'shopping_relevance': ('workload', 2.0, lambda d: d[d.delta <= -3]),
        'damage_severity': ('workload', 2.0, lambda d: d[d.delta >= 2]),
        'moral_evaluation': ('workload', 2.0, lambda d: d[d.delta <= -3]),
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    for ax, task in zip(axes, TASKS):
        attack, eps, flt = pick[task]
        pred = pd.read_csv(RES / 'generalisation' / task / 'predictions.csv')
        man = pd.read_csv(RES / 'generalisation' / task / 'sample_manifest.csv')
        clean = pred[pred.attack == 'clean'].set_index('sample_id')
        pert = pred[(pred.attack == attack) & (np.isclose(pred.epsilon, eps))].set_index('sample_id')
        d = pert.join(clean[['model_label', 'explanation']], rsuffix='_clean').reset_index()
        d = d.dropna(subset=['delta']); cand = flt(d)
        row = (cand if len(cand) else d).sort_values('delta', key=lambda s: -s.abs()).iloc[0]
        m = man[man.sample_id == row.sample_id].iloc[0]
        ax.axis('off')
        ax_img = ax.inset_axes([0.0, 0.02, 0.36, 0.96])
        ax_img.imshow(Image.open(m.img_path).convert('RGB')); ax_img.set_xticks([]); ax_img.set_yticks([])
        if task == 'shopping_relevance':
            prompt = f'Query: "{m["query"]}"\nRate how relevant the product is (1–5).'
        elif task == 'moral_evaluation':
            prompt = 'Judge the morality of what the image portrays (1–5).'
        else:
            prompt = 'Assess the severity of visible physical damage (0–2).'
        text = (f"{TASK_TITLE[task]}   ·   attack = {attack}, ε = {eps:.1f}\n\n"
                f"Prompt: {prompt}\n\n"
                f"clean  → label {int(row.model_label_clean)}\n   \"{str(row.explanation_clean)[:420]}\"\n\n"
                f"perturbed → label {int(row.model_label)}  (Δ = {row.delta:+.0f})\n   \"{str(row.explanation)[:420]}\"")
        ax.text(0.40, 0.5, text, va='center', fontsize=10, family='DejaVu Sans', wrap=True)
        ax.set_title(f'{task}  ·  {row.sample_id}', loc='left', fontsize=11)
    fig.suptitle('One real example per generalisation task: image, prompt, clean vs perturbed output', fontsize=12)
    fig.tight_layout()
    _save(fig, '09_generalisation_examples.png')


# ── 12 label distribution: red (Not) -> dark blue (Extremely), blank as bottom row ─────
def fig_label_distribution_gradient():
    df = pd.read_csv(OUT / 'label_distribution_16.csv', index_col=0)
    labels = ['Not Interesting', 'Slightly Interesting', 'Moderately Interesting', 'Very Interesting', 'Extremely Interesting']
    blank = df.loc[['blank (no persona)']]
    rest = df.drop(index='blank (no persona)').sort_values('mean_score')
    df = pd.concat([rest, blank])
    grad = plt.cm.RdBu(np.linspace(0.02, 0.97, len(labels)))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    bottom = np.zeros(len(df))
    for lbl, col in zip(labels, grad):
        ax.bar(range(len(df)), df[lbl], bottom=bottom, color=col, edgecolor='white', lw=0.6,
               label=lbl.replace(' Interesting', ''))
        bottom += df[lbl].values
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([i.replace('_', ' ') for i in df.index], rotation=35, ha='right', fontsize=8.5)
    ax.set_ylabel('share of images (%)'); ax.set_ylim(0, 100)
    ax.legend(title='rating: red = not interesting → dark blue = extremely interesting', fontsize=8.5,
              title_fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=5, frameon=False)
    ax.set_title('Interestingness rating distribution by persona condition (sorted by mean score; blank = no persona)')
    fig.tight_layout()
    _save(fig, '12_label_distribution_gradient.png')

# ── copies of already-generated figures ───────────────────────────────────────
def copy_existing():
    pairs = [
        (RES / 'analytics' / 'fig_joint_geometry_progression.png', '02_geometry_progression.png'),
        (RES / 'analytics' / 'fig_gender_stability_schematic.png', '03_gender_stability_schematic.png'),
        (RES / 'analytics' / 'fig_additivity_showcase.png', '04_additivity_showcase.png'),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    for src, dst in pairs:
        shutil.copy(src, OUT / dst)
        print('copied', OUT / dst)


if __name__ == '__main__':
    fig_decodability()
    copy_existing()
    fig_dose_response()
    fig_uap_effect()
    fig_uap_examples()
    fig_generalisation_transfer()
    fig_generalisation_examples()
    fig_label_distribution_gradient()
