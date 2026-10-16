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
        ('Country: Nigeria  (language_23)', RES / 'country_validation/language_23/dose_response_blank/control_results.csv', '#8B4513'),
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

# ── 13 direction alignment vs causal slope (Tier II × Tier III) ───────────────
def fig_alignment_vs_slope():
    """x: cosine distance (1 − cosine) of each injected unit direction from the
    interestingness direction at its own injection layer; y: rating change per unit α, (mean(α=+2) − mean(α=−2)) / 4,
    paired over images. ±2 is the only strength every direction was run at."""
    D = RES / 'representation_discovery'
    L = lambda n: f'language_{n}_D5120'
    unit = lambda v: v / np.linalg.norm(v)
    one = lambda path, layer: unit(np.load(path, allow_pickle=True).item()[layer].astype(np.float64))
    avg = lambda paths, layer: unit(np.mean([np.load(f, allow_pickle=True).item()[layer] for f in sorted(paths)],
                                            axis=0).astype(np.float64))
    interest = lambda layer: one(D / 'interestingness/md_vectors/blank_interest_high_vs_blank_interest_low.npy', layer)
    workload = D / 'mental_workload/md_vectors/workload_overwhelming_vs_workload_minimal.npy'
    emo_dir = D / 'base_emotion/md_vectors'
    GREY = '#9CA3AF'

    # (label, layer, vector, results csv, condition column, +2 key, −2 key, colour)
    sweep = lambda p: (RES / p / 'dose_response_blank/control_results.csv', 'alpha', 2.0, -2.0)
    single = lambda p: (RES / p / 'blank_results.csv', 'condition', 'inject_pos2', 'inject_neg2')
    runs = [
        ('interestingness', 29, interest(L(29)), *sweep('interest_validation/language_29'), INTEREST_COLOR),
        ('excited vs angry', 29, one(emo_dir / 'avg_excitement_vs_avg_anger.npy', L(29)),
         *sweep('emotion_validation/language_29/excited_vs_angry'), EXCITE_ANGER_COLOR),
        ('workload', 29, one(workload, L(29)), *sweep('workload_validation/language_29'), WORKLOAD_COLOR),
        ('workload', 24, one(workload, L(24)), *sweep('workload_validation/language_24'), WORKLOAD_COLOR),
        ('country (Nigeria)', 23, avg((D / 'extended_nigeria_country/md_vectors').glob('*.npy'), L(23)),
         *sweep('country_validation/language_23'), '#8B4513'),
        ('gender', 29, avg((D / 'base/md_vectors').glob('female_*_vs_male_*.npy'), L(29)),
         *single('blank_validation/language_29/primary'), GREY),
    ] + [
        (e, 29, one(emo_dir / f'avg_{e}_vs_avg_rest_{e}.npy', L(29)),
         *single(f'emotion_validation/language_29/{e}'), EMO_COLORS[e])
        for e in ['awe', 'excitement', 'amusement', 'contentment', 'anger', 'sad', 'fear', 'disgust']
    ]

    # language_29 country injections are added once their sweep has been written
    for country in ['Nigeria', 'Germany']:
        path = f'country_validation/language_29/{country.lower()}'
        if (RES / path / 'dose_response_blank/control_results.csv').exists():
            runs.append((f'country ({country})', 29,
                         avg((D / f'extended_{country.lower()}_country/md_vectors').glob('*.npy'), L(29)),
                         *sweep(path), COUNTRY_TEXT[country]))

    rows = []
    for label, layer, vec, path, col, pos, neg, colour in runs:
        df = pd.read_csv(path)
        p = df[df.score.notna()].pivot_table(index='filename', columns=col, values='score')
        d = ((p[pos] - p[neg]) / 4).dropna()
        iv = interest(L(layer))
        rows.append(dict(direction=label, layer=layer, cosine=float(vec @ iv), pearson_r=float(np.corrcoef(vec, iv)[0, 1]),
                         slope=d.mean(), slope_sem=d.sem(), n_images=len(d), colour=colour))
    t = pd.DataFrame(rows)
    t.insert(3, 'cosine_distance', 1 - t.cosine)
    t.drop(columns='colour').to_csv(OUT / 'alignment_vs_slope.csv', index=False)

    ref = t[(t.direction == 'interestingness')].slope.iloc[0]
    same = t[(t.layer == 29) & (t.direction != 'interestingness')]
    r = np.corrcoef(same.cosine_distance, same.slope)[0, 1]

    fig, ax = plt.subplots(figsize=(9.2, 6))
    xs = np.array([-0.08, 1.6])
    ax.plot(xs, ref * (1 - xs), color='grey', ls='--', lw=1, zorder=1,
            label='projection only: slope = (1 − distance) × interestingness slope (not a fit)')
    ax.axhline(0, color='grey', lw=0.6); ax.axvline(1, color='grey', lw=0.6)
    markers = {29: 'o', 24: 's', 23: '^'}
    for _, q in t.iterrows():
        ax.errorbar(q.cosine_distance, q.slope, yerr=q.slope_sem, fmt=markers[q.layer], ms=8, color=q.colour,
                    mec='white', mew=1, capsize=3, lw=1.2, zorder=3)
    offsets = {'interestingness': (-10, -4, 'right'), 'awe': (9, -9, 'left'), 'excited vs angry': (-8, 7, 'right'),
               'excitement': (9, -9, 'left'), 'workload@24': (9, 3, 'left'), 'workload@29': (9, -12, 'left'),
               'country (Nigeria)@23': (9, -11, 'left'), 'gender': (0, 22, 'center'), 'amusement': (9, -3, 'left'),
               'contentment': (-9, 6, 'right'), 'anger': (-9, -3, 'right'), 'disgust': (9, -9, 'left'),
               'sad': (-9, 4, 'right'), 'fear': (0, 16, 'center')}
    for _, q in t.iterrows():
        dx, dy, ha = offsets.get(f'{q.direction}@{q.layer}', offsets.get(q.direction, (9, -3, 'left')))
        dx, ha = -dx, {'left': 'right', 'right': 'left', 'center': 'center'}[ha]  # offsets were set on the cosine axis
        name = q.direction + (f'  (L{q.layer})' if q.layer != 29 or (t.direction == q.direction).sum() > 1 else '')
        ax.annotate(name, (q.cosine_distance, q.slope), xytext=(dx, dy), textcoords='offset points', ha=ha, fontsize=8.5, color='#333333')
    for layer, m in markers.items():
        ax.plot([], [], m, color='#555555', mec='white', ms=8, label=f'injected at language_{layer}')
    ax.set_xlim(-0.08, 1.6)
    ax.set_xlabel('cosine distance from the interestingness direction (1 − cosine), at the injection layer\n'
                  '0 = same direction, 1 = orthogonal, above 1 = opposed')
    ax.set_ylabel('rating change per unit α   (mean at α=+2 − mean at α=−2) / 4,  ± s.e.m.')
    ax.set_title('The further a direction is from the interestingness direction, the less it raises the rating when injected\n'
                 f'language_29, {len(same)} directions other than interestingness: Pearson r = {r:.2f}', fontsize=11)
    ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc='upper right')
    fig.tight_layout()
    _save(fig, '13_alignment_vs_causal_slope.png')
    print(t.drop(columns='colour').round(4).to_string(index=False))
    print(f'language_29, excluding interestingness: Pearson r(cosine, slope) = {r:.3f}, n = {len(same)}')


# ── 14 cross-model transfer of the workload UAP (Tier IV) ─────────────────────
def fig_cross_model_transfer():
    """Workload UAP (trained on Llama 4 Scout) shown to two other models, against
    magnitude-matched controls. One panel per task × model; y is the paired label
    shift as a share of the rating scale, mean ± s.e.m. over images."""
    from scipy.stats import wilcoxon
    XM = RES / 'generalisation_cross_model'
    width = {'damage_severity': 2.0, 'moral_evaluation': 4.0, 'shopping_relevance': 4.0}
    ylab = {'damage_severity': 'Damage severity\n↑ rated as more damaged',
            'moral_evaluation': 'Moral evaluation\n↓ rated as more immoral',
            'shopping_relevance': 'Shopping relevance\n↓ rated as less relevant'}
    models = [('llama', 'Llama 4 Scout\nperturbation was trained on this model'),
              ('qwen2.5-vl-7b', 'Qwen2.5-VL-7B\nnever saw the perturbation in training'),
              ('llava-1.5-7b', 'LLaVA-1.5-7B\nnever saw the perturbation in training')]
    CONTROL, NOISE = '#7A808A', '#A9AEB6'
    TINT = {'ref': 'white', 'same': '#E6F4EC', 'opposite': '#FCEBDD', 'none': '#F1F1F1'}

    def shifts(task, model):
        """{series: DataFrame(index=epsilon, mean, sem)}, paired per-image deltas, p vs shuffled"""
        if model == 'llama':
            pred = pd.concat([pd.read_csv(RES / 'generalisation' / task / 'predictions.csv'),
                              pd.read_csv(RES / 'generalisation_noise' / task / 'predictions.csv')])
        else:
            pred = pd.read_csv(XM / model / task / 'predictions.csv')
        pred = pred[pred.attack != 'clean'].dropna(subset=['delta']).drop_duplicates(['sample_id', 'attack', 'epsilon'])
        pred['attack'] = pred.attack.str.replace(r'noise_seed\d', 'noise', regex=True)
        wide = pred.pivot_table(index='sample_id', columns=['attack', 'epsilon'], values='delta') / width[task]
        out = {a: pd.DataFrame({'mean': wide[a].mean(), 'sem': wide[a].sem()})
               for a in ['workload', 'shuffled_workload', 'noise'] if a in wide.columns.get_level_values(0)}
        ref = 'shuffled_workload' if 'shuffled_workload' in out else 'noise'
        pvals = {}
        for eps in out[ref].index:
            d = (wide['workload'][eps] - wide[ref][eps]).dropna()
            d = d[d != 0]
            pvals[eps] = wilcoxon(d).pvalue if len(d) > 5 else 1.0
        return out, pvals, pd.read_csv((RES / 'generalisation' if model == 'llama' else XM / model) / task / 'summary.csv')

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10), sharex=True, sharey='row')
    xpos = {e: i for i, e in enumerate(EPS)}
    for r, task in enumerate(['damage_severity', 'moral_evaluation', 'shopping_relevance']):
        for c, (model, title) in enumerate(models):
            ax = axes[r, c]
            out, pvals, summary = shifts(task, model)
            ax.axhline(0, color='grey', lw=0.8)
            for name, colour, ls, marker, lab in [('noise', NOISE, ':', 's', 'random noise, same strength'),
                                                  ('shuffled_workload', CONTROL, '--', 'D', 'same perturbation, pixels shuffled'),
                                                  ('workload', WORKLOAD_COLOR, '-', 'o', 'workload / stress perturbation')]:
                if name not in out:
                    continue
                g = out[name]
                ax.errorbar([xpos[e] for e in g.index], g['mean'], yerr=g['sem'], color=colour, ls=ls, marker=marker,
                            ms=7 if name == 'workload' else 5, lw=2.4 if name == 'workload' else 1.6, capsize=3,
                            mec='white', mew=0.8, label=lab, zorder=3 if name == 'workload' else 2)
            ref = out.get('shuffled_workload', out['noise'])
            common = [e for e in ref.index if e in out['workload'].index]
            ax.fill_between([xpos[e] for e in common], out['workload'].loc[common, 'mean'], ref.loc[common, 'mean'],
                            color=WORKLOAD_COLOR, alpha=0.16, lw=0, zorder=1)
            for e in common:
                if pvals[e] < 0.05:
                    y = out['workload'].loc[e, 'mean']
                    ax.annotate('★', (xpos[e], y), xytext=(0, 9 if y >= ref.loc[e, 'mean'] else -17),
                                textcoords='offset points', ha='center', fontsize=11, color='#333333')
            # one-line reading of the panel, as its title; the panel tint repeats it
            clean = summary[[k for k in summary.columns if k.startswith('clean_n_')]].iloc[0]
            run = []  # largest budgets that all differ from the control
            for e in reversed(common):
                if pvals[e] >= 0.05:
                    break
                run.insert(0, e)
            if clean.max() / clean.sum() > 0.95:
                kind, note = 'none', f'Not evaluable: same answer to {int(clean.max())} of {int(clean.sum())} clean images'
            elif model == 'llama':
                kind, note = 'ref', 'Reference: the effect on the training model'
            elif not run:
                kind, note = 'none', 'No effect beyond the control'
            else:
                gap = out['workload'].loc[run[-1], 'mean'] - ref.loc[run[-1], 'mean']
                same = gap * shifts(task, 'llama')[0]['workload']['mean'].iloc[-1] > 0
                kind = 'same' if same else 'opposite'
                when = f'only at ε = {run[0]:g}' if len(run) == 1 else f'from ε = {run[0]:g}'
                note = ('Transfers ' if same else 'Affects the model ') + when + (', same direction as Llama' if same else ', but in the opposite direction')
            ax.set_facecolor(TINT[kind])
            ax.set_title(note, fontsize=9.5, loc='left', color='#222222', fontweight='bold' if kind in ('same', 'opposite') else 'normal')
            ax.margins(y=0.16)
            ax.grid(alpha=0.25)
            ax.set_xticks(range(len(EPS))); ax.set_xticklabels([f'{e:g}' for e in EPS])
            if r == 0:
                ax.text(0.5, 1.2, title, transform=ax.transAxes, ha='center', va='bottom', fontsize=11)
            if r == 2:
                ax.set_xlabel('perturbation budget ε')
            if c == 0:
                ax.set_ylabel(ylab[task] + '\n(shift as share of the scale)', fontsize=9.5)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle('Does the workload / stress perturbation work on models it was not trained on?', fontsize=13, y=0.985)
    fig.text(0.5, 0.005, 'Panel colour: green = effect beyond the control in the same direction as on Llama, orange = beyond the control but '
             'opposite direction, grey = no effect or not evaluable.   Shaded band = effect beyond the control.\n'
             '★ = differs from the control at p < 0.05 (paired Wilcoxon over images, not corrected for the number of tests; control = '
             'shuffled perturbation, or noise for Llama, where the shuffled version was not run on these tasks).   '
             'Single runs, 274–300 images per panel.', ha='center', fontsize=8.5, color='#555555')
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.subplots_adjust(hspace=0.24)
    _save(fig, '14_cross_model_transfer.png')


# ── 15/16 cross-model heatmaps: one row of task panels per model ──────────────
XM_MODELS = [('llama4-scout', 'Llama 4 Scout (UAPs trained here)'),
             ('qwen2.5-vl-7b', 'Qwen2.5-VL-7B (Alibaba)'),
             ('mistral-small-3.1-24b', 'Mistral-Small-3.1-24B (Mistral)'),
             ('llava-1.5-7b', 'LLaVA-1.5-7B (academic)')]
XM_ATTACKS = [('workload', 'workload / stress'), ('interest', 'interestingness'),
              ('label_loss', 'output labels'), ('excited_vs_angry', 'excitement (vs anger)'),
              ('angry_vs_excited', 'anger (vs excitement)'),
              ('shuffled_workload', 'shuffled workload'), ('noise', 'random noise')]
XM_N_UAPS = 5  # rows above the white line; the rest are controls
SCALE_WIDTH = {'damage_severity': 2.0, 'moral_evaluation': 4.0, 'shopping_relevance': 4.0}


def _xm_dirs(model):
    if model == 'llama4-scout':
        return [RES / 'generalisation', RES / 'generalisation_noise', *sorted(RES.glob('generalisation_label_loss*')),
                RES / 'generalisation_shuffled', RES / 'generalisation_anger']
    return [RES / f'generalisation_cross_model{s}' / model for s in ('', '_label', '_excite', '_anger')]


def _model_summaries(model):
    """All summary.csv rows for one model and task, noise seeds averaged; later dirs win."""
    out = {}
    for t in TASKS:
        parts = [pd.read_csv(d / t / 'summary.csv') for d in _xm_dirs(model) if (d / t / 'summary.csv').exists()]
        if not parts:
            continue
        df = pd.concat(parts).drop_duplicates(['attack', 'epsilon'], keep='last')
        df['attack'] = df.attack.str.replace(r'noise_seed\d', 'noise', regex=True)
        out[t] = df.groupby(['attack', 'epsilon'], as_index=False).mean(numeric_only=True)
    return out


def _model_predictions(model):
    """Per-image shifts for one model, {task: df[sample_id, attack, epsilon, delta]}; later dirs win."""
    out = {}
    for t in TASKS:
        parts = [pd.read_csv(d / t / 'predictions.csv', usecols=['sample_id', 'attack', 'epsilon', 'delta'])
                 for d in _xm_dirs(model) if (d / t / 'predictions.csv').exists()]
        if parts:
            df = pd.concat(parts)
            df = df[df.attack.notna() & (df.attack != 'clean')]
            out[t] = df.drop_duplicates(['sample_id', 'attack', 'epsilon'], keep='last')
    return out


def cross_model_agreement():
    """One row per model × task × perturbation × ε: how many answers moved up / down
    (direction consistency) and, for the transfer models, the Spearman correlation of
    the per-image shift with Llama 4 Scout's shift on the same images. Noise seeds are
    pooled (counts) / averaged (ρ, each seed paired with the same seed on Llama)."""
    from scipy.stats import spearmanr
    preds = {m: p for m, _ in XM_MODELS if (p := _model_predictions(m))}
    llama = preds.get('llama4-scout', {})
    rows = []
    for m, per_task in preds.items():
        for t, df in per_task.items():
            for (a, e), g in df.groupby(['attack', 'epsilon']):
                r = dict(model=m, task=t, attack=a, epsilon=e, n=len(g),
                         n_up=int((g.delta > 0).sum()), n_down=int((g.delta < 0).sum()), rho_llama=np.nan)
                if m != 'llama4-scout' and t in llama:
                    ref = llama[t][(llama[t].attack == a) & np.isclose(llama[t].epsilon, e)]
                    j = g.merge(ref, on='sample_id', suffixes=('', '_llama'))
                    if len(j) > 20 and j.delta.nunique() > 1 and j.delta_llama.nunique() > 1:
                        r['rho_llama'] = spearmanr(j.delta, j.delta_llama)[0]
                rows.append(r)
    df = pd.DataFrame(rows)
    df['attack'] = df.attack.str.replace(r'noise_seed\d', 'noise', regex=True)
    df = df.groupby(['model', 'task', 'attack', 'epsilon'], as_index=False).agg(
        n=('n', 'sum'), n_up=('n_up', 'sum'), n_down=('n_down', 'sum'), rho_llama=('rho_llama', 'mean'))
    moved = df.n_up + df.n_down
    df['direction_consistency'] = (df.n_up - df.n_down) / moved.where(moved > 0)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'cross_model_agreement.csv', index=False)
    return df


def _xm_grid(models, cell, *, vmin, vmax, cmap, cbar, title, footer, fname, signed=True):
    """Task × (perturbation × ε) heatmap panels, one row per model, one shared colour scale.
    cell(model, task, attack, eps, summaries) -> (value, text): text is shown instead of the
    formatted value when given (the reason for a NaN, or the value with its CI)."""
    models = [(m, lab, s) for m, lab in models if (s := _model_summaries(m))]
    row_h = 0.51 * len(XM_ATTACKS)
    fig, axes = plt.subplots(len(models), len(TASKS), figsize=(14, row_h * len(models) + 0.9), squeeze=False)
    for r, (model, label, summ) in enumerate(models):
        for c, t in enumerate(TASKS):
            ax = axes[r, c]
            mat = np.full((len(XM_ATTACKS), len(EPS)), np.nan)
            missing = {}
            df = summ.get(t)
            for i, (a, _) in enumerate(XM_ATTACKS):
                for j, e in enumerate(EPS):
                    mat[i, j], missing[i, j] = cell(model, t, a, e, summ)
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            for i in range(len(XM_ATTACKS)):
                for j in range(len(EPS)):
                    v = mat[i, j]
                    if np.isnan(v):
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='#F3F3F3', edgecolor='#CCCCCC',
                                                   hatch='///', lw=0))
                        ax.text(j, i, missing[i, j], ha='center', va='center', fontsize=7, color='#888888')
                    else:
                        strong = abs(v) > 0.55 * vmax
                        txt = missing[i, j] or ('0.00' if abs(v) < 0.005 else f'{v:+.2f}' if signed else f'{v:.2f}')
                        ax.text(j, i, txt, ha='center', va='center', linespacing=1.1,
                                fontsize=8 if '\n' not in txt else 7.5, color='white' if strong else '#222222')
            # a model that gives (nearly) one answer to every clean image cannot show a shift
            note = ''
            if df is not None:
                clean = df[[k for k in df.columns if k.startswith('clean_n_')]].iloc[0]
                if clean.max() / clean.sum() > 0.95:
                    note = f'not evaluable: {clean.max() / clean.sum():.0%} of clean answers identical'
                    ax.add_patch(plt.Rectangle((-0.5, -0.5), len(EPS), len(XM_ATTACKS), facecolor='white', alpha=0.6, lw=0))
            ax.set_xticks(range(len(EPS))); ax.set_xticklabels([f'ε={e:g}' for e in EPS], fontsize=8.5)
            ax.set_yticks(range(len(XM_ATTACKS)))
            ax.set_yticklabels([l for _, l in XM_ATTACKS] if c == 0 else [], fontsize=8.5)
            ax.axhline(XM_N_UAPS - 0.5, color='white', lw=3)  # UAPs above, controls below
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(TASK_TITLE[t], fontsize=11, fontweight='bold', pad=22)
            if note:
                ax.text(0.5, 1.03, note, transform=ax.transAxes, ha='center', fontsize=8, color='#A33')
            if c == 0:
                ax.text(-0.5, 0.5, label, transform=ax.transAxes, rotation=90, ha='center', va='center',
                        fontsize=10, fontweight='bold')
    fig.subplots_adjust(left=0.155, right=0.88, top=1 - 0.95 / fig.get_figheight(), bottom=0.75 / fig.get_figheight(), hspace=0.35, wspace=0.08)
    cax = fig.add_axes([0.9, 0.2, 0.012, 0.6])
    fig.colorbar(im, cax=cax).set_label(cbar, fontsize=9)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.text(0.5, 0.012, footer, ha='center', fontsize=8.5, color='#555555')
    _save(fig, fname)


_XM_FOOTER = ('Rows above the white line are trained UAPs; below are controls of the same strength '
              '(shuffled = the workload UAP with its pixels permuted; noise = 3 permutation seeds, '
              'seed 0 is identical to the shuffled control).\n')


def fig_cross_model_heatmaps(metric='signed'):
    """Same layout as 08 (task × ε), one row per model, one shared colour scale so
    models compare by colour. metric='signed': normalized signed shift (direction);
    metric='abs': mean |shift| as a share of the scale (how many answers move)."""
    def cell(model, t, a, e, summ):
        df = summ.get(t)
        row = df[(df.attack == a) & np.isclose(df.epsilon, e)] if df is not None else []
        if not len(row):
            return np.nan, 'not run'
        row = row.iloc[0]
        return (row.normalized_signed_shift if metric == 'signed' else row.mean_abs_shift / SCALE_WIDTH[t]), ''
    if metric == 'signed':
        vmin, vmax, cmap, cbar = -0.6, 0.6, 'RdBu_r', 'shift as share of the scale\nred = rated higher, blue = rated lower'
    else:
        vmin, vmax, cmap, cbar = 0.0, 0.6, 'Purples', 'mean |shift| as share of the scale\n(how far answers move, either way)'
    what = 'direction of the shift' if metric == 'signed' else 'size of the shift'
    _xm_grid(XM_MODELS, cell, vmin=vmin, vmax=vmax, cmap=cmap, cbar=cbar, signed=metric == 'signed',
             title=f'UAPs trained on Llama 4 Scout, shown to other models: {what} by perturbation × ε',
             footer=_XM_FOOTER + 'One shared colour scale across all panels. Single runs, 274–300 images per cell; '
                                 'noise = mean of the 3 seeds.',
             fname=f"{'15' if metric == 'signed' else '16'}_cross_model_heatmap_{metric}.png")


def fig_cross_model_heatmap_abs():
    fig_cross_model_heatmaps('abs')


# ── 17/18 does the shift agree? within a model (direction) and with Llama (per image) ──
MIN_MOVED = 10


def _agreement_ci():
    """Bootstrap CIs from analytics/presentation/cross_model_bootstrap.py (a SLURM job), if it has run."""
    p = OUT / 'cross_model_agreement_ci.csv'
    return pd.read_csv(p) if p.exists() else None


def _agreement_cell(column, agr, ci=None):
    """Value, and with bootstrap CIs: '[lo, hi]' below it and * when the paired
    difference from the matched control (noise, else shuffled) excludes 0."""
    def cell(model, t, a, e, summ):
        row = agr[(agr.model == model) & (agr.task == t) & (agr.attack == a) & np.isclose(agr.epsilon, e)]
        if not len(row):
            return np.nan, 'not run'
        row = row.iloc[0]
        if row.n_up + row.n_down < MIN_MOVED:
            return np.nan, f'<{MIN_MOVED} moved'
        if np.isnan(row[column]):
            return np.nan, 'no Llama run' if column == 'rho_llama' else ''
        v, txt = row[column], ''
        b = None if ci is None else ci[(ci.model == model) & (ci.task == t) & (ci.attack == a)
                                       & np.isclose(ci.epsilon, e) & (ci.metric == column)]
        if b is not None and len(b) and np.isclose(b.iloc[0].estimate, v):  # skip CIs from older results
            b = b.iloc[0]
            star = '*' if b.diff_lo > 0 or b.diff_hi < 0 else ''
            txt = f'{v:+.2f}{star}\n[{b.ci_lo:+.2f}, {b.ci_hi:+.2f}]'
        return v, txt
    return cell


_CI_NOTE = ('With bootstrap CIs (2000 image resamples): [95% CI] under each value; * = the paired difference from '
            'the matched control at the same ε (noise, else shuffled) excludes 0.')


def fig_cross_model_direction():
    """17: of the answers that changed, the share that went up minus the share that went
    down. ±1 = every changed answer moved the same way, however few changed."""
    agr, ci = cross_model_agreement(), _agreement_ci()
    _xm_grid(XM_MODELS, _agreement_cell('direction_consistency', agr, ci), vmin=-1, vmax=1, cmap='RdBu_r',
             cbar='(n up − n down) / n changed\n+1 = every changed answer went up, −1 = every one went down',
             title='Is the shift one-directional? Direction consistency of the changed answers, by perturbation × ε',
             footer=_XM_FOOTER + 'Independent of how many answers changed: read together with 15 (net shift) and 16 (size). '
                                 f'Cells with fewer than {MIN_MOVED} changed answers are left blank; noise = 3 seeds pooled.'
                                 + ('\n' + _CI_NOTE if ci is not None else ''),
             fname='17_cross_model_direction_consistency.png')


def fig_cross_model_agreement():
    """18: per-image Spearman ρ between a model's shift and Llama 4 Scout's shift
    under the same perturbation and ε, on the same images."""
    agr, ci = cross_model_agreement(), _agreement_ci()
    models = [(m, lab) for m, lab in XM_MODELS if m != 'llama4-scout']
    _xm_grid(models, _agreement_cell('rho_llama', agr, ci), vmin=-0.6, vmax=0.6, cmap='PRGn',
             cbar='Spearman ρ of the per-image shift with Llama 4 Scout\n(same images, same perturbation and ε)',
             title='Do the same images shift as in Llama 4 Scout? Per-image agreement by perturbation × ε',
             footer=_XM_FOOTER + 'The noise row is the same noise shown to both models: agreement there comes from images '
                                 'that are fragile under any perturbation, so read the UAP rows against it. '
                                 'Noise = mean ρ of the 3 seeds.' + ('\n' + _CI_NOTE if ci is not None else ''),
             fname='18_cross_model_agreement_with_llama.png')


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
    if len(_s.argv) > 1:  # e.g. `... make_presentation_figures.py fig_alignment_vs_slope`
        for _name in _s.argv[1:]:
            globals()[_name]()
        raise SystemExit
    fig_decodability()
    copy_existing()
    fig_dose_response()
    fig_uap_effect()
    fig_uap_examples()
    fig_generalisation_transfer()
    fig_generalisation_examples()
    fig_label_distribution_gradient()
    fig_alignment_vs_slope()
    fig_cross_model_transfer()
    fig_cross_model_heatmaps()
    fig_cross_model_heatmap_abs()
    fig_cross_model_direction()
    fig_cross_model_agreement()
