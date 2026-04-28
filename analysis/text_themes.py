#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""
Thematic analysis of model explanations via LDA topic modelling.

Strategy
--------
Explanations are ~17 words each — too short for stable per-document LDA.
Solution: fit LDA on *per-image meta-documents* (all 9 conditions concatenated
per image → 500 docs, ~83 content words each).  Then transform each individual
explanation through the fitted model for per-condition / per-label breakdowns.

Outputs (under results/experiments/gender_emotion_analysis/lda/)
  topics_words.png          — bar charts, top words per topic
  topics_words.txt          — plain-text topic descriptions
  topic_condition_heatmap.png  — 9 conditions × n_topics (mean topic weight)
  topic_label_heatmap.png      — 5 labels × n_topics (across all conditions)
  topic_x_interestingness.png  — per topic: mean interestingness score by condition
  topic_condition_label_detail.png — per condition: label × topic weight grid
  lda_model_info.txt           — vocabulary size, perplexity, topic words
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ───────────────────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────────────────

_GE_DIR = 'data/experiments/gender_emotion'
OUT_DIR = 'results/experiments/gender_emotion_analysis/lda'

EXPERIMENTS = [
    'female_anger', 'female_fear', 'female_disgust', 'female_sad',
    'female_amusement', 'female_awe', 'female_contentment',
    'female_excitement', 'male_anger',
]

LABEL_ORDER = [
    'Not Interesting', 'Slightly Interesting', 'Moderately Interesting',
    'Very Interesting', 'Extremely Interesting',
]
LABEL_SCORE = {l: i for i, l in enumerate(LABEL_ORDER)}

SHORT_NAMES = {
    'female_anger': 'F-Anger', 'female_fear': 'F-Fear',
    'female_disgust': 'F-Disgust', 'female_sad': 'F-Sad',
    'female_amusement': 'F-Amusement', 'female_awe': 'F-Awe',
    'female_contentment': 'F-Contentment', 'female_excitement': 'F-Excitement',
    'male_anger': 'M-Anger',
}

N_TOPICS = 10

# ── Evaluative / template vocabulary to suppress ──────────────────────────────
# These are structural words the model uses to frame its evaluation, not to
# describe image content.  Removing them forces LDA to cluster on what's
# actually in the image.
EVALUATIVE_STOP = {
    # evaluation mechanics
    'evoke', 'evokes', 'evoked', 'evoking',
    'spark', 'sparks', 'sparked', 'sparking',
    'capture', 'captures', 'captured', 'capturing',
    'catches', 'caught', 'catch', 'holds', 'hold', 'held', 'holding',
    'resonates', 'resonate', 'resonated', 'resonating',
    'contrasts', 'contrast', 'contrasted', 'contrasting',
    'aligns', 'align', 'aligned',
    # hedging / filler
    'somewhat', 'rather', 'quite', 'particularly', 'especially',
    'briefly', 'brief', 'moment', 'momentarily',
    'seem', 'seems', 'seemed', 'appear', 'appears', 'appeared',
    'suggest', 'suggests', 'suggested', 'indicating', 'indicates',
    'typically', 'typical', 'usual', 'usually', 'ordinary', 'mundane',
    'common', 'commonly', 'simple', 'simply', 'certain', 'certain',
    'particular', 'generally', 'overall', 'often', 'sometimes',
    # emotional evaluation tokens
    'strong', 'stronger', 'strongest', 'weak', 'weaker',
    'emotion', 'emotions', 'emotional', 'feeling', 'feelings', 'feel', 'feels',
    'attention', 'curiosity', 'curious', 'wonder', 'sense',
    'mood', 'state', 'current',
    'interest', 'interested', 'interesting', 'interesting', 'fascinated',
    'fascinating', 'compelling', 'captivating', 'captivated',
    'significant', 'significance', 'notable', 'noteworthy',
    'intriguing', 'intrigued', 'engaging', 'engaged',
    # persona-state emotion words (appear in "my current [X] mood" and as
    # evaluative descriptors — not reliably describing image content)
    'anger', 'angry', 'fear', 'fearful', 'fearing',
    'disgust', 'disgusted', 'disgusting',
    'sadness', 'saddened',
    'unease', 'uneasy', 'unsettling', 'unsettled',
    'excitement', 'excited',
    # generic scene tokens
    'image', 'scene', 'photo', 'picture', 'shows', 'show', 'shown',
    'depicts', 'depict', 'depicting', 'featuring', 'features', 'displayed',
    'taken', 'captured', 'view', 'look', 'looks',
    # LM connectives that bleed into template
    'doesn', 'isn', 'wasn', 'wouldn', 'couldn',
    'due', 'given', 'because', 'since', 'though', 'although', 'despite',
    'however', 'while', 'whereas', 'yet', 'still',
    # catch-all short words not in sklearn stop list
    'the', 'and', 'but', 'for', 'are', 'was', 'were', 'has', 'had',
    'its', 'this', 'that', 'with', 'from', 'into', 'about', 'than',
    'not', 'also', 'any', 'all', 'some', 'each', 'both', 'more',
    'just', 'make', 'makes', 'made', 'get', 'got', 'take', 'taken',
    'one', 'two', 'three', 'can', 'could', 'would', 'will', 'may',
    'might', 'must', 'shall', 'have', 'being', 'been', 'which', 'who',
    'what', 'where', 'when', 'how', 'why', 'such', 'like', 'well',
    'too', 'very', 'even', 'most', 'many', 'much', 'way', 'use',
}


# ───────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ───────────────────────────────────────────────────────────────────────────────

# Clause-level stripping — removes evaluative suffixes from content sentences
_CLAUSE_PATTERNS = [
    # ", which doesn't evoke …" / ", which is a common scene"
    r',\s*which\s+(?:is|are|was|were|could|might|may|doesn|don|isn|can|has|had|does)\b.*',
    # ", but it doesn't / but doesn't …"
    r',\s*but\s+(?:it\s+)?(?:doesn|don|isn|can|couldn|wouldn)\b.*',
    # "which is a common …" standalone
    r'\bwhich\s+is\s+(?:a\s+)?(?:common|typical|usual|simple|ordinary)\b.*',
    # "but doesn't hold …"
    r'\bbut\s+doesn.?t\s+(?:hold|evoke|capture|spark|seem|make|feel)\b.*',
    # "due to my current …" / "because of my …"
    r'(?:due to|because of)\s+my\s+(?:current\s+)?\w+\s+(?:state|mood|feeling|situation)\b.*',
    # "my current [emotion] mood/state"
    r'my\s+current\s+\w+\s+(?:mood|state|feeling|situation)\b[^.]*',
    # "The image does not evoke …" (whole sentence worth dropping)
    r'\bthe image does not evoke\b.*',
    # "doesn't evoke strong emotions" — trailing clause
    r"doesn.?t evoke\s+(?:strong\s+)?(?:any\s+)?(?:emotions?|curiosity|interest|response)\b.*",
]
_CLAUSE_RE = [re.compile(p, re.IGNORECASE) for p in _CLAUSE_PATTERNS]


def clean_text(text: str) -> str:
    for pat in _CLAUSE_RE:
        text = pat.sub('', text)
    # Collapse multiple spaces / punctuation artefacts
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s*[,;]\s*$', '', text)
    return text.strip()


# ───────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ───────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[dict, list]:
    """
    Returns
    -------
    records : dict  image_idx → {exp_name → {'label': str, 'text': str, 'text_clean': str}}
    meta    : list of dicts with all flat fields for DataFrame construction
    """
    records: dict[int, dict] = defaultdict(dict)
    meta = []

    for exp in EXPERIMENTS:
        path = f'{_GE_DIR}/results_{exp}.npy'
        d = np.load(path, allow_pickle=True).item()
        lkey = 'interestingness_label' if 'interestingness_label' in d['results'][0] else 'interestingness'
        for i, r in enumerate(d['results']):
            raw  = r['explanation']
            clean = clean_text(raw)
            records[i][exp] = {
                'label':      r[lkey],
                'filename':   r.get('filename', ''),
                'text_raw':   raw,
                'text_clean': clean,
            }
            meta.append({
                'image_idx': i,
                'filename':  r.get('filename', ''),
                'condition': exp,
                'label':     r[lkey],
                'label_score': LABEL_SCORE.get(r[lkey], 2),
                'text_raw':  raw,
                'text_clean': clean,
            })

    return records, meta


# ───────────────────────────────────────────────────────────────────────────────
# LDA PIPELINE
# ───────────────────────────────────────────────────────────────────────────────

def build_meta_docs(records: dict) -> list[str]:
    """One document per image: all 9 cleaned conditions concatenated."""
    docs = []
    for i in sorted(records.keys()):
        combined = ' '.join(records[i][exp]['text_clean'] for exp in EXPERIMENTS)
        docs.append(combined)
    return docs


def fit_lda(meta_docs: list[str], n_topics: int = N_TOPICS):
    vectorizer = CountVectorizer(
        stop_words='english',        # sklearn built-in stop list
        min_df=5,                    # word must appear in ≥5 images
        max_df=0.55,                 # word in ≤55% of images (kills near-universal template words)
        ngram_range=(1, 2),
        token_pattern=r'\b[a-zA-Z]{3,}\b',
    )

    # Add our evaluative stopwords by post-filtering vocabulary
    vectorizer.fit(meta_docs)
    vocab = vectorizer.get_feature_names_out()
    bad = {
        i for i, w in enumerate(vocab)
        if any(tok in EVALUATIVE_STOP for tok in w.split())
    }
    # Rebuild with restricted vocabulary
    clean_vocab = [w for i, w in enumerate(vocab) if i not in bad]
    print(f"  Vocabulary: {len(vocab)} → {len(clean_vocab)} after evaluative filter")

    vectorizer2 = CountVectorizer(
        vocabulary=clean_vocab,
        token_pattern=r'\b[a-zA-Z]{3,}\b',
    )
    X = vectorizer2.fit_transform(meta_docs)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        learning_method='batch',    # batch EM → more stable on 500 docs
        random_state=42,
        doc_topic_prior=0.1,        # sparse doc-topic: each image has few dominant themes
        topic_word_prior=0.01,
    )
    lda.fit(X)
    print(f"  LDA perplexity on meta-docs: {lda.perplexity(X):.1f}")

    return lda, vectorizer2, X


def get_topic_words(lda, vectorizer, n_words: int = 12) -> list[list[str]]:
    vocab = vectorizer.get_feature_names_out()
    topics = []
    for comp in lda.components_:
        top_idx = comp.argsort()[::-1][:n_words]
        topics.append([vocab[i] for i in top_idx])
    return topics


def transform_individual(meta, lda, vectorizer) -> np.ndarray:
    """Transform every individual explanation → topic distribution (n_meta × n_topics)."""
    texts = [m['text_clean'] for m in meta]
    X = vectorizer.transform(texts)
    return lda.transform(X)   # shape: (n_docs, n_topics)


# ───────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ───────────────────────────────────────────────────────────────────────────────

def _topic_label(topic_words: list[list[str]], t: int, n: int = 4) -> str:
    return f"T{t+1}: {' / '.join(topic_words[t][:n])}"


def plot_topic_words(topic_words: list[list[str]], lda, vectorizer, out_path: str):
    n = len(topic_words)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = axes.flatten()
    vocab = vectorizer.get_feature_names_out()

    for t, (words, ax) in enumerate(zip(topic_words, axes)):
        comp = lda.components_[t]
        scores = comp[np.array([np.where(vocab == w)[0][0] for w in words if w in vocab])]
        scores = scores / scores.sum()
        ax.barh(range(len(words)), scores[::-1], color=plt.colormaps['tab10'](t / n))
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words[::-1], fontsize=8)
        ax.set_xlabel('Rel. weight', fontsize=8)
        ax.set_title(f'Topic {t+1}', fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes[len(topic_words):]:
        ax.set_visible(False)

    plt.suptitle('LDA Topics — Top Words (image-content vocabulary)', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_condition_topic_heatmap(df_meta: pd.DataFrame, topic_words: list[list[str]], out_path: str):
    """Mean topic weight per condition (rows) × topic (columns)."""
    topic_cols = [c for c in df_meta.columns if c.startswith('topic_')]
    col_labels  = [_topic_label(topic_words, int(c.split('_')[1])) for c in topic_cols]

    cond_mean = df_meta.groupby('condition')[topic_cols].mean()
    cond_mean.index = [SHORT_NAMES.get(i, i) for i in cond_mean.index]
    cond_mean.columns = col_labels

    # Order rows: negative → positive emotion, male last
    order = ['F-Disgust','F-Anger','F-Fear','F-Sad','F-Contentment','F-Amusement','F-Excitement','F-Awe','M-Anger']
    cond_mean = cond_mean.reindex([o for o in order if o in cond_mean.index])

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        cond_mean, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
        linewidths=0.4, cbar_kws={'label': 'Mean topic weight'},
        annot_kws={'size': 7},
    )
    ax.set_title('Mean Topic Weight per Condition\n(rows ordered neg→pos emotion)', fontsize=11, pad=10)
    ax.set_xlabel('LDA Topic', fontsize=9)
    ax.set_ylabel('Condition', fontsize=9)
    ax.tick_params(axis='x', rotation=40, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_label_topic_heatmap(df_meta: pd.DataFrame, topic_words: list[list[str]], out_path: str):
    """Mean topic weight per label × topic across all conditions."""
    topic_cols = [c for c in df_meta.columns if c.startswith('topic_')]
    col_labels  = [_topic_label(topic_words, int(c.split('_')[1])) for c in topic_cols]

    label_mean = df_meta.groupby('label')[topic_cols].mean().reindex(LABEL_ORDER)
    label_mean.columns = col_labels

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        label_mean, annot=True, fmt='.2f', cmap='Blues', ax=ax,
        linewidths=0.4, cbar_kws={'label': 'Mean topic weight'},
        annot_kws={'size': 8},
    )
    ax.set_title('Mean Topic Weight per Interestingness Label (all conditions)', fontsize=11, pad=10)
    ax.set_xlabel('LDA Topic', fontsize=9)
    ax.set_ylabel('Interestingness Label', fontsize=9)
    ax.tick_params(axis='x', rotation=40, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_topic_interestingness(df_meta: pd.DataFrame, topic_words: list[list[str]], out_path: str):
    """
    For each topic: scatter of (mean topic weight in doc) vs mean label_score,
    grouped by condition — shows which image themes drive high/low interestingness
    per emotion condition.
    """
    topic_cols = [c for c in df_meta.columns if c.startswith('topic_')]
    n_topics = len(topic_cols)
    ncols = 5
    nrows = (n_topics + ncols - 1) // ncols

    cmap = plt.colormaps['tab10']
    conditions = EXPERIMENTS

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = axes.flatten()

    for t, (tc, ax) in enumerate(zip(topic_cols, axes)):
        for ci, cond in enumerate(conditions):
            sub = df_meta[df_meta['condition'] == cond]
            # Bin by topic weight quartile, compute mean label score
            q = pd.qcut(sub[tc], q=4, duplicates='drop')
            grp = sub.groupby(q, observed=True)['label_score'].mean()
            # x = bin midpoint, y = mean label score
            if len(grp) >= 2:
                xs = [iv.mid for iv in grp.index]
                ys = grp.values
                ax.plot(xs, ys, 'o-', color=cmap(ci / len(conditions)),
                        label=SHORT_NAMES.get(cond, cond), linewidth=1.2,
                        markersize=4, alpha=0.8)

        ax.set_title(f'T{t+1}: {" / ".join(topic_words[t][:3])}', fontsize=8, fontweight='bold')
        ax.set_xlabel('Topic weight quartile midpoint', fontsize=7)
        ax.set_ylabel('Mean label score (0-4)', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right',
               bbox_to_anchor=(1.0, 0.0), fontsize=7, title='Condition', ncol=1)

    for ax in axes[n_topics:]:
        ax.set_visible(False)

    plt.suptitle('Topic Weight vs Interestingness Score — by Condition\n'
                 '(upward slope = images with more of this theme get higher ratings)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_condition_label_detail(df_meta: pd.DataFrame, topic_words: list[list[str]], out_path: str):
    """
    Grid: one row per condition, one column per label.
    Each cell is a horizontal bar of dominant topic weights for that (cond, label) group.
    """
    topic_cols = [c for c in df_meta.columns if c.startswith('topic_')]
    n_topics = len(topic_cols)
    conditions = EXPERIMENTS
    labels = [l for l in LABEL_ORDER if l in df_meta['label'].unique()]

    fig, axes = plt.subplots(len(conditions), len(labels),
                              figsize=(len(labels) * 3.2, len(conditions) * 2.4),
                              sharey=True)

    cmap = plt.colormaps['tab10']
    topic_colors = [cmap(t / n_topics) for t in range(n_topics)]

    for ri, cond in enumerate(conditions):
        for ci, label in enumerate(labels):
            ax = axes[ri][ci]
            sub = df_meta[(df_meta['condition'] == cond) & (df_meta['label'] == label)]
            if len(sub) == 0:
                ax.set_visible(False)
                continue
            mean_weights = sub[topic_cols].mean().values
            top_idx = mean_weights.argsort()[::-1][:5]   # show top-5 topics
            ax.barh(range(5), mean_weights[top_idx][::-1],
                    color=[topic_colors[i] for i in top_idx[::-1]])
            ax.set_yticks(range(5))
            ax.set_yticklabels([f'T{top_idx[::-1][k]+1}' for k in range(5)], fontsize=7)
            ax.set_xlim(0, 0.5)
            ax.tick_params(axis='x', labelsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            n_str = f'n={len(sub)}'
            ax.set_title(n_str, fontsize=6, pad=2)

            if ri == 0:
                ax.set_xlabel(label[:12], fontsize=7, labelpad=3)
                ax.xaxis.set_label_position('top')
            if ci == 0:
                ax.set_ylabel(SHORT_NAMES.get(cond, cond), fontsize=7, labelpad=4)

    plt.suptitle('Dominant Topics per (Condition × Label)\nTop-5 topics shown per cell',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cross_condition_topic_comparison(df_meta: pd.DataFrame, topic_words: list[list[str]], out_path: str):
    """
    For the shared label 'Moderately Interesting': how do topic distributions
    differ across conditions?  Same for 'Slightly Interesting'.
    Reveals what kinds of images each emotion condition experiences as 'moderate'.
    """
    topic_cols = [c for c in df_meta.columns if c.startswith('topic_')]
    col_labels  = [_topic_label(topic_words, int(c.split('_')[1]), n=3) for c in topic_cols]
    focus_labels = ['Slightly Interesting', 'Moderately Interesting']

    fig, axes = plt.subplots(1, len(focus_labels), figsize=(16, 5), sharey=False)

    order = ['female_disgust','female_anger','female_fear','female_sad',
             'female_contentment','female_amusement','female_excitement','female_awe','male_anger']

    for ax, focus_label in zip(axes, focus_labels):
        sub = df_meta[df_meta['label'] == focus_label]
        grp = sub.groupby('condition')[topic_cols].mean()
        grp = grp.reindex([o for o in order if o in grp.index])
        grp.index = [SHORT_NAMES.get(i, i) for i in grp.index]
        grp.columns = col_labels

        sns.heatmap(
            grp, annot=True, fmt='.2f', cmap='PuBu', ax=ax,
            linewidths=0.3, cbar_kws={'label': 'Mean weight'},
            annot_kws={'size': 7},
        )
        ax.set_title(f'"{focus_label}"\nacross conditions', fontsize=10, pad=8)
        ax.set_xlabel('LDA Topic', fontsize=8)
        ax.set_ylabel('Condition', fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    plt.suptitle('Topic Profiles for the Same Label Across Conditions\n'
                 '(which image themes trigger each rating under different emotions)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ───────────────────────────────────────────────────────────────────────────────
# TEXT REPORT
# ───────────────────────────────────────────────────────────────────────────────

def write_topic_report(topic_words: list[list[str]], df_meta: pd.DataFrame,
                        lda, vectorizer, meta_docs: list[str], out_path: str):
    topic_cols = [c for c in df_meta.columns if c.startswith('topic_')]
    lines = []
    lines.append("LDA TOPIC MODEL — Gender × Emotion Experiment")
    lines.append(f"n_topics={len(topic_words)}, n_docs(meta)=500, vocab={len(vectorizer.get_feature_names_out())}")
    lines.append("")

    for t, words in enumerate(topic_words):
        lines.append(f"Topic {t+1:02d}:  {', '.join(words)}")
        # Top conditions for this topic
        cond_means = df_meta.groupby('condition')[topic_cols[t]].mean().sort_values(ascending=False)
        lines.append(f"  → Strongest condition: {cond_means.index[0]} ({cond_means.iloc[0]:.3f})")
        # Top label for this topic (across all conditions)
        lbl_means = df_meta.groupby('label')[topic_cols[t]].mean().sort_values(ascending=False)
        lines.append(f"  → Strongest label:     {lbl_means.index[0]} ({lbl_means.iloc[0]:.3f})")
        lines.append("")

    # Which topic best separates labels?
    lines.append("TOPIC DISCRIMINATIVENESS (variance of mean weight across labels):")
    for tc in topic_cols:
        lv = df_meta.groupby('label')[tc].mean().var()
        lines.append(f"  {tc}  var={lv:.5f}")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {out_path}")


# ───────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    print("1. Loading and cleaning data...")
    records, meta = load_data()
    print(f"   {len(meta)} individual explanations loaded.")

    print("2. Building per-image meta-documents...")
    meta_docs = build_meta_docs(records)

    print("3. Fitting LDA...")
    lda, vectorizer, X_meta = fit_lda(meta_docs, n_topics=N_TOPICS)
    topic_words = get_topic_words(lda, vectorizer, n_words=12)

    print("4. Transforming individual explanations...")
    topic_weights = transform_individual(meta, lda, vectorizer)

    # Attach topic weights to meta records
    for i, (m, tw) in enumerate(zip(meta, topic_weights)):
        for t, w in enumerate(tw):
            m[f'topic_{t}'] = float(w)

    df_meta = pd.DataFrame(meta)

    print("5. Generating plots and reports...\n")

    plot_topic_words(
        topic_words, lda, vectorizer,
        os.path.join(OUT_DIR, 'topics_words.png')
    )
    plot_condition_topic_heatmap(
        df_meta, topic_words,
        os.path.join(OUT_DIR, 'topic_condition_heatmap.png')
    )
    plot_label_topic_heatmap(
        df_meta, topic_words,
        os.path.join(OUT_DIR, 'topic_label_heatmap.png')
    )
    plot_topic_interestingness(
        df_meta, topic_words,
        os.path.join(OUT_DIR, 'topic_x_interestingness.png')
    )
    plot_condition_label_detail(
        df_meta, topic_words,
        os.path.join(OUT_DIR, 'topic_condition_label_detail.png')
    )
    plot_cross_condition_topic_comparison(
        df_meta, topic_words,
        os.path.join(OUT_DIR, 'topic_same_label_cross_condition.png')
    )
    write_topic_report(
        topic_words, df_meta, lda, vectorizer, meta_docs,
        os.path.join(OUT_DIR, 'topics_words.txt')
    )

    # Save model artefacts for manual inspection / reuse
    with open(os.path.join(OUT_DIR, 'lda_artefacts.pkl'), 'wb') as f:
        pickle.dump({'lda': lda, 'vectorizer': vectorizer,
                     'topic_words': topic_words, 'df_meta': df_meta}, f)
    print(f"  Saved: {os.path.join(OUT_DIR, 'lda_artefacts.pkl')}")

    print(f"\nAll outputs in: {OUT_DIR}")
