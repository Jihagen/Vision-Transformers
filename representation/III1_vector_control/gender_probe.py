"""
Gender-coding probe for explanation text.

A TF-IDF + LogisticRegression classifier trained on the existing
gender_emotion explanations, where the label is the PERSONA's stated
gender (female=1, male=0). This captures whatever gender-correlated
language patterns exist in model output under explicit "Gender: X"
prompting — including, but not limited to, direct prompt-echo.

Use case
--------
The trained probe is applied to explanations generated under a BLANK
(no-persona) prompt with gender-vector injection/ablation at
language_29_D5120. The blank prompt has no "Gender: X" field to echo,
so any shift in probe score between blank conditions must originate
from the injected/ablated activation itself — a confound-free test of
whether the vector carries gender-coded information that surfaces in
language.

score_explanations() returns signed decision-function values:
positive => more "female"-coded, negative => more "male"-coded,
per the trained probe's convention.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

_DEFAULT_BASE_DIR = Path("data/experiments/gender_emotion")
_EMOTIONS = ["anger", "amusement", "awe", "contentment", "disgust", "excitement", "fear", "sad"]


def load_gender_corpus(
    base_dir: str | Path = _DEFAULT_BASE_DIR,
    emotions: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Load explanation texts + gender labels (1=female, 0=male) pooled
    across the given emotions (default: all 8 base gender_emotion conditions).

    Pooling across emotions (rather than a single emotion like contentment)
    yields a probe that targets gender-coded language in general, not
    language tied to one emotional register — better suited to transfer
    to the emotion-unspecified blank prompt.
    """
    base_dir = Path(base_dir)
    emotions = emotions if emotions is not None else _EMOTIONS

    texts: list[str] = []
    labels: list[int] = []
    for emotion in emotions:
        female = np.load(base_dir / f"results_female_{emotion}.npy", allow_pickle=True).item()
        male   = np.load(base_dir / f"results_male_{emotion}.npy", allow_pickle=True).item()
        texts += [r["explanation"] for r in female["results"]]
        labels += [1] * len(female["results"])
        texts += [r["explanation"] for r in male["results"]]
        labels += [0] * len(male["results"])

    return texts, np.array(labels)


def train_gender_probe(
    texts: list[str],
    labels: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[Pipeline, float]:
    """
    Fit a TF-IDF + LogisticRegression pipeline on (texts, labels).

    Returns (fitted_pipeline, mean cross-validated accuracy).
    The CV score is computed on un-fitted folds before the final fit on
    all data, so it reflects out-of-sample decodability.
    """
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipe, texts, labels, cv=cv, scoring="accuracy")
    pipe.fit(texts, labels)
    return pipe, float(cv_scores.mean())


def score_explanations(pipe: Pipeline, texts: list[str]) -> np.ndarray:
    """
    Signed decision-function score per explanation: positive => more
    "female"-coded, negative => more "male"-coded, magnitude reflects
    confidence (distance from the decision boundary).
    """
    return pipe.decision_function(texts)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    texts, labels = load_gender_corpus()
    print(f"Corpus: {len(texts)} explanations "
          f"({int(labels.sum())} female / {int((1 - labels).sum())} male)")

    pipe, cv_acc = train_gender_probe(texts, labels)
    print(f"5-fold CV accuracy (gender-coding decodability): {cv_acc:.4f}")

    # Sanity check: scores on the training corpus should separate by label.
    scores = score_explanations(pipe, texts)
    female_mean = scores[labels == 1].mean()
    male_mean = scores[labels == 0].mean()
    print(f"Mean score | female-labeled: {female_mean:+.3f} | male-labeled: {male_mean:+.3f}")
