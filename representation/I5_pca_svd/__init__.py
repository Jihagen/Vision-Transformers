"""
I.5 Optional PCA/SVD Dimensionality Check

Use only if the one-vector story looks weak:
  - mean-difference vector is weak (low AUROC from I.4)
  - probe works but the vector is unstable across folds
  - the concept seems compound (e.g. interestingness decomposing into several
    orthogonal axes — novelty, aesthetics, topic salience)
  - later control experiments fail despite good representation evidence

Job: test whether a concept is better understood as a subspace (k directions)
rather than a single vector.

Ref: Linear Representation Hypothesis (arxiv 2311.03658),
     FALCON (arxiv 2307.10504), Conceptors (arxiv 2410.16314)
"""
