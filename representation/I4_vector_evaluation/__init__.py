"""
I.4 Basic Vector Evaluation

Decides whether a candidate vector is worth keeping before moving on to
analytics or control.  Prevents treating every contrast as meaningful.

Checks:
  1. Projection separation      — does projecting onto v separate pos from neg?
  2. Layer profile              — does the signal appear in specific layers or everywhere?
  3. Mean-diff vs CAV alignment — do I.2 and I.3 vectors agree (cosine similarity)?
  4. Probe accuracy above chance — is the linear probe significantly above 50 %?
  5. Signal vs baseline         — is the separation stronger than a null/random vector?

Ref: TCAV (arxiv 1711.11279), Representation Engineering (arxiv 2310.01405),
     Linear Representation Hypothesis (arxiv 2311.03658)
"""
