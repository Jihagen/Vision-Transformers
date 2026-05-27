"""
II.1 Persona Vector Analytics

Interprets the geometric structure of persona-feature vectors.

Questions answered here:
  - Is gender represented as a binary contrast or a gradient?
  - Do countries / jobs / emotions form meaningful distance structures?
  - Do some persona features fail to produce usable vectors?
  - Where in the model (which layers) do persona-feature vectors appear?

Methods:
  gdv.py             — General Discrimination Value (migrated from metrics/gdv.py)
  pca_embed.py       — PCA 2D scatter visualization (migrated from metrics/metrics.py)
  umap_embed.py      — UMAP embedding + trustworthiness (migrated from metrics/umap.py)
  tsne_embed.py      — t-SNE embedding + trustworthiness (migrated from metrics/tsne.py)
  cosine_similarity  — pairwise vector comparisons across persona conditions
  distance_matrices  — distance matrix heatmaps for persona/style/topic geometry
  layer_heatmaps     — layer-wise heatmaps locating where concepts appear

Ref: RSA (Frontiers in Systems Neuroscience 2008), CKA (arxiv 1905.00414),
     Linear Representation Hypothesis (arxiv 2311.03658)
"""
