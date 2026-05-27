"""
Tier I — Representation

Find candidate activation directions for persona and interestingness features.
This tier produces correlational / decodability evidence, not yet causal evidence.

Submodules:
  I1_contrast_design   — controlled contrast prompting + activation collection
  I2_mean_difference   — mean(h_positive) − mean(h_negative) vectors
  I3_linear_probe      — linear separator / CAV-style probe
  I4_vector_evaluation — decide whether a candidate vector is worth keeping
  I5_pca_svd           — optional: test whether concept needs a subspace
"""
