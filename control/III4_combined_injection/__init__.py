"""
III.4 Combined Feature-Vector Intervention

Combine single-feature vectors (e.g. emotion + job + country + cognitive-load)
and inject the sum into the model.

Compare the resulting output / activation shift to the observed compound-persona
condition (collected in I.1).

If the combined intervention approximates the compound persona effect:
  → supports compositional control of persona features.
If it fails:
  → persona behavior may be nonlinear, interaction-heavy, or located in a
     subspace rather than a simple vector sum.

This replaces composition as a strong analytics claim (cf. II.3 note).

Ref: Compositional Affine Steering (OpenReview 0Yu0eNdHyV),
     Conceptors (arxiv 2410.16314),
     Linear Representation Hypothesis (arxiv 2311.03658)
"""
