"""
Tier III — Control

Test whether candidate vectors causally influence model output.
Input prompt and image are NOT changed — we intervene on activations only.

Submodules:
  III1_vector_addition    — h' = h + alpha * v  (test sufficiency)
  III2_scaling            — sweep alpha to check dose-response
  III3_vector_subtraction — h' = h - alpha * v  or project out v (ablation)
  III4_combined_injection — sum of single-feature vectors vs compound persona
  III5_vector_swap        — replace persona A activations with persona B direction

All control experiments require:
  1. A loaded model (GPU, same setup as collection)
  2. Injection hooks from utils/hooks.py registered before inference
  3. Candidate vectors from Tier I (I.2 mean-diff or I.3 CAV)

III.6 Causal caution (no code):
  Control shows a direction is sufficient to influence output.
  It does not prove this is the unique true representation.
"""
