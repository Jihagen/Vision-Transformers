"""
III.2 Scaling / Dose-Response

Sweep alpha over a range to test whether stronger injection produces stronger output shift.

alpha = -2, -1, -0.5, 0, 0.5, 1, 2

Expected: monotonic shift in label distribution as |alpha| increases.
Non-monotonic behavior signals: instability, wrong vector, wrong layer,
or nonlinear saturation.

Ref: Activation Scaling for Steering and Interpreting Language Models
     (EMNLP 2024 Findings, aclanthology 2024.findings-emnlp.479)
"""
