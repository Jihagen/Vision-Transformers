"""
III.5 Optional Vector Swap

Use after addition / subtraction experiments work.

Example: input persona A + inject persona B direction → output should move toward B.

This is a stronger causal test than simple addition because it requires:
  1. The direction to be specific enough to shift toward B (not just away from A).
  2. The model to respond directionally, not just noisily.

Ref: Persona Vectors (arxiv 2507.21509), SteerVLM (arxiv 2510.26769)
"""
