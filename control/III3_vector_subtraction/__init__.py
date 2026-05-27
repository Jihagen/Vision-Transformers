"""
III.3 Vector Subtraction / Ablation

h' = h - alpha * v     (soft suppression)
  OR
h' = h - (h · v̂) * v̂  (full projection removal)

Tests whether removing a direction weakens persona-conditioned behavior.

Expected:
  - Removing a persona-feature direction should weaken persona-conditioned
    output (e.g. gender-neutral rationale despite female persona prompt).
  - Removing the interestingness direction should weaken high-interest
    response tendency (ratings should flatten toward the midpoint).

Ref: Persona Vectors (arxiv 2507.21509),
     ActAdd (arxiv 2308.10248),
     Representation Engineering (arxiv 2310.01405)
"""
