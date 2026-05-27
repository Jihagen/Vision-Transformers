"""
III.1 Vector Addition

h' = h + alpha * v

Add emotion, cognitive-load, job, country, or interestingness vectors to the
model's hidden state at a chosen layer, without changing the input.

Expected results:
  - Persona features: rationale / rating criteria in the output should shift.
  - Interestingness: the rating itself should shift toward the injected direction.

Ref: ActAdd (arxiv 2308.10248),
     Contrastive Activation Addition (arxiv 2312.06681),
     Persona Vectors (arxiv 2507.21509),
     Representation Engineering (arxiv 2310.01405)
"""
