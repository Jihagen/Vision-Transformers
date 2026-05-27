"""
II.2 Interestingness Vector Analytics

Interprets the structure of the interestingness direction.

Questions answered here:
  - Is there a global interestingness direction shared across personas?
  - Is interestingness persona-specific (different direction per persona)?
  - Do different personas have different interest directions?
  - Does interestingness align with emotion, novelty, aesthetics, or topic/style?
  - Does the interestingness vector look like one dimension or a compound space?

Important: avoid relying only on the rating-token representation — it may encode
the phrase "very interesting" rather than the decision state.  Use intermediate
LLM layer activations from the teacher-forced step instead.

Ref: Representation Engineering (arxiv 2310.01405),
     TCAV (arxiv 1711.11279),
     Language Models Linearly Represent Sentiment (ACL 2024 BlackboxNLP)
"""
