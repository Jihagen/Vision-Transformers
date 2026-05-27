"""
I.3 Linear Separator / CAV-Style Probe

Train a linear model to separate positive and negative examples.
Linear probing and CAV/TCAV-style vector finding are one method family here —
not two separate approaches.

Jobs:
  - Use model accuracy as evidence of decodability
  - Extract the classifier's weight normal vector as a CAV-style concept direction
  - This gives a second candidate vector alongside the mean-difference vector (I.2)

Ref: TCAV (arxiv 1711.11279), Representation Engineering (arxiv 2310.01405),
     Language Models Linearly Represent Sentiment (ACL 2024 BlackboxNLP),
     Linear Representation Hypothesis (arxiv 2311.03658)
"""
