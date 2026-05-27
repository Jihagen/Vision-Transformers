"""
I.2 Mean-Difference Vectors

Compute v = mean(h_positive) − mean(h_negative) for each contrast pair.

This is the simplest candidate direction and the most direct object to inject
later for steering (see control/III1_vector_addition).

Persona contrasts:   happy − sad, high_load − low_load, female − male,
                     Nigeria − Sweden, high_interest − low_interest, etc.
Interestingness:     high-interest activations − low-interest activations

Ref: ActAdd (arxiv 2308.10248), Persona Vectors (arxiv 2507.21509),
     Representation Engineering (arxiv 2310.01405)
"""
