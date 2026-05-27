"""
Prompt construction for persona-conditioned interestingness rating.

A prompt consists of:
  1. A persona description block (empty for baseline runs)
  2. The rating task instruction
  3. The image placeholder

The teacher-forced anchor suffix used for LLM activation capture is also defined here.

Migrated from: get_activations_base.py (build_persona_prompt)
"""

from __future__ import annotations


# Suffix appended before the model's label token during the teacher-forced step.
RATING_ANCHOR = '{"interestingness":"'

# Valid interestingness labels expected in JSON output.
VALID_LABELS = [
    "Not Interesting",
    "Slightly Interesting",
    "Moderately Interesting",
    "Very Interesting",
    "Extremely Interesting",
]


def build_persona_prompt(persona_dict: dict | None) -> str:
    """
    Construct the full text prompt for a given persona.

    Args:
        persona_dict: dict with keys like 'gender', 'emotion', 'mental_workload',
                      'job', 'country', 'age', etc.  Pass None or {} for baseline.

    Returns:
        Prompt string (without image token — processor adds that).
    """
    raise NotImplementedError


def build_blank_prompt() -> str:
    """Return the baseline prompt with no persona description."""
    return build_persona_prompt(persona_dict=None)


def parse_model_response(raw_text: str) -> dict:
    """
    Parse the model's JSON output into a structured result dict.

    Expected keys in response: "interestingness", "explanation".
    Raises ValueError if the response cannot be parsed into a valid label.
    """
    raise NotImplementedError
