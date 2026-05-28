"""
Prompt construction and response parsing for persona-conditioned interestingness rating.

Migrated from: get_activations_base.py (build_persona_prompt, extract_last_json_block)
"""

from __future__ import annotations
import json
import re

VALID_LABELS = [
    "Not Interesting",
    "Slightly Interesting",
    "Moderately Interesting",
    "Very Interesting",
    "Extremely Interesting",
]

# Suffix appended before the model's label token during the teacher-forced step.
RATING_ANCHOR = '{"interestingness":"'

_RATING_SUFFIX = (
    'You see this image; rate its interestingness from:\n'
    '"Not Interesting","Slightly Interesting","Moderately Interesting",\n'
    '"Very Interesting","Extremely Interesting"\n\n'
    'Return ONLY valid JSON.\n'
    'Do not use markdown.\n'
    'Do not use code fences.\n'
    'Do not add any text before or after the JSON.\n\n'
    'Keys:\n'
    '- "interestingness": one of the five strings above\n'
    '- "explanation": exactly 1 short sentence'
)


def build_persona_prompt(persona_dict: dict | None) -> str:
    """
    Build the full interestingness-rating prompt for a given persona.

    Reserved keys (stripped from the persona description):
      persona_key — checkpoint filename identifier
      _format     — "raw"     : values only, comma-separated
                               (matches original get_activations.py style; use for synonym test)
                  — "labeled" : "key: value" pairs (default; use for new experiments)

    All other keys are included in persona description order (Python 3.7+ insertion order).
    Pass None or {} for the blank/baseline prompt (no persona description).
    """
    if not persona_dict:
        return f"You are rating images.\n\n{_RATING_SUFFIX}"

    fmt    = persona_dict.get("_format", "labeled")
    fields = {k: v for k, v in persona_dict.items()
              if k != "persona_key" and not k.startswith("_")}

    if fmt == "raw":
        persona_str = ", ".join(str(v) for v in fields.values())
    else:
        persona_str = ", ".join(f"{k}: {v}" for k, v in fields.items())

    return (
        f"Imagine you are a person with the following details:\n"
        f"{persona_str}\n\n"
        f"{_RATING_SUFFIX}"
    )


def build_blank_prompt() -> str:
    """Return the baseline prompt with no persona description."""
    return build_persona_prompt(persona_dict=None)


def extract_last_json_block(text: str) -> dict:
    """Extract and parse the last valid JSON object from a model response string."""
    for raw in reversed(re.findall(r"\{.*?\}", text, flags=re.S)):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON found in response")


def parse_model_response(raw_text: str) -> dict:
    """
    Parse the model's JSON output into a structured result dict.

    Returns dict with keys: interestingness, explanation.
    Raises ValueError if response cannot be parsed into a valid label.
    """
    data = extract_last_json_block(raw_text)
    label = data.get("interestingness", "")
    if label not in VALID_LABELS:
        raise ValueError(
            f"Invalid interestingness label: {label!r}. "
            f"Expected one of: {VALID_LABELS}"
        )
    return data
