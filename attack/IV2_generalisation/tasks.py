"""
Task adapters for Generalisation Level 1.

Each task defines: exact prompt text, valid label set, scale width (for
normalized signed shift), and a strict-JSON-first / regex-fallback parser.
Dataset loading lives in datasets.py; this module is pure prompt/parse logic
so it can be unit-tested without touching HF hub or the model.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable

from utils.prompt_builder import extract_last_json_block

# ── Prompts (verbatim from the implementation brief) ──────────────────────

SHOPPING_PROMPT_TEMPLATE = """A user is shopping for:

"{query}"

Look at the product shown in the image.

How relevant is this product to what the user is looking for?

Rate its relevance using exactly this scale:

1 = Not relevant
2 = Slightly relevant
3 = Moderately relevant
4 = Highly relevant
5 = Extremely/directly relevant

Return only valid JSON in the following format:

{{
  "label": 1,
  "explanation": "Briefly explain why the product is or is not relevant."
}}

The label must be an integer from 1 to 5."""

MORAL_PROMPT = """Look at the image and judge the morality of what it portrays.

Use exactly this scale:

1 = Immoral / blameworthy
2 = Mostly immoral
3 = Morally neutral, ambiguous, or mixed
4 = Mostly moral
5 = Moral / praiseworthy

Judge only what is visibly portrayed in the image.

Return only valid JSON:

{
  "label": 1,
  "explanation": "Briefly explain the moral evaluation."
}

The label must be an integer from 1 to 5."""

DAMAGE_PROMPT = """Look at the image and assess the severity of the visible physical damage.

Use exactly these categories:

0 = Little or no visible damage
1 = Mild/moderate visible damage
2 = Severe visible damage

Judge the visible damage itself, not how dramatic or emotionally upsetting the image feels.

Return only valid JSON:

{
  "label": 0,
  "explanation": "Briefly explain the visible evidence for this severity judgment."
}

The label must be one of: 0, 1, 2."""


def build_shopping_prompt(row) -> str:
    return SHOPPING_PROMPT_TEMPLATE.format(query=row["query"])


def build_moral_prompt(row) -> str:
    return MORAL_PROMPT


def build_damage_prompt(row) -> str:
    return DAMAGE_PROMPT


# ── Parsing ──────────────────────────────────────────────────────────────

_FALLBACK_LABEL_RE = re.compile(r'"label"\s*:\s*(-?\d+)')


def parse_task_response(raw_text: str, valid_labels: list[int]) -> dict:
    """
    Strict JSON first (must contain a "label" key parseable as int and
    in-range); conservative regex fallback ONLY extracts a value the model
    already explicitly wrote as "label": N (e.g. JSON truncated by
    max_new_tokens before the explanation closed) — never guesses a number
    from free text elsewhere in the response.

    Returns dict: label (int|None), explanation (str|None), raw_output (str),
    parse_ok (bool), used_fallback (bool).
    """
    try:
        data = extract_last_json_block(raw_text)
        label = int(data["label"])
        if label not in valid_labels:
            raise ValueError(f"label {label} not in {valid_labels}")
        explanation = str(data.get("explanation", ""))
        return {
            "label": label, "explanation": explanation,
            "raw_output": raw_text, "parse_ok": True, "used_fallback": False,
        }
    except Exception:
        pass

    m = _FALLBACK_LABEL_RE.search(raw_text)
    if m:
        try:
            label = int(m.group(1))
            if label in valid_labels:
                return {
                    "label": label, "explanation": None,
                    "raw_output": raw_text, "parse_ok": True, "used_fallback": True,
                }
        except ValueError:
            pass

    return {
        "label": None, "explanation": None,
        "raw_output": raw_text, "parse_ok": False, "used_fallback": False,
    }


@dataclass
class TaskSpec:
    name: str
    dataset: str
    valid_labels: list[int]
    scale_width: float           # normalisation divisor for signed shift
    prompt_builder: Callable[[object], str]
    extra_manifest_cols: list[str]   # dataset-specific columns to carry through predictions.csv


TASKS: dict[str, TaskSpec] = {
    "shopping_relevance": TaskSpec(
        name="shopping_relevance",
        dataset="marqo_gs10m",
        valid_labels=[1, 2, 3, 4, 5],
        scale_width=4.0,
        prompt_builder=build_shopping_prompt,
        extra_manifest_cols=["query", "title", "product_id", "position",
                              "score_linear", "score_reciprocal", "reference_relevance_5"],
    ),
    "moral_evaluation": TaskSpec(
        name="moral_evaluation",
        dataset="smid",
        valid_labels=[1, 2, 3, 4, 5],
        scale_width=4.0,
        prompt_builder=build_moral_prompt,
        extra_manifest_cols=["human_morality", "human_arousal", "human_valence"],
    ),
    "damage_severity": TaskSpec(
        name="damage_severity",
        dataset="medic",
        valid_labels=[0, 1, 2],
        scale_width=2.0,
        prompt_builder=build_damage_prompt,
        extra_manifest_cols=["gt_damage_severity", "informative", "humanitarian",
                              "disaster_type", "event_name"],
    ),
}
