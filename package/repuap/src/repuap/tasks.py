"""
JudgmentTask: the one abstraction shared across all four pipeline stages.

Every stage — activation collection (discover.py), causal injection
(causal.py), UAP training/eval (uap.py), and out-of-domain generalisation
(generalise.py) — needs the same three things: a way to build a prompt, a
way to parse the model's strict-JSON judgment out of its response, and a way
to turn a label into a numeric score. Defining these once here is what lets
one task definition flow through the whole pipeline instead of being
redefined (and silently drifting) at each stage.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

_JSON_BLOCK_RE = re.compile(r"\{.*?\}", flags=re.S)


def extract_last_json_block(text: str) -> dict:
    """Extract and parse the last valid JSON object from a model response string."""
    for raw in reversed(_JSON_BLOCK_RE.findall(text)):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON found in response")


@dataclass
class JudgmentTask:
    """
    Defines one judgment the model is being asked to make about an image.

    valid_labels:   the closed set of acceptable label values, e.g. [1,2,3,4,5],
                     [0,1,2], or ["yes","no"].
    scale_width:    divisor used to normalise signed-shift metrics (e.g. 4 for
                     a 1-5 scale, 2 for a 0-2 scale, 1 for a binary yes/no scale).
    prompt_builder: ctx -> prompt string. ctx is whatever per-sample context
                     you need (e.g. a dict/row with a "query" field) — pass
                     None if the prompt doesn't vary per sample.
    json_key:       the JSON key the model is asked to emit the label under.
                     Default "label"; set to match your own prompt's schema
                     (e.g. "interestingness", "relevant").
    label_type:     python type used to cast the parsed JSON value (int, float,
                     or str).
    label_to_score: optional {label: float} map. Only needed when label_type
                     is str (numeric labels are already their own score) or
                     when you want a non-identity score mapping.
    json_anchor:    the literal text used to teacher-force generation up to
                     (but not including) the label value, for activation
                     capture at the label-token position. Auto-derived from
                     json_key/label_type if not given — override only if your
                     prompt's JSON key ordering/formatting is unusual.
    """

    valid_labels: list[Any]
    scale_width: float
    prompt_builder: Callable[[Any], str]
    json_key: str = "label"
    label_type: type = int
    label_to_score: dict[Any, float] | None = None
    json_anchor: str | None = None

    def anchor(self) -> str:
        if self.json_anchor is not None:
            return self.json_anchor
        quote = '"' if self.label_type is str else ""
        return f'{{"{self.json_key}":{quote}'

    def score(self, label: Any) -> float:
        if self.label_to_score is not None:
            return float(self.label_to_score[label])
        if isinstance(label, (int, float)):
            return float(label)
        raise ValueError(
            f"Label {label!r} is not numeric and no label_to_score map was given for this task."
        )


@dataclass
class ParsedResponse:
    label: Any | None
    explanation: str | None
    raw_output: str
    parse_ok: bool
    used_fallback: bool


_FALLBACK_VALUE_RE_TEMPLATE = r'"{key}"\s*:\s*"?([^",}}]+)"?'


def parse_response(raw_text: str, task: JudgmentTask) -> ParsedResponse:
    """
    Strict JSON first (must contain task.json_key, castable to task.label_type,
    and in task.valid_labels). Conservative regex fallback ONLY extracts a
    value the model already explicitly wrote under task.json_key (e.g. JSON
    truncated by max_new_tokens before the explanation closed) — never
    guesses a label from free text elsewhere in the response.
    """
    try:
        data = extract_last_json_block(raw_text)
        raw_label = data[task.json_key]
        label = task.label_type(raw_label)
        if label not in task.valid_labels:
            raise ValueError(f"label {label!r} not in {task.valid_labels}")
        explanation = str(data.get("explanation", ""))
        return ParsedResponse(label, explanation, raw_text, True, False)
    except Exception:
        pass

    pattern = _FALLBACK_VALUE_RE_TEMPLATE.format(key=re.escape(task.json_key))
    m = re.search(pattern, raw_text)
    if m:
        try:
            label = task.label_type(m.group(1).strip())
            if label in task.valid_labels:
                return ParsedResponse(label, None, raw_text, True, True)
        except (ValueError, TypeError):
            pass

    return ParsedResponse(None, None, raw_text, False, False)
