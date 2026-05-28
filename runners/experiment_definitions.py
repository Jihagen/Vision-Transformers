"""
Experiment condition definitions for the gender × emotion activation study.

Two experimental variants are defined:

  "base"     — Gender + Emotion, with three confounders held constant
               (age range, employment status, mental workload).

  "extended" — Same as base, plus a country-of-origin specification.
               Country value is left as None here and must be set before
               launching collection runs.

For each variant there are 16 conditions: 2 genders × 8 emotions.

All collected result files for a variant live in the same directory so that
the reconcile step and analysis pipeline can discover them automatically.
"""

from __future__ import annotations
from pathlib import Path

# ── Experimental axes ─────────────────────────────────────────────────────────

EMOTIONS: list[str] = [
    "anger",
    "amusement",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sad",
]

GENDERS: list[str] = ["female", "male"]


# ── Persona bases ─────────────────────────────────────────────────────────────

# Base: three confounders fixed — matches the existing gender_emotion dataset.
_GE_BASE: dict = {
    "Age range":       "30–45",
    "Employment":      "full-time employed",
    "Mental workload": "moderate",
}

# Extended: base + country.
# Set Country to the desired value before generating conditions or launching runs.
_GE_EXTENDED: dict = {
    **_GE_BASE,
    "Country": None,   # TODO: set before running (e.g. "Germany")
}


# ── Condition builders ────────────────────────────────────────────────────────

def _persona_key(gender: str, emotion: str, variant: str) -> str:
    suffix = "" if variant == "base" else f"_{variant}"
    return f"{gender}_{emotion}{suffix}"


def _make_persona(gender: str, emotion: str, base: dict) -> dict:
    return {**base, "Gender": gender.capitalize(), "Emotion": emotion.capitalize()}


def get_all_conditions(variant: str = "base") -> dict[str, dict]:
    """
    Return {persona_key: persona_dict} for all 16 conditions of a variant.

    Raises ValueError if variant is "extended" and Country is still None.
    """
    if variant == "base":
        base = _GE_BASE
    elif variant == "extended":
        if _GE_EXTENDED.get("Country") is None:
            raise ValueError(
                "Country is not set in _GE_EXTENDED. "
                "Edit runners/experiment_definitions.py and set a Country value "
                "before launching extended collection runs."
            )
        base = _GE_EXTENDED
    else:
        raise ValueError(f"Unknown variant {variant!r}. Expected 'base' or 'extended'.")

    return {
        _persona_key(g, e, variant): _make_persona(g, e, base)
        for g in GENDERS
        for e in EMOTIONS
    }


def set_extended_country(country: str) -> None:
    """
    Set the Country field for the extended variant at runtime.
    Call this before get_all_conditions('extended').
    """
    _GE_EXTENDED["Country"] = country


# ── Result path conventions ───────────────────────────────────────────────────

_RESULTS_ROOT = Path("data/experiments")

_VARIANT_DIRS: dict[str, Path] = {
    "base":     _RESULTS_ROOT / "gender_emotion",
    "extended": _RESULTS_ROOT / "gender_emotion_extended",
}


def get_result_path(persona_key: str, variant: str = "base") -> Path:
    """Return the canonical .npy path for a given condition."""
    return _VARIANT_DIRS[variant] / f"results_{persona_key}.npy"


def get_all_result_paths(variant: str = "base") -> dict[str, Path]:
    """Return {persona_key: path} for all 16 conditions of a variant."""
    return {
        pk: get_result_path(pk, variant)
        for pk in _make_all_keys(variant)
    }


def _make_all_keys(variant: str) -> list[str]:
    return [_persona_key(g, e, variant) for g in GENDERS for e in EMOTIONS]


def get_missing_conditions(variant: str = "base") -> list[str]:
    """Return persona_keys whose result .npy does not yet exist."""
    return [
        pk for pk, path in get_all_result_paths(variant).items()
        if not path.exists()
    ]


# ── Contrast definitions ──────────────────────────────────────────────────────

def get_gender_contrasts(variant: str = "base") -> list[tuple[str, str]]:
    """
    All within-emotion gender contrasts: (female_X, male_X) for each emotion.
    Used as the primary test of gender concept direction.
    """
    return [
        (_persona_key("female", e, variant), _persona_key("male", e, variant))
        for e in EMOTIONS
    ]


def get_cross_variant_contrasts(emotion: str) -> list[tuple[str, str]]:
    """
    Compare base vs extended for the same gender × emotion pair.
    Useful for quantifying the effect of adding country specification.
    E.g.: (female_anger, female_anger_extended) for each gender.
    """
    return [
        (_persona_key(g, emotion, "base"), _persona_key(g, emotion, "extended"))
        for g in GENDERS
    ]


def get_manifest_path() -> Path:
    """Canonical manifest shared by all conditions."""
    return Path("data/selected_uniform_500_manifest.pkl")
