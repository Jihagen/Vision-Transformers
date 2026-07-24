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

# Mental-workload dose-response poles (blank prompt + this one attribute only —
# NOT crossed with gender/emotion). "moderate" is deliberately excluded: every
# existing gender_emotion condition already fixes "Mental workload": "moderate"
# as a confound (_GE_BASE below), so that pole's activations already exist.
# Avoid the words "extreme"/"extremely" — they already appear in the
# interestingness rating lexicon (results/extra_checks/logit_lens/
# interest_global_language_33.json) and as literal rating-bucket names
# (results/metrics/extreme_positive, extreme_negative), which would confound
# any lexicon-count-based measurement of this vector.
WORKLOAD_LEVELS: list[str] = ["minimal", "overwhelming"]


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


def _workload_persona_key(level: str) -> str:
    return f"workload_{level}"


def get_workload_conditions() -> dict[str, dict]:
    """
    Return {persona_key: persona_dict} for the mental-workload dose-response
    study: the same Age range / Employment confound values used by every
    other condition (_GE_BASE), with 'Mental workload' overridden to each
    pole in WORKLOAD_LEVELS instead of the usual fixed 'moderate'.

    Gender/Emotion are deliberately left out (not crossed) — kept minimal
    since gender/country were both found orthogonal to interestingness with
    no measurable effect, so there's no reason to pay for that cross product
    here. Age range/Employment are kept, though, so 'constant persona' means
    the same thing it means for every other contrast in this study (all of
    _GE_BASE held fixed, one field varied) rather than a persona shape with
    no anchor at all.
    """
    anchor = {k: v for k, v in _GE_BASE.items() if k != "Mental workload"}
    return {
        _workload_persona_key(level): {**anchor, "Mental workload": level}
        for level in WORKLOAD_LEVELS
    }


def get_workload_contrast() -> tuple[str, str]:
    """(overwhelming, minimal) — positive direction = increasing stress/workload."""
    return (
        _workload_persona_key("overwhelming"),
        _workload_persona_key("minimal"),
    )


def get_all_conditions(variant: str = "base") -> dict[str, dict]:
    """
    Return {persona_key: persona_dict} for all conditions of a variant.

    Accepts:
        "base"                      — base conditions (no country)
        "extended"                  — extended with current _GE_EXTENDED["Country"]
        "extended_<country_lower>"  — auto-sets Country from the variant key
        "workload"                  — mental-workload dose-response poles
    """
    if variant == "workload":
        return get_workload_conditions()
    if variant == "base":
        base = _GE_BASE
    elif variant == "extended":
        if _GE_EXTENDED.get("Country") is None:
            raise ValueError(
                "Country is not set. Call set_extended_country() first."
            )
        base = _GE_EXTENDED
    elif variant.startswith("extended_"):
        # e.g. "extended_germany" → Country = "Germany"
        country = variant.replace("extended_", "").capitalize()
        set_extended_country(country)
        register_country_variant(country)
        base = _GE_EXTENDED
    else:
        raise ValueError(f"Unknown variant {variant!r}.")

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
    "workload": _RESULTS_ROOT / "mental_workload",
}

# Countries supported for extended country variants
COUNTRIES: list[str] = ["Germany", "Nigeria"]

def _country_variant_key(country: str) -> str:
    return f"extended_{country.lower()}"

def register_country_variant(country: str) -> None:
    """Register a country-specific extended variant directory (called at runtime)."""
    key = _country_variant_key(country)
    if key not in _VARIANT_DIRS:
        _VARIANT_DIRS[key] = _RESULTS_ROOT / f"gender_emotion_{country.lower()}"


def get_result_path(persona_key: str, variant: str = "base") -> Path:
    """Return the canonical .npy path for a given condition."""
    if variant not in _VARIANT_DIRS:
        raise KeyError(f"Unknown variant '{variant}'. Registered: {list(_VARIANT_DIRS)}")
    return _VARIANT_DIRS[variant] / f"results_{persona_key}.npy"


def get_all_result_paths(variant: str = "base") -> dict[str, Path]:
    """Return {persona_key: path} for all 16 conditions of a variant."""
    return {
        pk: get_result_path(pk, variant)
        for pk in _make_all_keys(variant)
    }


def _make_all_keys(variant: str) -> list[str]:
    if variant == "workload":
        return [_workload_persona_key(level) for level in WORKLOAD_LEVELS]
    return [_persona_key(g, e, variant) for g in GENDERS for e in EMOTIONS]


def get_missing_conditions(variant: str = "base") -> list[str]:
    """Return persona_keys whose result .npy is missing, empty, or short of the manifest size."""
    import numpy as np
    import pandas as pd
    n_target = len(pd.read_pickle(get_manifest_path()))
    missing = []
    for pk, path in get_all_result_paths(variant).items():
        if not path.exists():
            missing.append(pk)
            continue
        try:
            obj = np.load(path, allow_pickle=True).item()
            if len(obj.get("results", [])) < n_target:
                missing.append(pk)
        except Exception:
            missing.append(pk)
    return missing


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


# ── Emotion contrast definitions ──────────────────────────────────────────────

def get_emotion_vs_rest_contrasts() -> list[tuple[str, str]]:
    """
    Gender-averaged, anchor-free emotion contrasts: each emotion vs. the pooled
    mean of all other emotions ("one-vs-rest").

    This replaces an earlier anchor-based design that contrasted every emotion
    against a single privileged "neutral" emotion (contentment). That assumption
    does not survive contact with the geometry — contentment sits inside the
    positive-emotion cluster (next to amusement), not at a neutral midpoint — so
    "X vs contentment" was actually measuring "distance from one specific positive
    emotion", which is systematically weaker for emotions that neighbour it and
    systematically stronger for emotions on the opposite valence pole. See the
    persona-analytics notebook (Ch. 7, pre-rewrite) for the full critique.

    One-vs-rest needs no anchor at all: every emotion is measured against the
    same kind of reference — the pooled average of the *other seven* — so no
    single emotion is privileged and the design is symmetric by construction.
    "The layer where emotion is most clearly represented" is then simply the
    layer that maximises mean one-vs-rest separability across all 8 emotions.

    Each entry is (emotion_A, f"rest_{emotion_A}"). Callers build gender-averaged
    pseudo-conditions `avg_{emotion}` (merge female_X + male_X) AND
    `avg_rest_{emotion}` (merge the avg_* pseudo-conditions of the other seven
    emotions) — see merge_conditions() in representation/I1_contrast_design/load.py.

    Returns:
        List of (emotion_name, f"rest_{emotion_name}") string pairs — NOT persona_keys.
    """
    return [(e, f"rest_{e}") for e in EMOTIONS]


def get_gender_averaged_keys(emotion: str, variant: str = "base") -> list[str]:
    """Return [female_emotion, male_emotion] persona_keys for merging."""
    return [_persona_key(g, emotion, variant) for g in GENDERS]


# ── Country contrast definitions ──────────────────────────────────────────────

def get_country_gender_contrasts(country: str) -> list[tuple[str, str]]:
    """
    Gender contrasts for a specific country variant, mirroring get_gender_contrasts
    but using the country-specific variant key.
    """
    variant = _country_variant_key(country)
    register_country_variant(country)
    return get_gender_contrasts(variant)


def get_country_vs_base_contrasts(country: str) -> list[tuple[str, str]]:
    """
    For each gender × emotion, contrast base vs country variant.
    Used to derive the country concept vector:
      v_country = mean(country_conditions) - mean(base_conditions)
    """
    country_variant = _country_variant_key(country)
    register_country_variant(country)
    pairs = []
    for g in GENDERS:
        for e in EMOTIONS:
            base_key    = _persona_key(g, e, "base")
            country_key = _persona_key(g, e, country_variant)
            pairs.append((country_key, base_key))
    return pairs


# ── All variants helper ───────────────────────────────────────────────────────

def get_registered_variants() -> list[str]:
    """Return all currently registered variant keys."""
    return list(_VARIANT_DIRS.keys())
