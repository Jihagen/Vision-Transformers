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

    Accepts:
        "base"                      — base conditions (no country)
        "extended"                  — extended with current _GE_EXTENDED["Country"]
        "extended_<country_lower>"  — auto-sets Country from the variant key
    """
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
    return [_persona_key(g, e, variant) for g in GENDERS for e in EMOTIONS]


def get_missing_conditions(variant: str = "base") -> list[str]:
    """Return persona_keys whose result .npy does not exist or has no results."""
    import numpy as np
    missing = []
    for pk, path in get_all_result_paths(variant).items():
        if not path.exists():
            missing.append(pk)
            continue
        try:
            obj = np.load(path, allow_pickle=True).item()
            if len(obj.get("results", [])) == 0:
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

def get_emotion_contrast_pairs(anchor: str = "contentment") -> list[tuple[str, str]]:
    """
    Gender-averaged emotion contrasts: each emotion vs the anchor emotion.

    Each entry is (emotion_A, anchor_emotion). Callers should load both female
    and male variants of each side and merge them before computing vectors —
    see merge_conditions() in representation/I1_contrast_design/load.py.

    Returns:
        List of (emotion_name, anchor_name) string pairs — NOT persona_keys.
        Use get_gender_averaged_emotion_keys() to get the actual persona_key lists.
    """
    return [(e, anchor) for e in EMOTIONS if e != anchor]


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
