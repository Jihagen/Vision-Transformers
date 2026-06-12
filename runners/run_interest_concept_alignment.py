"""
Option 2: persona-specific concept entanglement of the interestingness direction.

For each of the 7 per-condition "high interest minus low interest" vectors
(results/representation_discovery/interestingness/md_vectors/), computes
cosine similarity against:
  - gender direction  (results/representation_discovery/base/md_vectors,
                        per-emotion female_vs_male)
  - emotion direction (results/representation_discovery/base_emotion/md_vectors,
                        one-vs-rest avg_<emotion>_vs_avg_rest_<emotion>)
  - country direction (results/representation_discovery/extended_{germany,nigeria}_country/md_vectors,
                        <condition>_extended_<country>_vs_<condition>)

For each concept type, "matched" = the concept vector belonging to that same
persona (same emotion / same country); "mismatched" = the mean over the
concept vectors of the *other* personas/conditions of the same type. A
positive matched-mismatched gap means the persona's interest direction is
specifically entangled with ITS OWN gender/emotion/country direction, not
just generically aligned with concept vectors in general.

v_interest_blank (no persona) is reported separately, against the
across-emotion / across-country average, as the "no persona to be entangled
with" reference point.

Output: results/representation_discovery/interestingness/analytics/
    concept_alignment.csv   -- per-layer, per-persona matched/mismatched cosines
    concept_alignment_blank.csv -- per-layer blank reference cosines
    summary.md
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("results/representation_discovery/base/md_vectors")
EMO  = Path("results/representation_discovery/base_emotion/md_vectors")
DE   = Path("results/representation_discovery/extended_germany_country/md_vectors")
NG   = Path("results/representation_discovery/extended_nigeria_country/md_vectors")
INT  = Path("results/representation_discovery/interestingness/md_vectors")
OUT_DIR = Path("results/representation_discovery/interestingness/analytics")

EMOTIONS = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sad"]
LAYERS = [f"language_{i}_D5120" for i in range(20, 48)]

# persona -> (interest contrast file, emotion key, country md_vectors dir, country contrast key)
PERSONAS = {
    "male_excitement_base": (
        "male_excitement_interest_high_vs_male_excitement_interest_low",
        "excitement", None, None,
    ),
    "female_excitement_germany": (
        "female_excitement_extended_germany_interest_high_vs_female_excitement_extended_germany_interest_low",
        "excitement", "DE", "female_excitement_extended_germany_vs_female_excitement",
    ),
    "male_awe_germany": (
        "male_awe_extended_germany_interest_high_vs_male_awe_extended_germany_interest_low",
        "awe", "DE", "male_awe_extended_germany_vs_male_awe",
    ),
    "male_excitement_germany": (
        "male_excitement_extended_germany_interest_high_vs_male_excitement_extended_germany_interest_low",
        "excitement", "DE", "male_excitement_extended_germany_vs_male_excitement",
    ),
    "female_excitement_nigeria": (
        "female_excitement_extended_nigeria_interest_high_vs_female_excitement_extended_nigeria_interest_low",
        "excitement", "NG", "female_excitement_extended_nigeria_vs_female_excitement",
    ),
    "male_awe_nigeria": (
        "male_awe_extended_nigeria_interest_high_vs_male_awe_extended_nigeria_interest_low",
        "awe", "NG", "male_awe_extended_nigeria_vs_male_awe",
    ),
    "male_excitement_nigeria": (
        "male_excitement_extended_nigeria_interest_high_vs_male_excitement_extended_nigeria_interest_low",
        "excitement", "NG", "male_excitement_extended_nigeria_vs_male_excitement",
    ),
}


def _load(p: Path) -> dict:
    return np.load(p, allow_pickle=True).item()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gender_vecs = {e: _load(BASE / f"female_{e}_vs_male_{e}.npy") for e in EMOTIONS}
    emotion_vecs = {e: _load(EMO / f"avg_{e}_vs_avg_rest_{e}.npy") for e in EMOTIONS}
    de_country_vecs = {f.stem: _load(f) for f in DE.glob("*.npy")}
    ng_country_vecs = {f.stem: _load(f) for f in NG.glob("*.npy")}
    country_dirs = {"DE": de_country_vecs, "NG": ng_country_vecs}
    blank_vec = _load(INT / "blank_interest_high_vs_blank_interest_low.npy")
    all_country = list(de_country_vecs.values()) + list(ng_country_vecs.values())

    rows, blank_rows = [], []
    for layer in LAYERS:
        for persona, (contrast_file, emo_key, country_set, country_key) in PERSONAS.items():
            v_int = _load(INT / f"{contrast_file}.npy")[layer]

            cos_gender_matched = abs(_cos(v_int, gender_vecs[emo_key][layer]))
            cos_gender_mismatched = float(np.mean(
                [abs(_cos(v_int, gender_vecs[e][layer])) for e in EMOTIONS if e != emo_key]
            ))
            cos_emo_matched = abs(_cos(v_int, emotion_vecs[emo_key][layer]))
            cos_emo_mismatched = float(np.mean(
                [abs(_cos(v_int, emotion_vecs[e][layer])) for e in EMOTIONS if e != emo_key]
            ))

            row = {
                "layer": layer, "persona": persona,
                "cos_gender_matched": cos_gender_matched, "cos_gender_mismatched": cos_gender_mismatched,
                "cos_emotion_matched": cos_emo_matched, "cos_emotion_mismatched": cos_emo_mismatched,
            }

            if country_set is not None:
                cvecs = country_dirs[country_set]
                cos_country_matched = abs(_cos(v_int, cvecs[country_key][layer]))
                mismatched_keys = [k for k in cvecs if k != country_key]
                cos_country_mismatched = float(np.mean(
                    [abs(_cos(v_int, cvecs[k][layer])) for k in mismatched_keys]
                ))
                row["cos_country_matched"] = cos_country_matched
                row["cos_country_mismatched"] = cos_country_mismatched
            else:
                row["cos_country_matched"] = np.nan
                row["cos_country_mismatched"] = np.nan

            rows.append(row)

        v_blank = blank_vec[layer]
        blank_rows.append({
            "layer": layer,
            "cos_gender_avg": float(np.mean([abs(_cos(v_blank, gender_vecs[e][layer])) for e in EMOTIONS])),
            "cos_emotion_avg": float(np.mean([abs(_cos(v_blank, emotion_vecs[e][layer])) for e in EMOTIONS])),
            "cos_country_avg": float(np.mean([abs(_cos(v_blank, d[layer])) for d in all_country])),
        })

    df = pd.DataFrame(rows)
    blank_df = pd.DataFrame(blank_rows)

    df.to_csv(OUT_DIR / "concept_alignment.csv", index=False)
    blank_df.to_csv(OUT_DIR / "concept_alignment_blank.csv", index=False)
    logger.info(f"Wrote {OUT_DIR / 'concept_alignment.csv'} ({len(df)} rows)")
    logger.info(f"Wrote {OUT_DIR / 'concept_alignment_blank.csv'} ({len(blank_df)} rows)")

    # Summary: matched-mismatched gap per concept, per layer, averaged over personas
    summary_rows = []
    for layer in LAYERS:
        sub = df[df.layer == layer]
        summary_rows.append({
            "layer": layer,
            "gender_matched": sub.cos_gender_matched.mean(),
            "gender_mismatched": sub.cos_gender_mismatched.mean(),
            "gender_gap": (sub.cos_gender_matched - sub.cos_gender_mismatched).mean(),
            "emotion_matched": sub.cos_emotion_matched.mean(),
            "emotion_mismatched": sub.cos_emotion_mismatched.mean(),
            "emotion_gap": (sub.cos_emotion_matched - sub.cos_emotion_mismatched).mean(),
            "country_matched": sub.cos_country_matched.mean(),
            "country_mismatched": sub.cos_country_mismatched.mean(),
            "country_gap": (sub.cos_country_matched - sub.cos_country_mismatched).mean(),
            "blank_cos_gender": blank_df.loc[blank_df.layer == layer, "cos_gender_avg"].iloc[0],
            "blank_cos_emotion": blank_df.loc[blank_df.layer == layer, "cos_emotion_avg"].iloc[0],
            "blank_cos_country": blank_df.loc[blank_df.layer == layer, "cos_country_avg"].iloc[0],
        })
    summary_df = pd.DataFrame(summary_rows).round(3)
    summary_df.to_csv(OUT_DIR / "concept_alignment_summary.csv", index=False)
    logger.info(f"Wrote {OUT_DIR / 'concept_alignment_summary.csv'}")

    with open(OUT_DIR / "summary.md", "w") as f:
        f.write("# Interestingness <-> persona-concept entanglement\n\n")
        f.write(
            "Mean |cosine| between each persona's `v_interest` and gender/emotion/"
            "country direction vectors, matched (own persona) vs mismatched "
            "(other personas, same concept type). v_interest_blank shown for "
            "reference (no persona to match against).\n\n"
        )
        f.write(summary_df.to_csv(index=False, sep="|"))
        f.write("\n")
    logger.info(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
