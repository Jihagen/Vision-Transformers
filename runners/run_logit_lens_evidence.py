"""
Persist logit-lens evidence for the discovered direction vectors.

For each vector of interest, projects the (unit-normalised) direction through
the model's final RMSNorm + LM head to obtain a one-step linear approximation
of "which vocabulary tokens this direction pushes the model towards / away
from" (logit-lens, nostalgebraist 2020). This is used as qualitative/semantic
evidence that a direction encodes a meaningful concept rather than an
arbitrary axis, complementing the AUC / probe-accuracy / cosine-subspace
checks from Tier I.2-I.5.

Outputs (results/extra_checks/logit_lens/):
    gender_language_29.json
    interest_blank_language_29.json
    interest_global_language_33.json
    summary.md   -- human-readable top-token tables for all three

Usage:
    python -m runners.run_logit_lens_evidence
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from control.III1_vector_control.control import (
    load_averaged_country_vector,
    load_averaged_gender_vector,
)
from control.III1_vector_control.logit_lens import (
    derive_lexicon,
    load_unembedding,
    project_to_vocab,
    top_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "hpc_infrastructure/hf_cache/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/local-repo"
GENDER_MD_VECTORS_DIR = "results/representation_discovery/base/md_vectors"
INTEREST_MD_VECTORS_DIR = "results/representation_discovery/interestingness/md_vectors"
COUNTRY_MD_VECTORS_DIRS = {
    "germany": "results/representation_discovery/extended_germany_country/md_vectors",
    "nigeria": "results/representation_discovery/extended_nigeria_country/md_vectors",
}
OUT_DIR = Path("results/extra_checks/logit_lens")
TOP_K = 30


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading direction vectors...")
    gender_vec = load_averaged_gender_vector(GENDER_MD_VECTORS_DIR, layer_key="language_29_D5120")

    interest_blank = np.load(
        Path(INTEREST_MD_VECTORS_DIR) / "blank_interest_high_vs_blank_interest_low.npy",
        allow_pickle=True,
    ).item()["language_29_D5120"]

    interest_global = np.load(
        Path(INTEREST_MD_VECTORS_DIR) / "global_interest_high_vs_global_interest_low.npy",
        allow_pickle=True,
    ).item()["language_33_D5120"]

    country_germany = load_averaged_country_vector(
        COUNTRY_MD_VECTORS_DIRS["germany"], layer_key="language_27_D5120"
    )
    country_nigeria = load_averaged_country_vector(
        COUNTRY_MD_VECTORS_DIRS["nigeria"], layer_key="language_23_D5120"
    )

    logger.info(f"Loading final norm + lm_head from {MODEL_PATH} (~2GB, CPU)...")
    norm_w, lm_head = load_unembedding(MODEL_PATH)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    targets = {
        "gender_language_29": {
            "vector": gender_vec,
            "description": (
                "Averaged female-minus-male gender direction (mean over 8 emotion "
                "contrasts in results/representation_discovery/base), language_29_D5120. "
                "Positive direction = towards 'female'."
            ),
            "source": f"{GENDER_MD_VECTORS_DIR} (averaged across emotions) @ language_29_D5120",
        },
        "interest_blank_language_29": {
            "vector": interest_blank,
            "description": (
                "Mean-difference 'high interest minus low interest' direction from the "
                "blank-prompt (no persona) discovery run, language_29_D5120 "
                "(within the AUC=1.0 plateau, peak md_cav_cosine). "
                "Positive direction = towards 'high interest'."
            ),
            "source": f"{INTEREST_MD_VECTORS_DIR}/blank_interest_high_vs_blank_interest_low.npy @ language_29_D5120",
        },
        "interest_global_language_33": {
            "vector": interest_global,
            "description": (
                "Mean-difference 'high interest minus low interest' direction from the "
                "global (cross-persona aggregated) discovery run, language_33_D5120 "
                "(within the AUC=1.0 plateau, peak md_cav_cosine). "
                "Positive direction = towards 'high interest'."
            ),
            "source": f"{INTEREST_MD_VECTORS_DIR}/global_interest_high_vs_global_interest_low.npy @ language_33_D5120",
        },
        "country_germany_language_27": {
            "vector": country_germany,
            "description": (
                "Averaged 'extended_germany minus base' direction (mean over 16 "
                "gender x emotion conditions), language_27_D5120 (most frequent "
                "AUC-peak layer for the Germany country contrasts). "
                "Positive direction = towards 'extended_germany' framing."
            ),
            "source": f"{COUNTRY_MD_VECTORS_DIRS['germany']} (averaged across 16 conditions) @ language_27_D5120",
        },
        "country_nigeria_language_23": {
            "vector": country_nigeria,
            "description": (
                "Averaged 'extended_nigeria minus base' direction (mean over 16 "
                "gender x emotion conditions), language_23_D5120 (AUC-peak layer "
                "for 15/16 of the Nigeria country contrasts). "
                "Positive direction = towards 'extended_nigeria' framing."
            ),
            "source": f"{COUNTRY_MD_VECTORS_DIRS['nigeria']} (averaged across 16 conditions) @ language_23_D5120",
        },
    }

    summary_lines = ["# Logit-lens evidence\n"]
    summary_lines.append(
        "One-step linear approximation (nostalgebraist 2020): "
        "`logits = lm_head @ (final_rmsnorm_weight * unit_direction)`. "
        f"Top {TOP_K} tokens by |delta logit| in each direction.\n"
    )

    for name, spec in targets.items():
        v = spec["vector"]
        deltas = project_to_vocab(v, norm_w, lm_head)
        pos, neg = top_tokens(deltas, tok, k=TOP_K)

        pos_words, neg_words = derive_lexicon(deltas, tok)

        out = {
            "name": name,
            "description": spec["description"],
            "source": spec["source"],
            "vector_norm": float(np.linalg.norm(v)),
            "top_positive_tokens": [{"token": t, "delta_logit": s} for t, s in pos],
            "top_negative_tokens": [{"token": t, "delta_logit": s} for t, s in neg],
            "lexicon": {
                "positive_words": pos_words,
                "negative_words": neg_words,
            },
        }
        out_path = OUT_DIR / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote {out_path}")

        summary_lines.append(f"\n## {name}\n")
        summary_lines.append(f"{spec['description']}\n")
        summary_lines.append(f"Source: `{spec['source']}`\n")
        summary_lines.append("\n**Top positive-direction tokens:**\n")
        summary_lines.append("\n".join(f"- `{t!r}`: {s:+.3f}" for t, s in pos))
        summary_lines.append("\n\n**Top negative-direction tokens:**\n")
        summary_lines.append("\n".join(f"- `{t!r}`: {s:+.3f}" for t, s in neg))
        summary_lines.append(
            f"\n\n**Derived lexicon** (positive_words, n={len(pos_words)}): "
            + ", ".join(pos_words)
        )
        summary_lines.append(
            f"\n\n**Derived lexicon** (negative_words, n={len(neg_words)}): "
            + ", ".join(neg_words)
        )
        summary_lines.append("\n")

    summary_path = OUT_DIR / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    logger.info(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
