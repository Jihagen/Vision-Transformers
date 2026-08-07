"""
Runner: logit-lens lexicon derivation for the emotion attack vector actually
used by the excited_vs_angry UAP (avg_excitement_vs_avg_anger, language_29 —
the default --layer of run_universal_perturbation.py, unlike the workload UAP
which pins it explicitly). No auto-resolution needed here: unlike the
mental-workload contrast (5 layers tied at AUC=1.0), this vector was already
picked, trained, and attacked at a single specific layer — this script just
derives the lexicon needed for a proper dose-response run at that same layer.

CPU-only: loads just the final-norm + lm_head tensors (~2GB), no full model.

Usage
-----
    python -m runners.run_emotion_attack_logit_lens
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = (
    "/anvme/workspace/iwi5268h-vision-transformers/hpc_infrastructure/hf_cache/"
    "models--meta-llama--Llama-4-Scout-17B-16E-Instruct/local-repo"
)
_VECTOR_PATH = Path("results/representation_discovery/base_emotion/md_vectors/avg_excitement_vs_avg_anger.npy")
_LAYER       = "language_29_D5120"
_OUT_DIR     = Path("results/extra_checks/logit_lens")
_TOP_K       = 30


def main() -> None:
    import numpy as np
    from transformers import AutoTokenizer
    from control.III1_vector_control.logit_lens import (
        load_unembedding, project_to_vocab, top_tokens, derive_lexicon,
    )

    vec = np.load(_VECTOR_PATH, allow_pickle=True).item()[_LAYER]
    vector_norm = float(np.linalg.norm(vec))

    logger.info(f"Loading final norm + lm_head (~2GB, CPU) from {_DEFAULT_MODEL_PATH}")
    norm_w, lm_head = load_unembedding(_DEFAULT_MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(_DEFAULT_MODEL_PATH)

    deltas = project_to_vocab(vec, norm_w, lm_head)
    pos_tokens, neg_tokens = top_tokens(deltas, tok, k=_TOP_K)
    pos_words, neg_words = derive_lexicon(deltas, tok, k=200, max_words=30)

    layer_slug = _LAYER.replace("_D5120", "").replace("_D1408", "")
    out = {
        "name": f"excited_vs_angry_{layer_slug}",
        "description": (
            f"Emotion attack direction (excitement - anger), gender-averaged, "
            f"@ {_LAYER} — the same vector and layer used by the excited_vs_angry UAP "
            f"(run_universal_perturbation_projected.slurm)."
        ),
        "source": str(_VECTOR_PATH),
        "vector_norm": vector_norm,
        "top_positive_tokens": [{"token": t, "delta_logit": s} for t, s in pos_tokens],
        "top_negative_tokens": [{"token": t, "delta_logit": s} for t, s in neg_tokens],
        "lexicon": {"positive_words": pos_words, "negative_words": neg_words},
    }

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"excited_vs_angry_{layer_slug}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved → {out_path}")
    print(f"\n✅ excited_vs_angry logit-lens lexicon complete -> {out_path}")


if __name__ == "__main__":
    main()
