"""
Runner: logit-lens lexicon derivation for the mental-workload concept vector
(overwhelming - minimal, blank prompt).

Unlike run_logit_lens_evidence.py (whose target layers are hand-picked from a
one-time manual read of the I.4 evaluation results), this script auto-resolves
the best layer from results/representation_discovery/mental_workload/summary.json
(top entry of best_5_by_auc for the workload contrast) by default, so it can
run unattended as a dependency-chained step right after representation
discovery. Pass --layer explicitly to pin a specific layer instead — useful
when several layers tie at AUC=1.0 (as they do here) and you want the same
layer used by other concept vectors (e.g. language_29, used for gender/
interest) for direct comparability, or a later layer on the theory that fewer
remaining transformer layers means less opportunity for an injected
perturbation to be transformed away before it reaches the output.

CPU-only: loads just the final-norm + lm_head tensors (~2GB), no full model.

Usage
-----
    python -m runners.run_workload_logit_lens
    python -m runners.run_workload_logit_lens --layer language_29_D5120
"""
from __future__ import annotations
import argparse
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
_DISCOVERY_DIR   = Path("results/representation_discovery/mental_workload")
_MD_VECTORS_DIR  = _DISCOVERY_DIR / "md_vectors"
_SUMMARY_PATH    = _DISCOVERY_DIR / "summary.json"
_CONTRAST_NAME   = "workload_overwhelming_vs_workload_minimal"
_OUT_DIR         = Path("results/extra_checks/logit_lens")
_TOP_K           = 30


def _resolve_layer(explicit_layer: str | None) -> tuple[str, dict]:
    """
    Resolve the target layer + its evaluation stats.

    If explicit_layer is None, auto-pick the highest-MD-AUC layer. Otherwise,
    look up the requested layer's stats (from best_5_by_auc if it's one of
    the top 5, else from the full per-layer evaluation CSV) so the saved
    JSON's description is always backed by real numbers, not just asserted.
    """
    if not _SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{_SUMMARY_PATH} not found — run representation discovery first: "
            f"python runners/run_representation_discovery.py --mode workload"
        )
    with open(_SUMMARY_PATH) as f:
        summary = json.load(f)
    info = summary["per_contrast"].get(_CONTRAST_NAME)
    if not info or not info.get("best_5_by_auc"):
        raise ValueError(
            f"No evaluated layers found for contrast '{_CONTRAST_NAME}' in {_SUMMARY_PATH}"
        )

    if explicit_layer is None:
        best = info["best_5_by_auc"][0]
        logger.info(
            f"Auto-resolved layer {best['layer']} for '{_CONTRAST_NAME}' "
            f"(AUC={best['auc']:.4f}, probe_acc={best['probe_acc']:.4f}, "
            f"md_cav_cos={best['md_cav_cos']:.4f})"
        )
        return best["layer"], best

    for b in info["best_5_by_auc"]:
        if b["layer"] == explicit_layer:
            logger.info(
                f"Using requested layer {explicit_layer} for '{_CONTRAST_NAME}' "
                f"(AUC={b['auc']:.4f}, probe_acc={b['probe_acc']:.4f}, "
                f"md_cav_cos={b['md_cav_cos']:.4f})"
            )
            return explicit_layer, b

    import pandas as pd
    eval_csv = _DISCOVERY_DIR / "evaluation" / "vector_evaluation.csv"
    df = pd.read_csv(eval_csv)
    row = df[(df["contrast"] == _CONTRAST_NAME) & (df["layer_key"] == explicit_layer)]
    if row.empty:
        raise ValueError(
            f"Layer '{explicit_layer}' not found for contrast '{_CONTRAST_NAME}' in {eval_csv}"
        )
    r = row.iloc[0]
    best = {
        "layer": explicit_layer,
        "auc": float(r["projection_auc"]),
        "probe_acc": float(r["probe_accuracy_cv"]),
        "md_cav_cos": float(r["md_cav_cosine"]),
    }
    logger.info(
        f"Using requested layer {explicit_layer} for '{_CONTRAST_NAME}' "
        f"(AUC={best['auc']:.4f}, from full evaluation CSV, not in top-5)"
    )
    return explicit_layer, best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Logit-lens lexicon derivation for the mental-workload vector."
    )
    p.add_argument(
        "--layer", default=None,
        help="Explicit layer key (e.g. language_29_D5120). "
             "Default: auto-resolve the top-AUC layer from summary.json.",
    )
    return p.parse_args()


def main() -> None:
    import numpy as np
    from transformers import AutoTokenizer
    from control.III1_vector_control.logit_lens import (
        load_unembedding, project_to_vocab, top_tokens, derive_lexicon,
    )

    args = parse_args()
    layer, best = _resolve_layer(args.layer)
    vec_path = _MD_VECTORS_DIR / f"{_CONTRAST_NAME}.npy"
    vec = np.load(vec_path, allow_pickle=True).item()[layer]
    vector_norm = float(np.linalg.norm(vec))

    logger.info(f"Loading final norm + lm_head (~2GB, CPU) from {_DEFAULT_MODEL_PATH}")
    norm_w, lm_head = load_unembedding(_DEFAULT_MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(_DEFAULT_MODEL_PATH)

    deltas = project_to_vocab(vec, norm_w, lm_head)
    pos_tokens, neg_tokens = top_tokens(deltas, tok, k=_TOP_K)
    pos_words, neg_words = derive_lexicon(deltas, tok, k=200, max_words=30)

    layer_slug = layer.replace("_D5120", "").replace("_D1408", "")
    resolution = "auto-resolved by top MD-AUC" if args.layer is None else "explicitly requested"
    out = {
        "name": f"workload_{layer_slug}",
        "description": (
            f"Mental-workload direction (overwhelming - minimal), blank prompt, "
            f"{resolution} to {layer} (AUC={best['auc']:.4f}) — "
            f"see mental_workload/summary.json."
        ),
        "source": str(vec_path),
        "vector_norm": vector_norm,
        "top_positive_tokens": [{"token": t, "delta_logit": s} for t, s in pos_tokens],
        "top_negative_tokens": [{"token": t, "delta_logit": s} for t, s in neg_tokens],
        "lexicon": {"positive_words": pos_words, "negative_words": neg_words},
    }

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"workload_{layer_slug}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved → {out_path}")
    print(f"\n✅ Workload logit-lens lexicon complete -> {out_path}")


if __name__ == "__main__":
    main()
