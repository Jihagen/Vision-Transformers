"""
Runner: Branch D -- causal injection validation for all 8 discovered emotion
vectors @ language_29_D5120 (Branch B's proven-causal layer), on the blank
(no-persona) prompt.

This is the last remaining Tier I/II direction without a Tier III causal
test. Branches A (gender), B (interest), C/C2 (Nigeria country) already
covered the persona-track and interestingness-track vectors; Branch D covers
the 8 base_emotion one-vs-rest vectors (anger, amusement, awe, contentment,
disgust, excitement, fear, sad).

Design (mirrors Branch B/C primary, Branch C2 scale):
    For each emotion vector, 3 new conditions on the blank prompt:
        inject_pos2  -- h' = h + 2*v
        inject_neg2  -- h' = h - 2*v
        ablate       -- h' = h - (h.v_hat) v_hat
    no_inject is reused from the existing 500-sample blank collection
    (data/results_blank_activations.npy), as in Branch B/C.

Primary measure: interestingness rating (1-5 score) -- the SAME outcome
variable used for every Tier III branch. Predicted direction, from the
valence split established geometrically (no arousal/stress framing -- see
README Sec 7.3/8.4):
    positive-valence emotions (awe, excitement, amusement, contentment):
        predict inject_pos2 > inject_neg2 (same direction as Branch B)
    negative-valence emotions (anger, sad, fear, disgust):
        predict inject_pos2 < inject_neg2

8 emotions x 3 conditions x n_images = 2400 new generations (n_images=100).

Usage
-----
Dry-run (check paths, no model load):
    python runners/run_emotion_validation.py --dry_run

Full run (defaults: language_29_D5120, n=100):
    python runners/run_emotion_validation.py
"""
from __future__ import annotations
import argparse
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
_DEFAULT_EMOTION_MD_DIR = "results/representation_discovery/base_emotion/md_vectors"

# valence per emotion, established geometrically in README Sec 7.3/8.4
# (NOT an arousal/stress axis -- purely positive/negative valence)
EMOTIONS_VALENCE: dict[str, str] = {
    "awe":         "+",
    "excitement":  "+",
    "amusement":   "+",
    "contentment": "+",
    "anger":       "-",
    "sad":         "-",
    "fear":        "-",
    "disgust":     "-",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Branch D: blank-prompt causal injection validation for "
                     "all 8 base_emotion vectors @ a single layer."
    )
    p.add_argument(
        "--emotion_md_dir", default=_DEFAULT_EMOTION_MD_DIR,
        help="Directory of 'avg_<emotion>_vs_avg_rest_<emotion>.npy' contrast vectors.",
    )
    p.add_argument(
        "--layer", default="language_29_D5120",
        help="Target layer for injection/ablation (default: language_29_D5120).",
    )
    p.add_argument(
        "--n_images", type=int, default=100,
        help="Number of images per condition, per emotion (default: 100).",
    )
    p.add_argument(
        "--manifest", default=None,
        help="Path to image manifest .pkl (defaults to data/selected_uniform_500_manifest.pkl).",
    )
    p.add_argument(
        "--blank_activations", default="data/results_blank_activations.npy",
        help="Path to existing blank-condition activations (no_inject baseline).",
    )
    p.add_argument(
        "--model_path", default=_DEFAULT_MODEL_PATH,
        help="Path to HuggingFace model directory.",
    )
    p.add_argument(
        "--offload_suffix", default="emotion_validation",
        help="Suffix for CPU offload directory (default: emotion_validation).",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Directory for results (default: results/emotion_validation/<layer>).",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print config and validate paths/vectors without loading the model.",
    )
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    layer_slug = args.layer.replace("_D5120", "").replace("_D1408", "")
    return Path("results/emotion_validation") / layer_slug


def main() -> None:
    args = parse_args()
    out_dir = _resolve_output_dir(args)
    emotion_md_dir = Path(args.emotion_md_dir)

    import numpy as np
    import pandas as pd

    if args.dry_run:
        print(f"Emotion md_vectors dir: {emotion_md_dir}")
        print(f"Layer:        {args.layer}")
        print(f"n_images:     {args.n_images}  -> {out_dir}")
        print(f"Total new generations: {len(EMOTIONS_VALENCE)} x 3 x {args.n_images} = "
              f"{len(EMOTIONS_VALENCE) * 3 * args.n_images}")
        for emotion, valence in EMOTIONS_VALENCE.items():
            vec_path = emotion_md_dir / f"avg_{emotion}_vs_avg_rest_{emotion}.npy"
            d = np.load(vec_path, allow_pickle=True).item()
            v = d[args.layer]
            print(f"  {emotion:12s} valence={valence}  norm={float(np.linalg.norm(v)):.4f}  "
                  f"shape={v.shape}")

        manifest_path = Path(args.manifest) if args.manifest else \
                        Path("data/selected_uniform_500_manifest.pkl")
        manifest_df = pd.read_pickle(manifest_path)
        print(f"Manifest: {len(manifest_df)} images @ {manifest_path}")

        baseline = np.load(args.blank_activations, allow_pickle=True).item()
        print(f"Blank baseline: {len(baseline['results'])} results @ {args.blank_activations}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    from representation.III1_vector_control.blank_validation import (
        run_blank_conditions, load_blank_baseline,
    )

    # ── Load images ───────────────────────────────────────────────────────────
    manifest_path = Path(args.manifest) if args.manifest else \
                    Path("data/selected_uniform_500_manifest.pkl")
    manifest_df = pd.read_pickle(manifest_path)
    images = manifest_df[["filename", "img_path"]].head(args.n_images).to_dict(orient="records")
    logger.info(f"Images: {len(images)} (from {manifest_path})")

    # ── no_inject baseline (shared across all emotions) ─────────────────────
    no_inject_results = load_blank_baseline(images, blank_path=args.blank_activations)

    # ── Load model ────────────────────────────────────────────────────────────
    from utils.model_loader import load_model_and_processor
    from utils.hpc import get_offload_dir, setup_cuda_env
    setup_cuda_env()
    offload = get_offload_dir(
        base=Path(args.model_path).parent / "offload_dir",
        suffix=args.offload_suffix,
    )
    logger.info("Loading model...")
    model, processor = load_model_and_processor(
        args.model_path, offload_dir=str(offload) if offload else None
    )

    # ── Per-emotion blank-prompt injection/ablation ──────────────────────────
    combined_rows = []
    for emotion, valence in EMOTIONS_VALENCE.items():
        vec_path = emotion_md_dir / f"avg_{emotion}_vs_avg_rest_{emotion}.npy"
        vec_dict = np.load(vec_path, allow_pickle=True).item()
        vector = vec_dict[args.layer].astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        emotion_out_dir = out_dir / emotion
        emotion_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"=== {emotion} (valence={valence}) @ {args.layer}, n={len(images)} ===")
        cond_results = run_blank_conditions(
            model=model, processor=processor, images=images,
            vector=vector, layer_key=args.layer, out_dir=emotion_out_dir,
        )

        df = pd.DataFrame([
            {"filename": r.filename, "condition": r.condition, "score": r.score, "rating": r.rating}
            for r in (no_inject_results + cond_results)
        ])
        summary = (df.groupby("condition")["score"]
                   .agg(mean_score="mean", std_score="std", n="count")
                   .reset_index().round(4))
        summary.to_csv(emotion_out_dir / "condition_summary.csv", index=False)
        print(f"\n{emotion} (valence={valence}) condition summary:")
        print(summary.to_string(index=False))

        by_cond = summary.set_index("condition")["mean_score"]
        combined_rows.append({
            "emotion": emotion,
            "valence": valence,
            "no_inject": by_cond.get("no_inject", float("nan")),
            "inject_neg2": by_cond.get("inject_neg2", float("nan")),
            "inject_pos2": by_cond.get("inject_pos2", float("nan")),
            "ablate": by_cond.get("ablate", float("nan")),
            "delta_pos_minus_neg": by_cond.get("inject_pos2", float("nan")) - by_cond.get("inject_neg2", float("nan")),
        })

    combined = pd.DataFrame(combined_rows).round(4)
    combined.to_csv(out_dir / "all_emotions_summary.csv", index=False)
    print("\n=== Branch D: all-emotions summary "
          "(predicted: delta>0 for '+' valence, delta<0 for '-' valence) ===")
    print(combined.to_string(index=False))

    print(f"\n✅ Branch D emotion validation complete -> {out_dir}")


if __name__ == "__main__":
    main()
