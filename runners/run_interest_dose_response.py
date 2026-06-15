"""
Runner: TIER IV Step 1 -- wide alpha-sweep dose-response for the
blank-derived interestingness vector (v_interest_blank) @ language_29_D5120,
on the blank (no-persona) prompt.

Motivation: Branch B's primary (+-2 / ablate) showed a clean, monotonic
Delta=0.247 rating shift. Before picking an alpha for TIER IV's
gradient-matching target (README Sec 9.1 step 1), sweep a much wider alpha
range -- mirroring Branch C2's design for the (null) country vector -- to find
where v_interest_blank's effect saturates and/or where JSON parsing starts to
break down.

Measures (per alpha):
    - rating-distribution (mean score, dose-response curve)
    - lexicon-count shift (high/low-interest words, from
      results/extra_checks/logit_lens/interest_blank_language_29.json)
    - JSON-parse failure rate (rating == "?"), sanity check for large alpha

Usage
-----
Dry-run (check paths, no model load):
    python runners/run_interest_dose_response.py --dry_run

Full run (defaults: language_29_D5120, n=100, alphas -8 -4 -2 -1 0 1 2 4 8):
    python runners/run_interest_dose_response.py
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
_DEFAULT_INTEREST_VECTOR = (
    "results/representation_discovery/interestingness/md_vectors/"
    "blank_interest_high_vs_blank_interest_low.npy"
)
_DEFAULT_LEXICON_JSON = "results/extra_checks/logit_lens/interest_blank_language_29.json"
_DEFAULT_ALPHAS = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TIER IV Step 1: wide alpha-sweep dose-response for "
                     "v_interest_blank on the blank prompt."
    )
    p.add_argument(
        "--vector_path", default=_DEFAULT_INTEREST_VECTOR,
        help="Path to the .npy mean-difference vector dict (default: blank interest contrast).",
    )
    p.add_argument(
        "--layer", default="language_29_D5120",
        help="Target layer for injection (default: language_29_D5120).",
    )
    p.add_argument(
        "--lexicon_json", default=_DEFAULT_LEXICON_JSON,
        help="Path to logit-lens evidence JSON with a 'lexicon' field.",
    )
    p.add_argument(
        "--n_images", type=int, default=100,
        help="Number of images per alpha (default: 100).",
    )
    p.add_argument(
        "--alphas", nargs="+", type=float, default=_DEFAULT_ALPHAS,
        help="Alpha values for the sweep (default: -8 -4 -2 -1 0 1 2 4 8).",
    )
    p.add_argument(
        "--manifest", default=None,
        help="Path to image manifest .pkl (defaults to data/selected_uniform_500_manifest.pkl).",
    )
    p.add_argument(
        "--model_path", default=_DEFAULT_MODEL_PATH,
        help="Path to HuggingFace model directory.",
    )
    p.add_argument(
        "--offload_suffix", default="interest_dose_response",
        help="Suffix for CPU offload directory (default: interest_dose_response).",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Directory for results (default: results/interest_validation/<layer>/dose_response_blank).",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print config and validate paths/vector/lexicon without loading the model.",
    )
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    layer_slug = args.layer.replace("_D5120", "").replace("_D1408", "")
    return Path("results/interest_validation") / layer_slug / "dose_response_blank"


def main() -> None:
    args = parse_args()
    out_dir = _resolve_output_dir(args)

    import numpy as np
    from representation.III1_vector_control.lexicon_validation import load_lexicon

    if args.dry_run:
        print(f"Vector path:  {args.vector_path}")
        print(f"Layer:        {args.layer}")
        print(f"Lexicon JSON: {args.lexicon_json}")
        print(f"Alphas:       {args.alphas}")
        print(f"n_images:     {args.n_images}  -> {out_dir}")
        print(f"Total new generations: {len(args.alphas)} x {args.n_images} = "
              f"{len(args.alphas) * args.n_images}")

        vec = np.load(args.vector_path, allow_pickle=True).item()
        v = vec[args.layer]
        print(f"Vector @ {args.layer}: shape={v.shape}, norm={float(np.linalg.norm(v)):.4f}")

        pos_words, neg_words = load_lexicon(args.lexicon_json)
        print(f"Lexicon: {len(pos_words)} positive words, {len(neg_words)} negative words")

        import pandas as pd
        manifest_path = Path(args.manifest) if args.manifest else \
                        Path("data/selected_uniform_500_manifest.pkl")
        manifest_df = pd.read_pickle(manifest_path)
        print(f"Manifest: {len(manifest_df)} images @ {manifest_path}")

        from utils.prompt_builder import build_blank_prompt
        print(f"Blank prompt:\n{build_blank_prompt()}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load vector + lexicon ────────────────────────────────────────────────
    from representation.III1_vector_control.control import (
        run_control_experiment, summarise_and_plot,
    )
    from representation.III1_vector_control.lexicon_validation import lexicon_count_by_alpha

    logger.info(f"Loading v_interest_blank @ {args.layer}")
    vec = np.load(args.vector_path, allow_pickle=True).item()
    vector = vec[args.layer].astype(np.float32)
    vector = vector / np.linalg.norm(vector)

    lexicon = load_lexicon(args.lexicon_json)
    logger.info(f"Lexicon: {len(lexicon[0])} positive / {len(lexicon[1])} negative words")

    # ── Load images ───────────────────────────────────────────────────────────
    import pandas as pd
    manifest_path = Path(args.manifest) if args.manifest else \
                    Path("data/selected_uniform_500_manifest.pkl")
    manifest_df = pd.read_pickle(manifest_path)
    images = manifest_df[["filename", "img_path"]].head(args.n_images).to_dict(orient="records")
    logger.info(f"Images: {len(images)} (from {manifest_path})")

    # ── Load model ────────────────────────────────────────────────────────────
    from utils.model_loader import load_model_and_processor
    from utils.hpc import get_offload_dir, setup_cuda_env
    from utils.prompt_builder import build_blank_prompt
    setup_cuda_env()
    offload = get_offload_dir(
        base=Path(args.model_path).parent / "offload_dir",
        suffix=args.offload_suffix,
    )
    logger.info("Loading model...")
    model, processor = load_model_and_processor(
        args.model_path, offload_dir=str(offload) if offload else None
    )

    # ── Wide alpha sweep on blank prompt ────────────────────────────────────
    logger.info(f"=== Blank-prompt dose-response, {args.layer}, "
                f"alphas={args.alphas}, n={len(images)} ===")
    results = run_control_experiment(
        model=model, processor=processor, images=images,
        prompt=build_blank_prompt(), vector=vector, layer_key=args.layer,
        alphas=sorted(args.alphas), out_dir=out_dir,
    )

    logger.info("Summarising rating dose-response...")
    summarise_and_plot(results, baseline_alpha=0.0, out_dir=out_dir)

    logger.info("Summarising lexicon-count dose-response...")
    lexicon_count_by_alpha(results, lexicon, out_path=out_dir / "lexicon_diff_by_alpha.csv")

    # Parse-failure diagnostic: fraction of "?" (unparseable) ratings per alpha.
    df = pd.DataFrame([{"alpha": r.alpha, "rating": r.rating} for r in results])
    parse_fail = df.groupby("alpha")["rating"].apply(lambda s: (s == "?").mean()).rename("parse_fail_rate")
    print("\nJSON-parse failure rate by alpha (sanity check for large |alpha|):")
    print(parse_fail.to_string())
    parse_fail.to_csv(out_dir / "parse_fail_rate_by_alpha.csv")

    print(f"\n✅ TIER IV Step 1 (interest dose-response) complete -> {out_dir}")


if __name__ == "__main__":
    main()
