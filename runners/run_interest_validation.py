"""
Runner: Branch B blank-prompt interestingness-vector validation (Option 1).

PRIMARY (confound-free): blank (no-persona) prompt, 4 conditions at a target
layer (default language_29_D5120, v_interest_blank direction):
    no_inject     — read from existing data/results_blank_activations.npy
    inject_pos2   — h' = h + 2*v  ("toward high interest")
    inject_neg2   — h' = h - 2*v  ("toward low interest")
    ablate        — h' = h - (h.v_hat) v_hat

Primary measure: rating-distribution shift (1-5 score) — the SAME axis the
vector was derived from. Secondary measure: logit-lens-derived lexicon-count
shift in the generated explanation (see
results/extra_checks/logit_lens/interest_blank_language_29.json).

SECONDARY (generalisation, persona prompt): does the BLANK-derived
(persona-agnostic) vector also shift ratings under a PERSONA prompt?
Default: male_awe_extended_germany — per the Option 2 concept-alignment
analysis (results/representation_discovery/interestingness/analytics/), this
persona's OWN interest direction is the MOST entangled with persona-specific
(emotion) features of the 7 candidates, making it the hardest test case for
transfer of a persona-agnostic vector. alpha sweep -2..2, rating distribution
+ lexicon-count shift.

Usage
-----
Dry-run (check paths, no model load):
    python runners/run_interest_validation.py --dry_run

Full run (defaults: language_29_D5120, n=100 each, male_awe_extended_germany alphas -2..2):
    python runners/run_interest_validation.py
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
_SECONDARY_ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Branch B: blank-prompt interestingness-vector validation (primary) "
                     "+ persona-prompt alpha sweep (secondary, generalisation)."
    )
    p.add_argument(
        "--vector_path", default=_DEFAULT_INTEREST_VECTOR,
        help="Path to the .npy mean-difference vector dict (default: blank interest contrast).",
    )
    p.add_argument(
        "--layer", default="language_29_D5120",
        help="Target layer for injection/ablation (default: language_29_D5120).",
    )
    p.add_argument(
        "--lexicon_json", default=_DEFAULT_LEXICON_JSON,
        help="Path to logit-lens evidence JSON with a 'lexicon' field "
             "(default: interest_blank_language_29.json).",
    )
    p.add_argument(
        "--n_images_primary", type=int, default=100,
        help="Number of images for the blank-prompt primary experiment (default: 100).",
    )
    p.add_argument(
        "--n_images_secondary", type=int, default=100,
        help="Number of images for the persona-prompt secondary sweep (default: 100).",
    )
    p.add_argument(
        "--secondary_persona", default="male_awe_extended_germany",
        help="Persona condition (from get_all_conditions('extended_germany')) for the "
             "secondary alpha sweep (default: male_awe_extended_germany).",
    )
    p.add_argument(
        "--secondary_alphas", nargs="+", type=float, default=_SECONDARY_ALPHAS,
        help="Alpha values for the secondary sweep (default: -2 -1 0 1 2).",
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
        "--offload_suffix", default="interest_validation",
        help="Suffix for CPU offload directory (default: interest_validation).",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Directory for results (default: results/interest_validation/<layer>).",
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
    return Path("results/interest_validation") / layer_slug


def main() -> None:
    args = parse_args()
    out_dir = _resolve_output_dir(args)
    primary_dir = out_dir / "primary"
    secondary_dir = out_dir / "secondary" / args.secondary_persona

    from representation.III1_vector_control.lexicon_validation import load_lexicon

    if args.dry_run:
        print(f"Vector path:         {args.vector_path}")
        print(f"Layer:               {args.layer}")
        print(f"Lexicon JSON:        {args.lexicon_json}")
        print(f"Primary n_images:    {args.n_images_primary}  -> {primary_dir}")
        print(f"Secondary persona:   {args.secondary_persona}")
        print(f"Secondary alphas:    {args.secondary_alphas}")
        print(f"Secondary n_images:  {args.n_images_secondary}  -> {secondary_dir}")
        print(f"Blank activations:   {args.blank_activations}")

        import numpy as np
        vec = np.load(args.vector_path, allow_pickle=True).item()
        if args.layer not in vec:
            raise KeyError(f"{args.layer!r} not found in {args.vector_path} (keys: {list(vec)[:3]}...)")
        v = vec[args.layer]
        print(f"Vector @ {args.layer}: shape={v.shape}, norm={float(np.linalg.norm(v)):.4f}")

        pos_words, neg_words = load_lexicon(args.lexicon_json)
        print(f"Lexicon: {len(pos_words)} positive words, {len(neg_words)} negative words")
        print(f"  positive: {pos_words}")
        print(f"  negative: {neg_words}")

        import pandas as pd
        manifest_path = Path(args.manifest) if args.manifest else \
                        Path("data/selected_uniform_500_manifest.pkl")
        manifest_df = pd.read_pickle(manifest_path)
        print(f"Manifest: {len(manifest_df)} images @ {manifest_path}")

        baseline = np.load(args.blank_activations, allow_pickle=True).item()
        print(f"Blank baseline: {len(baseline['results'])} results @ {args.blank_activations}")

        from runners.experiment_definitions import get_all_conditions
        conditions = get_all_conditions("extended_germany")
        if args.secondary_persona not in conditions:
            raise KeyError(
                f"Unknown persona {args.secondary_persona!r}. "
                f"Sample valid keys: {sorted(conditions)[:5]}..."
            )
        print(f"Secondary persona '{args.secondary_persona}' OK "
              f"({len(conditions)} conditions in extended_germany)")
        return

    primary_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)

    # ── Load vector + lexicon ────────────────────────────────────────────────
    import numpy as np
    from representation.III1_vector_control.lexicon_validation import (
        load_lexicon, summarise_lexicon_results, lexicon_count_by_alpha,
    )
    from representation.III1_vector_control.blank_validation import (
        run_blank_conditions, load_blank_baseline,
    )
    from representation.III1_vector_control.control import (
        run_control_experiment, summarise_and_plot,
    )

    logger.info(f"Loading interest vector from {args.vector_path} @ {args.layer}")
    vec_dict = np.load(args.vector_path, allow_pickle=True).item()
    vector = vec_dict[args.layer].astype(np.float32)
    vector = vector / np.linalg.norm(vector)

    lexicon = load_lexicon(args.lexicon_json)
    logger.info(f"Lexicon: {len(lexicon[0])} positive / {len(lexicon[1])} negative words")

    # ── Load images ───────────────────────────────────────────────────────────
    import pandas as pd
    manifest_path = Path(args.manifest) if args.manifest else \
                    Path("data/selected_uniform_500_manifest.pkl")
    manifest_df = pd.read_pickle(manifest_path)
    images_primary = manifest_df[["filename", "img_path"]].head(args.n_images_primary).to_dict(orient="records")
    images_secondary = manifest_df[["filename", "img_path"]].head(args.n_images_secondary).to_dict(orient="records")
    logger.info(f"Primary images: {len(images_primary)} | Secondary images: {len(images_secondary)} (from {manifest_path})")

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

    # ── PRIMARY: blank-prompt injection/ablation ────────────────────────────────
    logger.info(f"=== PRIMARY: blank prompt, {args.layer}, n={len(images_primary)} ===")
    primary_results = run_blank_conditions(
        model=model, processor=processor, images=images_primary,
        vector=vector, layer_key=args.layer, out_dir=primary_dir,
    )
    baseline_results = load_blank_baseline(images_primary, blank_path=args.blank_activations)
    all_primary = baseline_results + primary_results

    logger.info("Summarising primary (blank-prompt) results...")
    summarise_lexicon_results(
        all_primary, lexicon=lexicon, out_dir=primary_dir,
        title="Blank-prompt interest-condition summary",
    )

    # ── SECONDARY: persona-prompt alpha sweep (generalisation) ─────────────────
    logger.info(f"=== SECONDARY: {args.secondary_persona}, {args.layer}, "
                f"alphas={args.secondary_alphas}, n={len(images_secondary)} ===")
    from runners.experiment_definitions import get_all_conditions
    from utils.prompt_builder import build_persona_prompt
    conditions = get_all_conditions("extended_germany")
    if args.secondary_persona not in conditions:
        logger.error(f"Unknown persona '{args.secondary_persona}'. Valid: {sorted(conditions)}")
        sys.exit(1)
    secondary_prompt = build_persona_prompt(conditions[args.secondary_persona])

    secondary_results = run_control_experiment(
        model=model, processor=processor, images=images_secondary,
        prompt=secondary_prompt, vector=vector, layer_key=args.layer,
        alphas=sorted(args.secondary_alphas), out_dir=secondary_dir,
    )

    logger.info("Summarising secondary (persona-prompt) results [generalisation test]...")
    summarise_and_plot(secondary_results, baseline_alpha=0.0, out_dir=secondary_dir)

    # Exploratory: lexicon-count shift by alpha in the persona-prompt condition
    # [generalisation: does the blank-derived vector shift word usage under a persona prompt?]
    lexicon_count_by_alpha(secondary_results, lexicon, out_path=secondary_dir / "lexicon_diff_by_alpha.csv")

    print(f"\n✅ Branch B interest validation complete -> {out_dir}")


if __name__ == "__main__":
    main()
